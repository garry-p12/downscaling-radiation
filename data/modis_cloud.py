"""MODIS 1 km cloud state -> daily 10 km cloud fraction, added to the store.

Decoding (MOD09GA/MYD09GA state_1km, bits 0-1, per the AppEEARS quality
service): 0 clear -> 0, 1 cloudy -> 1, 2 mixed -> 0.5, 3 "not set, assumed
clear" -> 0. Fill / no-observation -> NaN.

Aggregation is the same area-weighted binning used for NSRDB
(common/regrid.binning_matrix), computed once from the 1 km lat/lon grid.
A 10 km cell is NaN on a day when less than ``min_coverage`` of its 1 km
pixels were observed (swath gaps), rather than a fraction of a fraction.

Outputs three Track B features (dyn_* in the store, picked up by
features/builder.py):
    modis_cf_terra   ~10:30 local overpass
    modis_cf_aqua    ~13:30 local (from 2002-07-04; NaN before)
    modis_cf         mean of the available overpasses

Usage:
    python -m data.modis_cloud aggregate      # raw AppEEARS nc -> fine-grid nc per (sat, year)
    python -m data.modis_cloud add-to-store   # append dyn_modis_cf* to the zarr store
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from common import build_target_grid, load_config
from common import regrid as rg

MIN_COVERAGE = 0.5
FILL_VALUES = (65535,)


def decode_state(v: np.ndarray) -> np.ndarray:
    """uint16 state_1km -> cloud score in [0, 1], NaN where not observed."""
    v = np.asarray(v)
    bad = ~np.isfinite(v) if v.dtype.kind == "f" else np.zeros(v.shape, bool)
    vi = np.where(bad, 0, v).astype("int64")
    for f in FILL_VALUES:
        bad |= vi == f
    lut = np.array([0.0, 1.0, 0.5, 0.0])
    out = lut[vi & 3]
    out[bad] = np.nan
    return out.astype("float32")


def _layer(ds: xr.Dataset) -> xr.DataArray:
    names = [k for k in ds.data_vars if "state_1km" in k]
    if not names:
        raise KeyError(f"no state_1km variable in {list(ds.data_vars)}")
    da = ds[names[0]]
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in da.dims}
    return da.rename(ren)


def _raw_dir(cfg) -> Path:
    return Path(cfg["sources"]["dynamic"]["modis_cf"].get(
        "raw_dir", Path(cfg["store_root"]) / "raw" / "modis"))


def aggregate_file(path: Path, grid, W=None, chunk: int = 32):
    """One AppEEARS netCDF -> (times, fine (T, ny, nx) float32, W)."""
    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=True)
    da = _layer(ds)
    if W is None:
        la, lo = np.meshgrid(da.lat.values, da.lon.values, indexing="ij")
        tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
        px, py = tf.transform(lo.ravel(), la.ravel())
        W, _ = rg.binning_matrix(px, py, grid, pt_weight=np.cos(np.deg2rad(la.ravel())))
    T = da.sizes["time"]
    out = np.full((T, *grid.shape), np.nan, dtype="float32")
    for s in range(0, T, chunk):
        v = decode_state(da.isel(time=slice(s, s + chunk)).values).reshape(
            min(chunk, T - s), -1)
        frac = rg.apply(W, v, out_shape=grid.shape)
        cov = rg.apply(W, np.isfinite(v).astype("float64"), out_shape=grid.shape)
        out[s:s + chunk] = np.where(cov >= MIN_COVERAGE, frac, np.nan)
    # AppEEARS writes a Julian-calendar cftime axis; go through ISO dates.
    times = pd.to_datetime([str(t)[:10] for t in da.time.values])
    return times, out, W


def aggregate(cfg) -> None:
    raw = _raw_dir(cfg)
    fine = raw / "fine"
    fine.mkdir(exist_ok=True)
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    W = None
    for d in sorted(p for p in raw.iterdir() if p.is_dir() and (p / "DONE").exists()):
        dest = fine / f"modis_cf_{d.name}.nc"
        if dest.exists():
            continue
        parts = []
        for f in sorted(d.glob("*.nc")):
            times, arr, W = aggregate_file(f, grid, W)
            parts.append(xr.DataArray(arr, dims=("time", "y", "x"),
                                      coords=dict(time=times, y=grid.y, x=grid.x)))
        if not parts:
            continue
        da = xr.concat(parts, "time").sortby("time")
        da = da.groupby("time").mean()                       # duplicate days, if any
        da.name = "modis_cf"
        da.to_netcdf(dest)
        print(f"[modis] {dest.name}: {da.sizes['time']} days, "
              f"NaN {100 * float(np.isnan(da.values).mean()):.1f}%", flush=True)


def add_to_store(cfg) -> None:
    store = cfg["dataset"]["out_zarr"]
    ds = xr.open_zarr(store)
    times = pd.DatetimeIndex(ds.time.values)
    fine = _raw_dir(cfg) / "fine"
    new = {}
    for sat in ("terra", "aqua"):
        files = sorted(fine.glob(f"modis_cf_{sat}_*.nc"))
        if not files:
            print(f"[modis] no {sat} files; skipped")
            continue
        da = xr.concat([xr.open_dataarray(f) for f in files], "time").sortby("time")
        new[f"dyn_modis_cf_{sat}"] = da.reindex(time=times).astype("float32")
    if not new:
        raise SystemExit("[modis] nothing to add")
    stack = np.stack([v.values for v in new.values()])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    new["dyn_modis_cf"] = xr.DataArray(mean.astype("float32"), dims=("time", "y", "x"))
    out = xr.Dataset({k: (("time", "y", "x"), v.values if hasattr(v, "values") else v)
                      for k, v in new.items()},
                     coords=dict(time=ds.time.values, y=ds.y.values, x=ds.x.values))
    out = out.chunk({"time": ds.chunks["time"][0]})
    from data.build_store import append_to_store

    append_to_store(out.drop_vars(["time", "y", "x"]), store)
    for k, v in new.items():
        arr = v.values if hasattr(v, "values") else v
        print(f"[modis] {k}: NaN {100 * float(np.isnan(arr).mean()):.1f}%", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["aggregate", "add-to-store"])
    a = ap.parse_args()
    c = load_config("data")
    aggregate(c) if a.cmd == "aggregate" else add_to_store(c)

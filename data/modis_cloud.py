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


def _fine_dir(cfg) -> Path:
    """Where the per-grid fractions go: beside the raw tiles by default, or a
    per-grid directory (``fine_dir``) when several grids share one raw download."""
    d = cfg["sources"]["dynamic"]["modis_cf"].get("fine_dir")
    return Path(d) if d else _raw_dir(cfg) / "fine"


def aggregate_file(path: Path, grid, W=None, chunk: int = 32, mode: str = "bin"):
    """One AppEEARS netCDF -> (times, fine (T, ny, nx) float32, W)."""
    ds = xr.open_dataset(path, mask_and_scale=False, decode_times=True)
    da = _layer(ds)
    if W is None:
        la, lo = np.meshgrid(da.lat.values, da.lon.values, indexing="ij")
        tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
        px, py = tf.transform(lo.ravel(), la.ravel())
        W, _ = rg.point_matrix(px, py, grid, mode, pt_weight=np.cos(np.deg2rad(la.ravel())))
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
    fine = _fine_dir(cfg)
    fine.mkdir(parents=True, exist_ok=True)
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    W = None
    mode = cfg["grid"].get("point_remap", "bin")
    y0, y1 = int(cfg["time"]["train"][0][:4]), int(cfg["time"]["test"][1][:4])
    for d in sorted(p for p in raw.iterdir() if p.is_dir() and (p / "DONE").exists()):
        dest = fine / f"modis_cf_{d.name}.nc"
        if dest.exists() or not y0 <= int(d.name.rsplit("_", 1)[-1]) <= y1:
            continue
        parts = []
        for f in sorted(d.glob("*.nc")):
            times, arr, W = aggregate_file(f, grid, W, mode=mode)
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
    """Terra, Aqua and their mean as dyn_modis_cf_*. One satellite in memory at
    a time and the mean computed lazily chunk by chunk from the store: at 5 km
    the stacked pair over 2001-2021 does not fit a login node."""
    from data.build_store import append_to_store

    store = cfg["dataset"]["out_zarr"]
    ds = xr.open_zarr(store)
    times = pd.DatetimeIndex(ds.time.values)
    fine = _fine_dir(cfg)
    chunks = {"time": ds.chunks["time"][0]}
    have = []
    for sat in ("terra", "aqua"):
        files = sorted(fine.glob(f"modis_cf_{sat}_*.nc"))
        if not files:
            print(f"[modis] no {sat} files; skipped")
            continue
        da = xr.concat([xr.open_dataarray(f) for f in files], "time").sortby("time")
        v = da.reindex(time=times).values.astype("float32")
        name = f"dyn_modis_cf_{sat}"
        append_to_store(xr.Dataset({name: (("time", "y", "x"), v)}).chunk(chunks), store)
        print(f"[modis] {name}: NaN {100 * float(np.isnan(v).mean()):.1f}%", flush=True)
        have.append(name)
        del v, da
    if not have:
        raise SystemExit("[modis] nothing to add")
    st = xr.open_zarr(store)
    pair = [st[k] for k in have]
    n = sum(p.notnull() for p in pair)
    mean = (sum(p.fillna(0) for p in pair) / n.where(n > 0)).astype("float32")
    append_to_store(xr.Dataset({"dyn_modis_cf": mean.drop_vars(
        [c for c in mean.coords if c in ("time", "y", "x")])}), store)
    print("[modis] dyn_modis_cf: mean of " + " + ".join(have), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["aggregate", "add-to-store"])
    a = ap.parse_args()
    c = load_config("data")
    aggregate(c) if a.cmd == "aggregate" else add_to_store(c)

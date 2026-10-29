"""Assemble POWER + NSRDB + statics (+ Track B dynamics) into one Zarr store.

Layout (all physical units; nothing normalised — scalers are fit later, on
training years and training pixels only, by whoever consumes the store):

    power_<name>  (time, clat, clon)   coarse POWER, kWh/m2/day or native
    nsrdb_ghi     (time, y, x)         target, kWh/m2/day, local solar day
    nsrdb_count   (y, x)               NSRDB points averaged into each cell
    dyn_<name>    (time, y, x)         Track B fine dynamics (optional)
    <static>      (y, x)               dem, slope, svf, coastal_dist, lc_*, ...
    horizon       (az, y, x)           horizon angles, rad
    shade_beam    (doy, y, x)          daily unblocked beam fraction (B2)
    lat, lon      (y, x)
    valid         (y, x)               land and enough NSRDB points
    holdout       (y, x)               Austin box
    exclusion     (y, x)               holdout + buffer: never trained on
    split         (time)               train / val / test / none

The NSRDB remap is a cached area-weighted binning matrix (common/regrid.py),
not bilinear (plan 3.3).

Usage:
    python -m data.build_store
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from common import build_target_grid, load_config
from common import regrid as rg
from common.splits import assign_splits, holdout_masks


def nsrdb_to_grid(da: xr.DataArray, grid, cache_dir: str | None,
                  min_points: int = 1, mode: str = "bin"):
    """(time, point) NSRDB daily totals -> (time, y, x), plus point counts.

    ``mode`` (config grid.point_remap): 'bin' averages the points in each cell
    (10 and 5 km); 'linear' interpolates them (1 km, finer than the 4 km points)."""
    lat, lon = da["lat"].values, da["lon"].values
    tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    px, py = tf.transform(lon, lat)
    key = rg.grid_key(lat, lon, grid.x, grid.y, [min_points])

    def build():
        W, _ = rg.point_matrix(px, py, grid, mode, pt_weight=np.cos(np.deg2rad(lat)),
                               min_count=min_points)
        return W

    tag = "bin" if mode == "bin" else f"bin_{mode}"
    W = rg.cached(Path(cache_dir) / f"nsrdb_{tag}_{key}.npz", build) if cache_dir \
        else build()
    _, counts = rg.point_matrix(px, py, grid, mode)
    # In blocks of days: the whole float64 stack does not fit a login node at 5 km.
    out = np.empty((da.sizes["time"], *grid.shape), "float32")
    step = int(np.clip(5e7 // np.prod(grid.shape), 1, 500))
    for s in range(0, da.sizes["time"], step):
        blk = da.isel(time=slice(s, s + step)).values
        out[s:s + len(blk)] = rg.apply(W, blk, out_shape=grid.shape)
    return out, counts


def regrid_dynamic(da: xr.DataArray, grid, lat2d, lon2d) -> np.ndarray:
    """A (time, lat, lon) stack onto the fine grid.

    Finer-than-target sources (MODIS 1 km) are binned (area-weighted mean);
    if that leaves empty cells the source is too coarse, and bilinear is used.
    """
    da = da.rename({k: v for k, v in (("latitude", "lat"), ("longitude", "lon"))
                    if k in da.dims}).sortby("lat").sortby("lon")
    la, lo = np.meshgrid(da.lat.values, da.lon.values, indexing="ij")
    tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    px, py = tf.transform(lo.ravel(), la.ravel())
    W, counts = rg.binning_matrix(px, py, grid, pt_weight=np.cos(np.deg2rad(la.ravel())))
    vals = da.values.reshape(da.sizes["time"], -1)
    if (counts == 0).any():
        W = rg.bilinear_matrix(da.lat.values, da.lon.values, lat2d, lon2d)
    return rg.apply(W, vals, out_shape=grid.shape).astype("float32")


def assemble(cfg, grid, power: xr.Dataset, target: np.ndarray, target_times,
             counts: np.ndarray, statics: dict, dynamic: dict | None = None,
             shade_beam: np.ndarray | None = None) -> xr.Dataset:
    """Pure assembly: aligned inputs -> the store Dataset. Used by the real
    build and by data/synthetic.py, so the synthetic smoke run exercises the
    same layout the real store has."""
    lat2d, lon2d = grid.latlon()
    tt = pd.DatetimeIndex(target_times).normalize()
    pt = pd.DatetimeIndex(power.time.values).normalize()
    lo = pd.Timestamp(cfg["time"]["train"][0])
    hi = pd.Timestamp(cfg["time"]["test"][1])
    times = tt.intersection(pt)
    times = times[(times >= lo) & (times <= hi)]
    if times.empty:
        raise ValueError("POWER and NSRDB share no days inside the configured window")
    ti = tt.get_indexer(times)
    power = power.assign_coords(time=pt).sel(time=times)

    hold, excl = holdout_masks(lat2d, lon2d, cfg.get("holdout"))
    land = statics.get("land_mask", np.ones(grid.shape)) > 0.5
    min_pts = int(cfg["dataset"].get("min_nsrdb_points", 1))
    valid = land & (counts >= min_pts)

    yx = ("y", "x")
    data = {k: (("time", "clat", "clon"), v.values.astype("float32"))
            for k, v in power.data_vars.items()}
    data["nsrdb_ghi"] = (("time",) + yx, target[ti].astype("float32"))
    data["nsrdb_count"] = (yx, counts.astype("int16"))
    for k, v in (dynamic or {}).items():
        data[f"dyn_{k}"] = (("time",) + yx, v[ti].astype("float32"))
    for k, v in statics.items():
        if k == "horizon":
            data["horizon"] = (("az",) + yx, v.astype("float32"))
        else:
            data[k] = (yx, np.asarray(v, dtype="float32"))
    if shade_beam is not None:
        data["shade_beam"] = (("doy",) + yx, shade_beam.astype("float32"))
    data["lat"] = (yx, lat2d.astype("float32"))
    data["lon"] = (yx, lon2d.astype("float32"))
    data["valid"] = (yx, valid.astype("int8"))
    data["holdout"] = (yx, hold.astype("int8"))
    data["exclusion"] = (yx, excl.astype("int8"))
    data["split"] = (("time",), assign_splits(times, cfg["time"]))

    ds = xr.Dataset(data, coords=dict(time=times.values, y=grid.y, x=grid.x,
                                      clat=power.clat.values, clon=power.clon.values))
    ds.attrs.update(crs=grid.crs, res_m=float(grid.res_m),
                    config=json.dumps(dict(cfg), default=str))
    return ds


def write(ds: xr.Dataset, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    enc_chunks = {"time": min(366, ds.sizes["time"])}
    ds = ds.chunk({k: v for k, v in enc_chunks.items() if k in ds.dims})
    ds.to_zarr(path, mode="w")
    print(f"[store] wrote {path}: {dict(ds.sizes)}")


def append_to_store(ds: xr.Dataset, store: str) -> None:
    """Append variables to the zarr store with ONE thread.

    Login nodes cap per-user threads; zarr's async pool plus dask's threaded
    scheduler hit "RuntimeError: can't start new thread" on vista2
    (2026-10-22). mode="a" overwrites a partially written variable, so a
    failed append can simply be re-run.
    """
    import dask
    import zarr

    with dask.config.set(scheduler="synchronous"), \
            zarr.config.set({"threading.max_workers": 1, "async.concurrency": 4}):
        ds.to_zarr(store, mode="a")


def open_store(cfg=None, path: str | None = None) -> xr.Dataset:
    cfg = cfg or load_config("data")
    return xr.open_zarr(path or cfg["dataset"]["out_zarr"])


def main() -> None:
    from data.download_nsrdb import open_nsrdb
    from data.download_power import open_power
    from data.statics import build_statics
    from data.terrain import beam_shading

    cfg = load_config("data")
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    lat2d, lon2d = grid.latlon()
    print(f"[store] fine grid {grid.shape} at {grid.res_m:.0f} m")

    power = open_power(cfg)
    nsrdb = open_nsrdb(cfg, "ghi")
    # only the configured window (assemble keeps no more): at 1 km every day is ~1.4 GB
    nsrdb = nsrdb.sel(time=slice(cfg["time"]["train"][0], cfg["time"]["test"][1]))
    target, counts = nsrdb_to_grid(nsrdb, grid, cfg["dataset"].get("cache_dir"),
                                   int(cfg["dataset"].get("min_nsrdb_points", 1)),
                                   cfg["grid"].get("point_remap", "bin"))
    statics = build_statics(cfg, grid)
    pre = cfg["sources"]["statics"].get("precomputed")
    if pre and Path(pre).exists() and "shade_beam" in np.load(pre).files:
        shade = np.load(pre)["shade_beam"]
    else:
        shade = beam_shading(statics["horizon"], lat2d, lon2d) if "horizon" in statics else None

    dynamic = {}
    for name, spec in (cfg["sources"].get("dynamic") or {}).items():
        if not spec or not spec.get("path"):
            continue
        da = xr.open_mfdataset(spec["path"])[spec["variable"]].load() * float(spec.get("scale", 1))
        da = da.reindex(time=pd.DatetimeIndex(nsrdb.time.values))
        dynamic[name] = regrid_dynamic(da, grid, lat2d, lon2d)
    if not dynamic:
        warnings.warn("[store] no Track B dynamic sources configured; Track B "
                      "will use POWER PW/AOD/cloud only")

    ds = assemble(cfg, grid, power, target, nsrdb.time.values, counts, statics,
                  dynamic, shade)
    write(ds, cfg["dataset"]["out_zarr"])


if __name__ == "__main__":
    main()

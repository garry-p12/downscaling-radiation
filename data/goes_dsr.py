"""GOES-16 ABI Downward Shortwave Radiation (DSR) -> daily GHI, a THIRD reference.

Purpose: an independent satellite radiation estimate for the station check,
not a training target. NOAA's operational DSR algorithm (hybrid lookup-table
retrieval at 0.25 deg, hourly) shares GOES with NSRDB but not NSRDB's cloud
products or its PSM radiative transfer, so where DSR and NSRDB disagree at a
station, the station says which one is right. Available only from 2019-12-05,
so it covers the test years (2020-21) and nothing earlier.

Source: s3://noaa-goes16/ABI-L2-DSRC, one CONUS file per hour, regular
lat/lon grid. Each file's DSR is stamped at its scan midpoint, rounded to the
hour, bilinearly regridded to 10 km (0.25 deg -> 10 km: interpolation, no new
detail) and integrated to local-solar-day totals by common/daily.daily_totals
with ``stamp="instant"`` like the hourly station data. Days missing more than
10% of daylight hours are rejected.

Usage (vista2):
    python -m data.goes_dsr fetch --start 2019-12-05 --end 2021-12-31 --workers 8
    python -m data.goes_dsr add-to-store       # writes dsr_ghi
"""
from __future__ import annotations

import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import build_target_grid, load_config
from common import regrid as rg
from common.daily import daily_totals

BUCKET = "noaa-goes16/ABI-L2-DSRC"


def _root(cfg) -> Path:
    d = Path(cfg["store_root"]) / "raw" / "goes_dsr"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hour(fs, t: pd.Timestamp, box):
    """Domain subset of the DSR file for UTC hour t, or None."""
    try:
        keys = fs.ls(f"{BUCKET}/{t:%Y}/{t.dayofyear:03d}/{t:%H}/")
    except FileNotFoundError:
        return None
    for k in keys:
        try:
            ds = xr.open_dataset(io.BytesIO(fs.cat(k)), engine="h5netcdf")
            ds = ds.sortby("lat").sortby("lon").sel(lat=slice(box[0], box[1]),
                                                    lon=slice(box[2], box[3]))
            v = ds["DSR"].where(ds["DQF"] == 0).values.astype("float32")
            return ds.lat.values, ds.lon.values, v
        except Exception:  # noqa: BLE001  (a corrupt file is a missing hour)
            continue
    return None


def fetch(cfg, start: str, end: str, workers: int) -> None:
    """One npz per month of hourly domain subsets (small: ~40 x 45 cells)."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    d, pad = cfg["domain"], 0.5
    box = (d["lat_min"] - pad, d["lat_max"] + pad, d["lon_min"] - pad, d["lon_max"] + pad)
    for m0 in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        out = _root(cfg) / f"dsr_{m0:%Y%m}.npz"
        if out.exists():
            continue
        hours = pd.date_range(m0, m0 + pd.offsets.MonthBegin(1), freq="h", inclusive="left")
        with ThreadPoolExecutor(workers) as ex:
            got = list(ex.map(lambda t: _hour(fs, t, box), hours))
        ok = [g for g in got if g is not None]
        if not ok:
            print(f"[dsr] {m0:%Y-%m}: no data", flush=True)
            continue
        lat, lon = ok[0][0], ok[0][1]
        cube = np.full((len(hours), lat.size, lon.size), np.nan, "float32")
        for i, g in enumerate(got):
            if g is not None and g[2].shape == cube.shape[1:]:
                cube[i] = g[2]
        np.savez_compressed(out, time=hours.values, lat=lat, lon=lon, dsr=cube)
        print(f"[dsr] {m0:%Y-%m}: {len(ok)}/{len(hours)} hours", flush=True)


def gap_fill(cube: np.ndarray, times: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             max_gap_h: int = 3) -> np.ndarray:
    """Fill DSR gaps in instantaneous-clearness space, on the 0.25 deg grid.

    DSR is not retrieved at low sun (solar zenith beyond ~80 deg) or where its
    quality flag is set, and daily_totals rejects a day with >10% of daylight
    missing. So: kt_inst = DSR / (E0 cos z); gaps of up to ``max_gap_h`` hours
    are filled by linear interpolation in time (ends held constant), and the
    sun-down hours are exactly zero. Longer gaps stay missing.
    """
    from common.solar import cos_zenith, extraterrestrial_normal

    T = len(times)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cz = cos_zenith(times, LA.ravel(), LO.ravel())                  # (T, N)
    toa_h = extraterrestrial_normal(times)[:, None] * np.clip(cz, 0, None)
    v = cube.reshape(T, -1).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = np.where(toa_h > 1.0, v / toa_h, np.nan)
    kt = pd.DataFrame(kt).interpolate(limit=max_gap_h, limit_direction="both",
                                      limit_area=None).to_numpy()
    out = np.where(cz <= 0, 0.0, kt * toa_h)
    return out.reshape(cube.shape)


def add_to_store(cfg) -> None:
    files = sorted(_root(cfg).glob("dsr_*.npz"))
    if not files:
        raise SystemExit("no DSR files")
    z = [np.load(f) for f in files]
    lat, lon = z[0]["lat"], z[0]["lon"]
    times = np.concatenate([a["time"] for a in z]).astype("datetime64[ns]")
    cube = np.concatenate([a["dsr"] for a in z])
    # Each file is the hour's scan (~HH:01-HH:04 UTC): instantaneous at HH:02,
    # treated as HH:00 -- a 2-minute shift, far inside the hourly sampling.
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    glat, glon = grid.latlon()
    W = rg.bilinear_matrix(lat, lon, glat, glon).tocsr()
    src = gap_fill(cube, times, lat, lon).reshape(len(times), -1)
    # In blocks of grid cells: the whole (T, ny*nx) float64 stack does not fit
    # the login node's memory limit.
    gl, go = glat.ravel(), glon.ravel()
    parts, days = [], None
    for k in range(0, gl.size, 1000):
        sl = slice(k, k + 1000)
        fine = rg.apply(W[sl], src).astype("float32")                # (T, block)
        days, tot, _ = daily_totals(fine, times, gl[sl], go[sl], stamp="instant",
                                    site_chunk=256)
        parts.append(tot.astype("float32"))
    tot = np.concatenate(parts, axis=1)
    da = xr.DataArray(tot.reshape(len(days), *grid.shape).astype("float32"),
                      dims=("time", "y", "x"),
                      coords=dict(time=days.astype("datetime64[ns]")))
    store = cfg["dataset"]["out_zarr"]
    st = xr.open_zarr(store)
    da = da.reindex(time=st.time.values)
    out = xr.Dataset({"dsr_ghi": (("time", "y", "x"), da.values)}).chunk(
        {"time": st.chunks["time"][0]})
    from data.build_store import append_to_store

    append_to_store(out, store)
    test = da.sel(time=slice("2020", "2021")).values
    print(f"[dsr] dsr_ghi added: NaN in 2020-21 {100 * float(np.isnan(test).mean()):.1f}%",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--start", default="2019-12-05")
    f.add_argument("--end", default="2021-12-31")
    f.add_argument("--workers", type=int, default=8)
    sub.add_parser("add-to-store")
    a = ap.parse_args()
    c = load_config("data")
    if a.cmd == "fetch":
        fetch(c, a.start, a.end, a.workers)
    else:
        import dask

        with dask.config.set(scheduler="synchronous"):
            add_to_store(c)

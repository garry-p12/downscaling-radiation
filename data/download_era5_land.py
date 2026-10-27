"""ERA5-Land surface solar radiation (ssrd, 0.1 deg) — the SECOND TARGET.

Purpose: a target-swap experiment, not a production target. The plan rules
ERA5-Land out as the target (2.1: its radiation is ERA5 forcing interpolated,
~31 km of information on a 9 km grid). That is exactly why it is useful here:
its errors come from a reanalysis cloud scheme, not a satellite retrieval, so
if models trained on it inherit ITS station errors the way NSRDB-trained models
inherit NSRDB's (eval: per-target mimicry at USCRN), the target sets the
ceiling.

Accumulation convention (ERA5-Land hourly): ``ssrd`` at hour h is the energy
accumulated since 00 UTC of that day, and the 00 UTC value holds the WHOLE
previous day. Hourly energy for the hour ending at t is therefore
    E(01) = A(01);  E(h) = A(h) - A(h-1), h = 02..23;  E(00, D+1) = A(00, D+1) - A(23, D)
converted to a mean flux E/3600 W/m2 stamped at the hour END, then integrated
to local-solar-day totals by common/daily.daily_totals like every other source.

Requests are one month each (a full hourly year over CDS times out), split
across processes by month with --months; every request writes its own file.

Usage (vista2):
    python -m data.download_era5_land fetch --years 2001 ... 2021 --months 1 2 3 4
    python -m data.download_era5_land add-to-store
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import build_target_grid, load_config
from common import regrid as rg
from common.daily import daily_totals

DATASET = "reanalysis-era5-land"
VAR = "surface_solar_radiation_downwards"


def _dir(cfg) -> Path:
    d = Path(cfg["store_root"]) / "raw" / "era5_land"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(cfg, years, months) -> None:
    import cdsapi

    d, pad = cfg["domain"], 0.3
    area = [d["lat_max"] + pad, d["lon_min"] - pad, d["lat_min"] - pad, d["lon_max"] + pad]
    client = cdsapi.Client()
    failed = 0
    for y in years:
        for m in months:
            out = _dir(cfg) / f"era5l_ssrd_{y}{m:02d}.nc"
            if out.exists() and out.stat().st_size > 0:
                continue
            req = {"variable": [VAR], "year": str(y), "month": [f"{m:02d}"],
                   "day": [f"{k:02d}" for k in range(1, 32)],
                   "time": [f"{h:02d}:00" for h in range(24)],
                   "area": area, "data_format": "netcdf", "download_format": "unarchived"}
            try:
                client.retrieve(DATASET, req, str(out) + ".part")
                Path(str(out) + ".part").rename(out)
                print(f"[era5l] {out.name}", flush=True)
            except Exception as e:  # noqa: BLE001 - keep going; re-run retries
                print(f"[era5l] {y}-{m:02d} FAILED: {e}", flush=True)
                failed += 1
    print(f"[era5l] fetch done, {failed} failed", flush=True)


def hourly_flux(acc: xr.DataArray) -> xr.DataArray:
    """De-accumulate ssrd (J/m2 since 00 UTC) -> mean W/m2 over the hour ENDING
    at each stamp. Works across month files once they are concatenated."""
    a = acc.values.astype("float64")
    t = pd.DatetimeIndex(acc.time.values)
    e = np.full_like(a, np.nan)
    prev = np.vstack([np.full((1, *a.shape[1:]), np.nan), a[:-1]])
    hour = t.hour.values
    e[hour == 1] = a[hour == 1]
    mid = (hour >= 2) & (hour <= 23)
    e[mid] = a[mid] - prev[mid]
    e[hour == 0] = a[hour == 0] - prev[hour == 0]    # needs 23 UTC of the previous day
    e = np.clip(e, 0, None) / 3600.0
    return acc.copy(data=e.astype("float32"))


def _open(files) -> xr.DataArray:
    ds = xr.open_mfdataset(files, combine="by_coords")
    ren = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon"),
                             ("valid_time", "time")) if a in ds.variables}
    ds = ds.rename(ren).sortby("lat").sortby("lon").sortby("time")
    name = [k for k in ds.data_vars if k in ("ssrd", VAR)][0]
    return ds[name]


def daily_year(files_by_month: dict, year: int):
    """Local-solar-day totals (days of ``year`` only) on the ERA5-Land grid.

    One year at a time: all 21 years of hourly data at once is ~14 GB. The
    previous December rides along so 00 UTC on Jan 1 (which closes Dec 31's
    accumulation) and the solar-day boundary both have their neighbours.
    """
    keys = [k for k in sorted(files_by_month) if k[0] == year or k == (year - 1, 12)
            or k == (year + 1, 1)]
    acc = _open([files_by_month[k] for k in keys]).load()
    flux = hourly_flux(acc)
    lat, lon = flux.lat.values, flux.lon.values
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    days, tot, _ = daily_totals(flux.values.reshape(flux.sizes["time"], -1),
                                flux.time.values, LA.ravel(), LO.ravel(), stamp="end")
    d = pd.DatetimeIndex(days.astype("datetime64[ns]"))
    keep = d.year == year
    return d[keep], tot[keep], lat, lon


def add_to_store(cfg) -> None:
    files = sorted(_dir(cfg).glob("era5l_ssrd_*.nc"))
    if not files:
        raise SystemExit("no ERA5-Land files")
    by_month = {(int(f.stem[-6:-2]), int(f.stem[-2:])): f for f in files}
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    glat, glon = grid.latlon()
    W, parts = None, []
    for year in sorted({y for y, _ in by_month}):
        days, tot, lat, lon = daily_year(by_month, year)
        if W is None:
            W = rg.bilinear_matrix(lat, lon, glat, glon)   # 0.1 deg -> 10 km: same scale
        fine = rg.apply(W, tot, out_shape=grid.shape).astype("float32")
        parts.append(xr.DataArray(fine, dims=("time", "y", "x"), coords=dict(time=days)))
        print(f"[era5l] {year}: {len(days)} days", flush=True)
    da = xr.concat(parts, "time")

    store = cfg["dataset"]["out_zarr"]
    st = xr.open_zarr(store)
    da = da.reindex(time=st.time.values)
    out = xr.Dataset({"era5l_ghi": (("time", "y", "x"), da.values)}) \
        .chunk({"time": st.chunks["time"][0]})
    from data.build_store import append_to_store

    append_to_store(out, store)
    print(f"[era5l] era5l_ghi added: NaN {100 * float(np.isnan(da.values).mean()):.1f}%",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--years", type=int, nargs="+", required=True)
    f.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    sub.add_parser("add-to-store")
    a = ap.parse_args()
    c = load_config("data")
    if a.cmd == "fetch":
        fetch(c, a.years, a.months)
    else:
        # Single-threaded throughout: dask's threaded scheduler reading the
        # monthly files hit the login node's per-user thread cap
        # ("can't start new thread", vista2, 2026-10-23).
        import dask

        with dask.config.set(scheduler="synchronous"):
            add_to_store(c)

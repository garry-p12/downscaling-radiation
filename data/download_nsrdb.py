"""Download NSRDB GHI over the domain and aggregate to daily totals.

Source: NSRDB GOES aggregated v4.0.0 on the NREL public bucket, read
anonymously with h5py over s3fs — no API key, no HSDS server. Layout
(verified 2026-10 on nsrdb_2020.h5):

    time_index  (17568,)            bytes, '2020-01-01 00:00:00+00:00', 30 min UTC
    meta        (2018267,)          latitude, longitude, elevation, timezone, ...
    ghi         (17568, 2018267)    uint16 W/m2, psm_scale_factor 1, chunks (2000, 500)
    fill_flag   (17568, 2018267)    uint8, percent of native samples gap-filled

Only the domain's columns are read. Sites are grouped into contiguous gid runs
(the file is ordered by longitude column) so each HDF5 read is a slab, not a
fancy-index scatter. Each batch of sites is aggregated to LOCAL-SOLAR-DAY
totals by common/daily.daily_totals before the next batch is read, so memory
stays at one batch (~0.3 GB) rather than the domain-year (~4 GB).

Output: one netCDF per (variable, year) under sources.nsrdb.out_dir with
dims (time, point) in kWh/m2/day, plus point lat/lon. build_store.py does the
conservative remap onto the 10 km grid.

Usage:
    python -m data.download_nsrdb --years 2001 2002 ...
    python -m data.download_nsrdb --years 2020 --uri /local/nsrdb_{year}.h5
"""
from __future__ import annotations

import argparse
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config
from common.daily import daily_totals


def _open(uri: str):
    import h5py

    if uri.startswith("s3://"):
        import s3fs

        fs = s3fs.S3FileSystem(anon=True)
        fh = fs.open(uri, "rb", block_size=2 ** 22)
        return h5py.File(fh, "r")
    return h5py.File(uri, "r")


def select_sites(lat: np.ndarray, lon: np.ndarray, domain: dict, pad: float):
    m = ((lat >= domain["lat_min"] - pad) & (lat <= domain["lat_max"] + pad)
         & (lon >= domain["lon_min"] - pad) & (lon <= domain["lon_max"] + pad))
    return np.nonzero(m)[0]


def contiguous_runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Sorted indices -> [(start, stop)) runs of consecutive integers."""
    if idx.size == 0:
        return []
    br = np.nonzero(np.diff(idx) != 1)[0]
    starts = np.r_[idx[0], idx[br + 1]]
    stops = np.r_[idx[br] + 1, idx[-1] + 1]
    return list(zip(starts.tolist(), stops.tolist()))


def batch_runs(runs, max_sites: int):
    """Group runs into batches of at most ``max_sites`` sites (runs split)."""
    batch, n = [], 0
    for a, b in runs:
        while a < b:
            take = min(b - a, max_sites - n)
            batch.append((a, a + take))
            n += take
            a += take
            if n >= max_sites:
                yield batch
                batch, n = [], 0
    if batch:
        yield batch


def decode_times(raw) -> np.ndarray:
    s = pd.to_datetime([t.decode() if isinstance(t, bytes) else t for t in raw],
                       utc=True)
    return s.tz_convert(None).values.astype("datetime64[ns]")


def read_variable(f, var: str, runs, drop_at: float | None) -> np.ndarray:
    """(T, n_sites) float32 physical values for the given gid runs; NaN = bad.

    ``drop_at``: samples whose fill_flag (percent gap-filled, aggregated
    product) is >= this are set to NaN; None keeps all.
    """
    ds = f[var]
    scale = float(ds.attrs.get("psm_scale_factor", ds.attrs.get("scale_factor", 1.0)))
    vmax = float(ds.attrs.get("physical_max", np.inf))
    vmin = float(ds.attrs.get("physical_min", -np.inf))
    parts = [ds[:, a:b] for a, b in runs]
    v = np.concatenate(parts, axis=1).astype("float32") / scale
    v[(v > vmax) | (v < vmin)] = np.nan
    if drop_at is not None and "fill_flag" in f:
        flag = np.concatenate([f["fill_flag"][:, a:b] for a, b in runs], axis=1)
        v[flag >= drop_at] = np.nan
    return v


def process_year(cfg, year: int, uri: str | None = None,
                 max_sites: int = 4000) -> list[Path]:
    import xarray as xr

    src = cfg["sources"]["nsrdb"]
    uri = (uri or src["uri"]).format(year=year)
    out_dir = Path(src["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    variables = list(src.get("variables", ["ghi"]))
    targets = {v: out_dir / f"nsrdb_{v}_{year}.nc" for v in variables}
    todo = [v for v, p in targets.items() if not p.exists()]
    if not todo:
        print(f"[nsrdb] {year}: exists, skipping")
        return list(targets.values())

    t0 = _time.time()
    f = _open(uri)
    gid_cache = out_dir / "nsrdb_domain_sites.npz"
    if gid_cache.exists():
        z = np.load(gid_cache)
        gids, lat, lon = z["gid"], z["lat"], z["lon"]
    else:
        meta = f["meta"]
        mlat = meta["latitude"]
        mlon = meta["longitude"]
        gids = select_sites(mlat, mlon, cfg["domain"], float(src.get("pad_deg", 0.25)))
        lat, lon = mlat[gids].astype("float64"), mlon[gids].astype("float64")
        np.savez(gid_cache, gid=gids, lat=lat, lon=lon)
    times = decode_times(f["time_index"][:])
    print(f"[nsrdb] {year}: {gids.size} sites, {times.size} steps")

    runs = contiguous_runs(gids)
    year_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").values
    for var in todo:
        cols = []
        pos = 0
        for batch in batch_runs(runs, max_sites):
            n = sum(b - a for a, b in batch)
            v = read_variable(f, var, batch, src.get("fill_flag_drop_at", 100))
            days, tot, _ = daily_totals(v, times, lat[pos:pos + n], lon[pos:pos + n],
                                        stamp=src.get("stamp", "instant"))
            s = pd.DataFrame(tot, index=pd.DatetimeIndex(days.astype("datetime64[ns]")))
            cols.append(s.reindex(year_days).to_numpy("float32"))
            pos += n
            print(f"[nsrdb] {year} {var}: {pos}/{gids.size} sites "
                  f"({_time.time() - t0:.0f}s)", flush=True)
        data = np.concatenate(cols, axis=1)
        da = xr.DataArray(data, dims=("time", "point"),
                          coords=dict(time=year_days, lat=("point", lat),
                                      lon=("point", lon), gid=("point", gids)),
                          name=f"nsrdb_{var}",
                          attrs={"units": "kWh/m2/day",
                                 "day": "local solar day (common/daily.py)",
                                 "source": uri})
        da.to_netcdf(targets[var])
        print(f"[nsrdb] wrote {targets[var]} "
              f"(NaN {100 * np.isnan(data).mean():.2f}%)")
    return list(targets.values())


def open_nsrdb(cfg, var: str = "ghi"):
    """Concatenate the per-year point files: (time, point) DataArray."""
    import xarray as xr

    files = sorted(Path(cfg["sources"]["nsrdb"]["out_dir"]).glob(f"nsrdb_{var}_*.nc"))
    if not files:
        raise FileNotFoundError("no NSRDB files; run python -m data.download_nsrdb")
    return xr.concat([xr.open_dataarray(p).load() for p in files], dim="time")


def _one(args):
    cfg, year, uri = args
    try:
        process_year(cfg, year, uri)
        return None
    except Exception as e:  # noqa: BLE001
        print(f"[nsrdb] {year} FAILED: {e!r}", flush=True)
        return year


def main(years: list[int], uri: str | None, workers: int = 1) -> None:
    """Years in parallel PROCESSES (each its own HDF5/S3 handle). Each year
    writes its own file and skips if present, so a re-run resumes."""
    from concurrent.futures import ProcessPoolExecutor

    cfg = dict(load_config("data"))
    with ProcessPoolExecutor(max(1, workers)) as ex:
        failed = [y for y in ex.map(_one, [(cfg, y, uri) for y in years]) if y]
    if failed:
        raise SystemExit(f"[nsrdb] failed years {failed} — re-run to retry")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--uri", default=None, help="override sources.nsrdb.uri; "
                    "'{year}' is substituted")
    ap.add_argument("--workers", type=int, default=1, help="years in parallel")
    a = ap.parse_args()
    main(a.years, a.uri, a.workers)

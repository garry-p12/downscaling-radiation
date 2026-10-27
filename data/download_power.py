"""Download the coarse predictors: NASA POWER daily solar fields — no credentials.

Adapted from downscaling-temperature data/download_power.py (same tiling and
retry logic, same file naming). Differences for radiation:

  * several parameters (GHI, clear-sky GHI, kt, TOA, diffuse, albedo, cloud,
    PW, AOD) — POWER serves one per request;
  * ``time-standard=LST``: daily totals over the LOCAL SOLAR day, the same day
    definition used for NSRDB and stations (common/daily.py). The temperature
    pipeline used UTC, which for irradiance would splice two afternoons;
  * units are read from each file and converted to kWh/m2/day (plan 3.2);
  * the native grid is read back and checked: solar parameters are CERES
    SYN1deg at 1 x 1 deg, coarser than POWER meteorology (0.5 x 0.625).

Verified 2026-10 against a live POWER v10.0.0 response: ALLSKY_SFC_SW_DWN,
CLRSKY_SFC_SW_DWN, TOA_SW_DWN, ALLSKY_SFC_SW_DIFF in kW-hr/m^2/day;
ALLSKY_KT, ALLSKY_SRF_ALB, AOD_55 dimensionless; CLOUD_AMT in %; PW in cm;
lat/lon centres at x.5 deg.

Usage:
    python -m data.download_power --years 2001 2002 ...   # all configured params
    python -m data.download_power --years 2020 --only ghi kt
"""
from __future__ import annotations

import argparse
import functools
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import numpy as np

from common import load_config
from common.units import is_dimensionless, to_kwh_day

BASE = "https://power.larc.nasa.gov/api/temporal/daily/regional"
MAX_SPAN = 4.0      # deg; service cap is 4.5
MIN_SPAN = 2.0      # deg; service rejects smaller boxes
FILL = -999.0
# Store variables that are irradiation (converted to kWh/m2/day).
IRRADIATION = {"ghi", "ghi_clr", "toa", "dhi", "dni", "lw"}


def _edges(lo: float, hi: float) -> list[tuple[float, float]]:
    span = hi - lo
    n = max(1, int(np.ceil(span / MAX_SPAN)))
    step = span / n
    out = []
    for i in range(n):
        a, b = lo + i * step, lo + (i + 1) * step
        if b - a < MIN_SPAN:
            c = 0.5 * (a + b)
            a, b = c - MIN_SPAN / 2, c + MIN_SPAN / 2
        out.append((round(a, 4), round(b, 4)))
    return out


def tiles(domain: dict, pad: float) -> list[dict]:
    lat_lo, lat_hi = domain["lat_min"] - pad, domain["lat_max"] + pad
    lon_lo, lon_hi = domain["lon_min"] - pad, domain["lon_max"] + pad
    return [dict(lat_min=a, lat_max=b, lon_min=c, lon_max=d)
            for a, b in _edges(lat_lo, lat_hi) for c, d in _edges(lon_lo, lon_hi)]


def _url(box: dict, year: int, parameter: str, community: str,
         time_standard: str) -> str:
    q = {
        "parameters": parameter,
        "community": community,
        "format": "netcdf",
        "time-standard": time_standard,
        "latitude-min": box["lat_min"],
        "latitude-max": box["lat_max"],
        "longitude-min": box["lon_min"],
        "longitude-max": box["lon_max"],
        "start": f"{year}0101",
        "end": f"{year}1231",
    }
    return BASE + "?" + "&".join(f"{k}={v}" for k, v in q.items())


def fetch(url: str, dest: Path, retries: int = 8, backoff: float = 10.0):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                payload = resp.read()
            if not payload.startswith(b"\x89HDF") and not payload.startswith(b"CDF"):
                raise ValueError(f"not netCDF: {payload[:200]!r}")
            dest.write_bytes(payload)
            print(f"[power] {dest.name} ok ({len(payload) / 1e6:.1f} MB)", flush=True)
            return dest
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"[power] {dest.name} attempt {attempt}/{retries} failed: {e}", flush=True)
            if attempt < retries:
                # 429 = POWER's per-client rate limit: honour Retry-After, else
                # back off exponentially (10, 20, 40 ... s, capped at 5 min).
                wait = min(300.0, backoff * 2 ** (attempt - 1))
                if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    if ra and ra.isdigit():
                        wait = max(wait, float(ra))
                time.sleep(wait)
    print(f"[power] GIVING UP on {dest.name}\n         url: {url}")
    return None


def _fname(parameter: str, name: str, year: int, tile: int) -> str:
    return f"power_{parameter}_{name}_{year}_t{tile:02d}.nc"


# --------------------------------------------------------------------------- #
# Read back
# --------------------------------------------------------------------------- #
def open_parameter(out_dir: str | Path, parameter: str):
    """Stitch one parameter's (tile, year) files into a (time, lat, lon) DataArray.

    combine_first per year, then concat in time — NOT xr.merge(compat=
    'override'), which silently keeps one tile (see the temperature repo).
    """
    import xarray as xr

    files = sorted(Path(out_dir).glob(f"power_{parameter}_*.nc"))
    if not files:
        raise FileNotFoundError(f"no POWER files for {parameter} in {out_dir}; "
                                f"run python -m data.download_power")
    by_year: dict[str, list] = {}
    for f in files:
        by_year.setdefault(f.stem.split("_")[-2], []).append(f)
    years = []
    for y in sorted(by_year):
        parts = [xr.open_dataset(f)[parameter].load() for f in by_year[y]]
        years.append(functools.reduce(lambda a, b: a.combine_first(b), parts))
    da = xr.concat(years, dim="time") if len(years) > 1 else years[0]
    rename = {a: b for a, b in (("latitude", "lat"), ("longitude", "lon"))
              if a in da.coords}
    da = da.rename(rename).sortby("lat").sortby("lon").sortby("time")
    da = da.where(da > FILL / 2)
    return da


def check_grid(lat: np.ndarray, lon: np.ndarray, expected_deg: float) -> float:
    """Return the grid spacing; warn if it is not the documented solar grid."""
    dlat = float(np.median(np.diff(lat))) if lat.size > 1 else expected_deg
    dlon = float(np.median(np.diff(lon))) if lon.size > 1 else expected_deg
    if not (np.isclose(dlat, expected_deg) and np.isclose(dlon, expected_deg)):
        warnings.warn(f"[power] grid is {dlat} x {dlon} deg, config says "
                      f"{expected_deg}; check grid.coarse_res_deg", RuntimeWarning)
    return dlat


def open_power(cfg, names: list[str] | None = None):
    """All configured POWER parameters as one Dataset in store units.

    Irradiation -> kWh/m2/day, CLOUD_AMT % -> fraction, others as served.
    Variables are named ``power_<storename>``.
    """
    import xarray as xr

    src = cfg["sources"]["power"]
    params = dict(src["parameters"])
    names = names or list(params)
    out = {}
    for name in names:
        da = open_parameter(src["out_dir"], params[name])
        units = da.attrs.get("units", "")
        if name in IRRADIATION:
            da = da.copy(data=to_kwh_day(da.values, units))
            da.attrs["units"] = "kWh/m2/day"
        elif units.strip() == "%":
            da = da / 100.0
            da.attrs["units"] = "1"
        elif not (is_dimensionless(units) or name == "pw"):
            warnings.warn(f"[power] {name}: unexpected units {units!r}")
        out[f"power_{name}"] = da.astype("float32")
    ds = xr.Dataset(out)
    check_grid(ds.lat.values, ds.lon.values,
               float(cfg["grid"].get("coarse_res_deg", 1.0)))
    frac = {k: float(np.isnan(v.values).mean()) for k, v in ds.items()}
    bad = {k: f for k, f in frac.items() if f > 0.001}
    if bad:
        warnings.warn(f"[power] NaN fractions after stitching: {bad}")
    return ds.rename(lat="clat", lon="clon")


# --------------------------------------------------------------------------- #
def main(years: list[int], only: list[str] | None = None, workers: int = 1) -> None:
    from concurrent.futures import ThreadPoolExecutor

    cfg = load_config("data")
    src = cfg["sources"]["power"]
    out_dir = Path(src["out_dir"])
    name = cfg["domain"]["name"]
    params = {k: v for k, v in src["parameters"].items() if not only or k in only}
    boxes = tiles(cfg["domain"], float(src.get("pad_deg", 1.5)))
    jobs = [(_url(box, year, parameter, src.get("community", "RE"),
                  src.get("time_standard", "LST")),
             out_dir / _fname(parameter, name, year, i))
            for parameter in params.values() for year in years
            for i, box in enumerate(boxes)]
    print(f"[power] {len(params)} params x {len(boxes)} tiles x {len(years)} years "
          f"= {len(jobs)} requests, {workers} at a time")
    # A few concurrent requests only: the POWER API rate-limits per client.
    with ThreadPoolExecutor(max(1, workers)) as ex:
        failed = sum(r is None for r in ex.map(lambda j: fetch(*j), jobs))
    if failed:
        raise SystemExit(f"[power] {failed} request(s) failed — re-run to retry")
    print("[power] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--only", nargs="+", default=None,
                    help="store names from sources.power.parameters")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.years, a.only, a.workers)

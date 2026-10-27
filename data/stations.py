"""Ground-station GHI: download, parse, BSRN-style QC, daily totals, pixel map.

Stations are INDEPENDENT TRUTH (plan 2.1): never a training input, never a
feature, never used for early stopping or model selection.

Networks and what this domain (27-35 N, 104-94 W) actually has:
  * Oklahoma Mesonet  5-min SRAD (W/m2, interval mean, stamped at interval
                      END), pyranometer. Only stations south of 35 N fall in
                      the domain. Downloader + parser here.
  * SURFRAD           1-3 min GHI/DNI/DHI. No site lies in this domain (the
                      nearest, Goodwin Creek MS, is 34.25 N 89.87 W); the parser
                      is kept for an extended domain.
  * ARM SGP           Central Facility is at 36.6 N, north of the domain;
                      reader for ARM netCDF (e.g. radflux / qcrad VAPs) kept.
  * TexMesonet, West Texas Mesonet: no public bulk API format to target, so
                      they enter through the generic CSV:
                      station_id,time_utc,ghi[,dni,dhi][,lat,lon]

QC (Long & Dutton 2002, BSRN): physically-possible limits on every sample,
the closure test GHI ~ DNI cos z + DHI where all three components exist,
then common/daily.daily_totals drops days missing > 10% of daylight samples.

Usage:
    python -m data.stations download-mesonet --years 2019 2020 2021
    python -m data.stations build
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import build_target_grid, load_config
from common.daily import daily_totals, reindex_regular
from common.solar import cos_zenith, extraterrestrial_normal
from common.splits import holdout_masks

MTS_MISSING = -990.0          # Mesonet codes -994 .. -999 are all "missing"


# --------------------------------------------------------------------------- #
# Parsers -> DataFrame(time [UTC, naive], ghi[, dni, dhi])
# --------------------------------------------------------------------------- #
def parse_mts(text: str) -> pd.DataFrame:
    """Oklahoma Mesonet MTS file. Line 2 holds the UTC date; TIME is minutes
    after 00 UTC; SRAD is the 5-minute mean ending at TIME."""
    lines = text.splitlines()
    parts = lines[1].split()
    day = pd.Timestamp(year=int(parts[1]), month=int(parts[2]), day=int(parts[3]))
    df = pd.read_csv(io.StringIO("\n".join(lines[2:])), sep=r"\s+")
    srad = df["SRAD"].astype(float).where(lambda s: s > MTS_MISSING)
    t = day + pd.to_timedelta(df["TIME"].astype(int), unit="min")
    return pd.DataFrame({"time": t, "ghi": srad.values,
                         "station_id": df["STID"].astype(str).values})


SURFRAD_COLS = ["year", "jday", "month", "day", "hour", "min", "dt", "zen",
                "dw_solar", "qc_dwsolar", "uw_solar", "qc_uwsolar",
                "direct_n", "qc_direct_n", "diffuse", "qc_diffuse"]


def parse_surfrad(text: str) -> pd.DataFrame:
    """SURFRAD daily .dat (2 header lines, then fixed columns, -9999.9 missing,
    instantaneous/1-min samples stamped in UTC). Non-zero QC flags -> NaN."""
    rows = [ln.split() for ln in text.splitlines()[2:] if ln.strip()]
    a = pd.DataFrame([r[:len(SURFRAD_COLS)] for r in rows], columns=SURFRAD_COLS).astype(float)
    t = pd.to_datetime(dict(year=a.year, month=a.month, day=a.day, hour=a.hour,
                            minute=a["min"]))

    def col(v, q):
        x = a[v].where(a[v] > -9000)
        return x.where(a[q] == 0).values

    return pd.DataFrame({"time": t, "ghi": col("dw_solar", "qc_dwsolar"),
                         "dni": col("direct_n", "qc_direct_n"),
                         "dhi": col("diffuse", "qc_diffuse")})


def read_arm(path: str, ghi="BestEstimate_down_short_hemisp",
             dni="short_direct_normal", dhi="down_short_diffuse_hemisp") -> pd.DataFrame:
    import xarray as xr

    ds = xr.open_dataset(path)
    out = {"time": pd.DatetimeIndex(ds.time.values)}
    for k, v in (("ghi", ghi), ("dni", dni), ("dhi", dhi)):
        if v in ds:
            x = ds[v].values.astype(float)
            out[k] = np.where(x < -9000, np.nan, x)
    return pd.DataFrame(out)


def parse_uscrn(text: str) -> pd.DataFrame:
    """USCRN hourly02 (CRNH0203) file: whitespace columns, UTC_DATE/UTC_TIME
    stamp the END of the hour, SOLARAD = hourly mean global irradiance (W/m2),
    SOLARAD_FLAG != 0 or -9999 -> missing. Station lat/lon are per row."""
    rows = [ln.split() for ln in text.splitlines() if ln.strip()]
    a = pd.DataFrame([r[:15] for r in rows],
                     columns=["wban", "utc_date", "utc_time", "lst_date", "lst_time",
                              "crx", "lon", "lat", "t_calc", "t_avg", "t_max", "t_min",
                              "p", "solarad", "solarad_flag"])
    t = pd.to_datetime(a.utc_date + a.utc_time.str.zfill(4), format="%Y%m%d%H%M")
    ghi = a.solarad.astype(float)
    ghi = ghi.where((ghi > -9000) & (a.solarad_flag == "0"))
    return pd.DataFrame({"time": t, "ghi": ghi.values,
                         "lat": a.lat.astype(float).values, "lon": a.lon.astype(float).values})


def read_generic(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df.pop("time_utc"), utc=True).dt.tz_convert(None)
    return df


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def bsrn_qc(df: pd.DataFrame, lat: float, lon: float) -> tuple[pd.DataFrame, dict]:
    """Physically-possible limits + closure. Failing GHI samples -> NaN.

    Returns the cleaned frame and counts for the per-station QC report.
    """
    t = df["time"].values.astype("datetime64[ns]")
    mu = cos_zenith(t, [lat], [lon])[:, 0]
    mu0 = np.clip(mu, 0, None)
    sa = extraterrestrial_normal(t)
    ghi = df["ghi"].to_numpy(float).copy()
    present = np.isfinite(ghi)
    ppl = (ghi > -4) & (ghi < sa * 1.5 * mu0 ** 1.2 + 100)
    fail_ppl = present & ~ppl
    fail_cl = np.zeros_like(present)
    if {"dni", "dhi"} <= set(df.columns):
        dni = df["dni"].to_numpy(float)
        dhi = df["dhi"].to_numpy(float)
        bad_comp = (np.isfinite(dhi) & ((dhi < -4) | (dhi > sa * 0.95 * mu0 ** 1.2 + 50))) | \
                   (np.isfinite(dni) & ((dni < -4) | (dni > sa)))
        s = dni * mu0 + dhi
        z = np.degrees(np.arccos(np.clip(mu, -1, 1)))
        ok = np.isfinite(s) & (s > 50) & present
        ratio = np.where(ok, ghi / np.where(ok, s, 1), 1.0)
        tol = np.where(z < 75, 0.08, 0.15)
        fail_cl = ok & (z < 93) & (np.abs(ratio - 1) > tol)
        fail_cl |= present & bad_comp
    ghi[fail_ppl | fail_cl] = np.nan
    out = df.copy()
    out["ghi"] = ghi
    day = mu > 0
    rep = {"daylight_present": int((present & day).sum()),
           "fail_ppl": int((fail_ppl & day).sum()),
           "fail_closure": int((fail_cl & day).sum())}
    return out, rep


# --------------------------------------------------------------------------- #
# Daily
# --------------------------------------------------------------------------- #
def station_daily(df: pd.DataFrame, lat: float, lon: float, freq: str,
                  stamp: str, max_missing: float) -> pd.DataFrame:
    times, vals = reindex_regular(df["time"].values, df["ghi"].values, freq)
    days, tot, miss = daily_totals(vals, times, [lat], [lon], stamp=stamp,
                                   max_missing_daylight=max_missing)
    return pd.DataFrame({"date": pd.DatetimeIndex(days.astype("datetime64[ns]")),
                         "ghi_kwh": tot[:, 0], "missing_frac": miss[:, 0]})


def map_to_grid(meta: pd.DataFrame, grid, holdout_cfg) -> pd.DataFrame:
    """Add row/col of the containing fine cell and holdout membership."""
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tf.transform(meta["lon"].values, meta["lat"].values)
    col = np.floor((x - (grid.x[0] - grid.res_m / 2)) / grid.res_m).astype(int)
    row = np.floor(((grid.y[0] + grid.res_m / 2) - y) / grid.res_m).astype(int)
    inside = (row >= 0) & (row < grid.y.size) & (col >= 0) & (col < grid.x.size)
    lat2d, lon2d = grid.latlon()
    hold, excl = holdout_masks(lat2d, lon2d, holdout_cfg)
    r, c = np.clip(row, 0, grid.y.size - 1), np.clip(col, 0, grid.x.size - 1)
    out = meta.copy()
    out["row"], out["col"], out["in_grid"] = row, col, inside
    out["region"] = np.where(~inside, "outside",
                             np.where(hold[r, c], "holdout",
                                      np.where(excl[r, c], "buffer", "train_region")))
    return out


# --------------------------------------------------------------------------- #
# Oklahoma Mesonet download
# --------------------------------------------------------------------------- #
def mesonet_meta(cfg) -> pd.DataFrame:
    sc = cfg["sources"]["stations"]
    with urllib.request.urlopen(sc["mesonet_meta_url"], timeout=120) as r:
        meta = pd.read_csv(io.BytesIO(r.read()))
    meta = meta.rename(columns={"stid": "station_id", "nlat": "lat", "elon": "lon"})
    d = cfg["domain"]
    m = meta["lat"].between(d["lat_min"], d["lat_max"]) & \
        meta["lon"].between(d["lon_min"], d["lon_max"])
    meta = meta.loc[m, ["station_id", "lat", "lon", "elev"]].copy()
    meta["network"] = "ok_mesonet"
    return meta


def download_mesonet(cfg, years: list[int], delay_s: float = 1.0,
                     max_forbidden: int = 20) -> None:
    """One request at a time, ``delay_s`` apart, resumable.

    Serial on purpose: 4 parallel workers from TACC Vista tripped the
    Mesonet CloudFront rate limit after ~350 files (2026-10-22) and the IP was
    then refused (HTTP 403) on every request. ``max_forbidden`` consecutive
    403s abort the run instead of hammering a block. For multi-year bulk data,
    request it from the Oklahoma Mesonet directly.
    """
    import time
    import urllib.error

    sc = cfg["sources"]["stations"]
    root = Path(sc["out_dir"]) / "ok_mesonet"
    root.mkdir(parents=True, exist_ok=True)
    meta = mesonet_meta(cfg)
    meta.to_csv(root / "meta.csv", index=False)
    print(f"[stations] {len(meta)} Oklahoma Mesonet stations inside the domain")
    got = forbidden = 0
    for stid in meta["station_id"]:
        for day in pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31"):
            dest = root / stid / f"{day:%Y%m%d}.mts"
            if day.year not in years or dest.exists():
                continue
            url = sc["mesonet_mts_url"].format(y=f"{day:%Y}", m=f"{day:%m}",
                                               d=f"{day:%d}", stid=stid.lower())
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    body = r.read()
                forbidden = 0
                if not body.lstrip().startswith(b"<"):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                    got += 1
            except urllib.error.HTTPError as e:
                forbidden = forbidden + 1 if e.code == 403 else 0
                if forbidden >= max_forbidden:
                    raise SystemExit(f"[stations] {forbidden} consecutive HTTP 403 — "
                                     f"blocked; stopping (fetched {got})") from e
            except Exception:  # noqa: BLE001
                pass
            time.sleep(delay_s)
    print(f"[stations] fetched {got} station-days")


USCRN_BASE = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02"
USCRN_STATES = ("TX", "OK", "NM", "LA")


def download_uscrn(cfg, years: list[int]) -> None:
    """USCRN hourly02: one small file per station-year, anonymous HTTPS. Every
    TX/OK/NM/LA station is fetched; map_to_grid drops the out-of-domain ones."""
    import re
    import time

    root = Path(cfg["sources"]["stations"]["out_dir"]) / "uscrn"
    root.mkdir(parents=True, exist_ok=True)
    states = "|".join(USCRN_STATES)
    pat = re.compile(rf"CRNH0203-(\d{{4}})-((?:{states})_[A-Za-z0-9_.]+)\.txt")
    got = 0
    for y in years:
        with urllib.request.urlopen(f"{USCRN_BASE}/{y}/", timeout=120) as r:
            names = sorted(set(pat.findall(r.read().decode())))
        for _, name in names:
            dest = root / name / f"{y}.txt"
            if dest.exists():
                continue
            with urllib.request.urlopen(f"{USCRN_BASE}/{y}/CRNH0203-{y}-{name}.txt",
                                        timeout=120) as r:
                body = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            got += 1
            time.sleep(0.2)
    print(f"[stations] USCRN: fetched {got} station-years")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(cfg) -> None:
    sc = cfg["sources"]["stations"]
    root = Path(sc["out_dir"])
    maxm = float(sc.get("max_missing_daylight", 0.10))
    daily, metas, reports = [], [], []

    meso = root / "ok_mesonet"
    if (meso / "meta.csv").exists():
        meta = pd.read_csv(meso / "meta.csv")
        for _, s in meta.iterrows():
            files = sorted((meso / s.station_id).glob("*.mts"))
            if not files:
                continue
            df = pd.concat([parse_mts(f.read_text()) for f in files], ignore_index=True)
            df, rep = bsrn_qc(df, s.lat, s.lon)
            d = station_daily(df, s.lat, s.lon, "5min", "end", maxm)
            daily.append(d.assign(station_id=s.station_id))
            metas.append(dict(s))
            reports.append(dict(station_id=s.station_id, network="ok_mesonet", **rep,
                                days_total=len(d),
                                days_kept=int(d.ghi_kwh.notna().sum())))

    uscrn = root / "uscrn"
    for d in sorted(uscrn.iterdir()) if uscrn.exists() else []:
        files = sorted(d.glob("*.txt"))
        if not files:
            continue
        df = pd.concat([parse_uscrn(f.read_text()) for f in files], ignore_index=True)
        lat, lon = float(df.lat.median()), float(df.lon.median())
        df, rep = bsrn_qc(df[["time", "ghi"]].sort_values("time"), lat, lon)
        dd = station_daily(df, lat, lon, "1h", "end", maxm)
        daily.append(dd.assign(station_id=d.name))
        metas.append(dict(station_id=d.name, lat=lat, lon=lon, network="uscrn"))
        reports.append(dict(station_id=d.name, network="uscrn", **rep,
                            days_total=len(dd), days_kept=int(dd.ghi_kwh.notna().sum())))

    for path in sc.get("generic_csv") or []:
        g = read_generic(path)
        for sid, df in g.groupby("station_id"):
            lat, lon = float(df["lat"].iloc[0]), float(df["lon"].iloc[0])
            freq = pd.infer_freq(df["time"].sort_values().iloc[:50]) or "5min"
            df, rep = bsrn_qc(df.sort_values("time"), lat, lon)
            d = station_daily(df, lat, lon, freq, "end", maxm)
            daily.append(d.assign(station_id=sid))
            metas.append(dict(station_id=sid, lat=lat, lon=lon, network=Path(path).stem))
            reports.append(dict(station_id=sid, network=Path(path).stem, **rep,
                                days_total=len(d), days_kept=int(d.ghi_kwh.notna().sum())))

    if not daily:
        raise SystemExit("[stations] no station data found under " + str(root))
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    meta = map_to_grid(pd.DataFrame(metas), grid, cfg.get("holdout"))
    rep = pd.DataFrame(reports)
    rep["qc_pass_rate"] = 1 - (rep.fail_ppl + rep.fail_closure) / rep.daylight_present.clip(lower=1)
    rep["day_keep_rate"] = rep.days_kept / rep.days_total.clip(lower=1)
    out = pd.concat(daily, ignore_index=True).dropna(subset=["ghi_kwh"])
    meta.to_csv(root / "stations.csv", index=False)
    out.to_csv(root / "station_daily.csv", index=False)
    rep.to_csv(root / "qc_report.csv", index=False)
    print(rep.to_string(index=False))
    print(meta.groupby("region").size().rename("stations per region").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download-mesonet")
    d.add_argument("--years", type=int, nargs="+", required=True)
    u = sub.add_parser("download-uscrn")
    u.add_argument("--years", type=int, nargs="+", required=True)
    sub.add_parser("build")
    a = ap.parse_args()
    c = load_config("data")
    if a.cmd == "download-mesonet":
        download_mesonet(c, a.years)
    elif a.cmd == "download-uscrn":
        download_uscrn(c, a.years)
    else:
        build(c)

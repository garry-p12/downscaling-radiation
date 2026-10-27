"""THE daily-aggregation utility: sub-daily irradiance -> daily kWh/m2/day.

Plan risk table: "Local solar day vs UTC day mismatch -> single tested utility
for daily aggregation, used everywhere." NSRDB (30 min), Oklahoma Mesonet
(5 min), SURFRAD (1-3 min) and any generic station series all come through
``daily_totals``; nothing else in the repo bins irradiance into days.

Why local SOLAR day. A UTC day boundary falls at ~18:00 local time over Texas,
so a UTC "day" glues the afternoon of one day to the morning of the next —
two different cloud regimes — and would never match POWER, which is requested
with ``time-standard=LST`` (download_power.py). Local solar time is
UTC + lon/15 h; it has no DST, so DST is handled by construction rather than by
a timezone table. The equation of time (<= 16 min) is ignored for the day
boundary: it moves the boundary around solar midnight, when GHI is zero.

Sample conventions (``stamp``):
  "instant"  value is an instantaneous sample at the stamp, representing
             [t - dt/2, t + dt/2]  (NSRDB PSM).
  "end"      value is the mean of the interval ending at the stamp
             (Oklahoma Mesonet SRAD 5-min averages).
  "start"    mean of the interval starting at the stamp.
Each sample is assigned to the solar day containing its interval MIDPOINT.

Missing data. Night samples (sun below the horizon at the midpoint) carry no
energy: they are set to 0 and never count as missing — including night slots
beyond the ends of the series. A day is kept only if at most
``max_missing_daylight`` of its
daylight samples are missing (plan 3.4: drop days with >10% missing daylight);
kept days have their daylight gaps linearly interpolated before integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .solar import cos_zenith

_NS_PER_S = 1_000_000_000
_DAY_S = 86_400


def _midpoints(times: np.ndarray, dt_s: float, stamp: str) -> np.ndarray:
    shift = {"instant": 0.0, "end": -dt_s / 2, "start": dt_s / 2}[stamp]
    return times + np.timedelta64(int(round(shift * _NS_PER_S)), "ns")


def check_regular(times_utc) -> float:
    """Return the sampling interval in seconds; raise if not regular."""
    t = np.asarray(times_utc, dtype="datetime64[ns]")
    if t.size < 2:
        raise ValueError("need at least two samples")
    d = np.diff(t).astype("int64")
    if not (d == d[0]).all() or d[0] <= 0:
        raise ValueError("series must be on a regular, increasing time axis; "
                         "reindex onto a full grid first (missing -> NaN)")
    return d[0] / _NS_PER_S


def solar_day_index(times_utc, lon_deg) -> np.ndarray:
    """Integer local-solar-day number (days since 1970-01-01) per (t, site).

    ``times_utc`` (T,), ``lon_deg`` (N,) -> (T, N) int64.
    """
    t = np.asarray(times_utc, dtype="datetime64[ns]").astype("int64") / _NS_PER_S
    lon = np.atleast_1d(np.asarray(lon_deg, dtype="float64"))
    return np.floor((t[:, None] + lon[None, :] * 240.0) / _DAY_S).astype("int64")


def daily_totals(values, times_utc, lat_deg, lon_deg, *,
                 stamp: str = "instant",
                 max_missing_daylight: float = 0.10,
                 site_chunk: int = 4096):
    """Integrate irradiance (W/m2) to local-solar-day totals (kWh/m2/day).

    values      (T, N) W/m2, NaN = missing. A 1-D series is treated as N=1.
    times_utc   (T,) datetime64, regular spacing.
    lat/lon     (N,) degrees.

    Returns (days, totals, missing_frac):
        days          (D,) datetime64[D] solar days spanned by the series
        totals        (D, N) kWh/m2/day, NaN where the day was rejected
        missing_frac  (D, N) fraction of daylight samples that were missing
    """
    v = np.asarray(values, dtype="float64")
    if v.ndim == 1:
        v = v[:, None]
    times = np.asarray(times_utc, dtype="datetime64[ns]")
    lat = np.atleast_1d(np.asarray(lat_deg, dtype="float64"))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype="float64"))
    T, N = v.shape
    if times.size != T or lat.size != N or lon.size != N:
        raise ValueError("shape mismatch between values, times and sites")
    dt_s = check_regular(times)
    per_day = _DAY_S / dt_s
    if abs(per_day - round(per_day)) > 1e-9:
        raise ValueError(f"interval {dt_s}s does not divide a day")
    per_day = int(round(per_day))
    # Pad one day of NaN slots on each side. A solar day that straddles the
    # series edge (Dec 31 at 100 W ends at 06:40 UTC Jan 1, in next year's
    # file) is then judged by its DAYLIGHT coverage only: the missing slots are
    # night, carry no energy, and must not reject the day.
    step = np.timedelta64(int(round(dt_s * _NS_PER_S)), "ns")
    pad = np.full((per_day, N), np.nan)
    times = np.concatenate([times[0] - step * np.arange(per_day, 0, -1), times,
                            times[-1] + step * np.arange(1, per_day + 1)])
    v = np.concatenate([pad, v, pad])
    T = times.size
    mid = _midpoints(times, dt_s, stamp)

    didx_all = solar_day_index(mid, lon)                  # (T, N)
    d0 = int(didx_all.min())
    D = int(didx_all.max()) - d0 + 1
    totals = np.full((D, N), np.nan)
    miss = np.full((D, N), np.nan)

    for s in range(0, N, site_chunk):
        sl = slice(s, min(N, s + site_chunk))
        n = sl.stop - sl.start
        vv = v[:, sl].copy()
        day = cos_zenith(mid, lat[sl], lon[sl]) > 0.0     # (T, n)
        didx = didx_all[:, sl] - d0

        missing = np.isnan(vv) & day
        vv[~day] = 0.0                                    # night: no energy
        # Interpolate daylight gaps along time; edges stay NaN.
        vv = pd.DataFrame(vv).interpolate(axis=0, limit_area="inside").to_numpy()
        vv = np.clip(vv, 0.0, None)

        flat = (didx * n + np.arange(n)[None, :]).ravel()
        size = D * n
        cnt = np.bincount(flat, minlength=size)
        n_day = np.bincount(flat, weights=day.ravel().astype("float64"),
                            minlength=size)
        n_miss = np.bincount(flat, weights=missing.ravel().astype("float64"),
                             minlength=size)
        energy = np.bincount(flat, weights=np.nan_to_num(vv).ravel(),
                             minlength=size)
        # A gap that interpolation could not close (series edge) is a hole.
        n_hole = np.bincount(flat, weights=np.isnan(vv).ravel().astype("float64"),
                             minlength=size)

        frac = np.where(n_day > 0, n_miss / np.maximum(n_day, 1), 0.0)
        ok = (cnt == per_day) & (frac <= max_missing_daylight) & (n_hole == 0)
        tot = energy * dt_s / 3600.0 / 1000.0             # W*s -> kWh
        totals[:, sl] = np.where(ok, tot, np.nan).reshape(D, n)
        miss[:, sl] = frac.reshape(D, n)

    days = (np.arange(D) + d0).astype("datetime64[D]")
    return days, totals, miss


def reindex_regular(times_utc, values, freq: str):
    """Put an irregular/gappy series onto a full regular grid (missing -> NaN).

    Station files drop rows rather than writing fill values; ``daily_totals``
    needs every slot present to tell a gap from a short day.
    """
    s = pd.DataFrame(np.asarray(values, dtype="float64"),
                     index=pd.DatetimeIndex(times_utc))
    s = s[~s.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(s.index[0].floor("D"), s.index[-1].ceil("D"),
                         freq=freq, inclusive="left")
    s = s.reindex(full)
    return s.index.values, s.to_numpy()

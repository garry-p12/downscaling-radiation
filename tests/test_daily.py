"""The single daily-aggregation utility (plan risk table). Includes the
phase-1 exit check: daily NSRDB totals match a manual calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.daily import daily_totals, solar_day_index
from common.solar import cos_zenith

T30 = pd.date_range("2020-06-01", "2020-06-05", freq="30min", inclusive="left").values


def _manual(values, times, lat, lon, day):
    """Brute force: loop over samples, keep those whose local solar day is
    ``day``, zero the night, sum x 0.5 h / 1000."""
    tot = 0.0
    for v, t in zip(values, times):
        sec = (t - np.datetime64("1970-01-01")).astype("timedelta64[s]").astype(float)
        d = np.floor((sec + lon * 240) / 86400)
        if d != (day - np.datetime64("1970-01-01")).astype(int):
            continue
        if cos_zenith(np.array([t]), [lat], [lon])[0, 0] > 0:
            tot += v
    return tot * 0.5 / 1000


def test_matches_manual_calc_for_three_random_pixels(rng):
    lat = rng.uniform(27, 35, 3)
    lon = rng.uniform(-104, -94, 3)
    cz = cos_zenith(T30, lat, lon)
    vals = np.clip(cz, 0, None) * 1000 * rng.uniform(0.3, 1.0, cz.shape)
    days, tot, _ = daily_totals(vals, T30, lat, lon)
    for i in range(3):
        for d in (np.datetime64("2020-06-02"), np.datetime64("2020-06-03")):
            got = tot[list(days).index(d), i]
            assert np.isclose(got, _manual(vals[:, i], T30, lat[i], lon[i], d), rtol=1e-9)


def test_solar_day_boundary_is_not_utc():
    """05:00 UTC at 100 W is 22:20 local solar time: still the previous day."""
    t = np.array(["2020-06-02T05:00"], dtype="datetime64[ns]")
    d = solar_day_index(t, [-100.0])[0, 0]
    assert d == (np.datetime64("2020-06-01") - np.datetime64("1970-01-01")).astype(int)


def test_constant_daylight_irradiance():
    lat, lon = np.array([30.0]), np.array([-97.0])
    day = cos_zenith(T30, lat, lon)[:, 0] > 0
    vals = np.where(day, 500.0, 0.0)
    days, tot, miss = daily_totals(vals, T30, lat, lon)
    i = list(days).index(np.datetime64("2020-06-02"))
    sd = solar_day_index(T30, lon)[:, 0]
    n = (day & (sd == sd[np.searchsorted(T30, np.datetime64("2020-06-02T18:00"))])).sum()
    assert np.isclose(tot[i, 0], 500 * n * 0.5 / 1000)
    assert miss[i, 0] == 0


def test_missing_daylight_threshold():
    lat, lon = np.array([30.0]), np.array([-97.0])
    cz = cos_zenith(T30, lat, lon)[:, 0]
    base = np.clip(cz, 0, None) * 800
    days, ref, _ = daily_totals(base, T30, lat, lon)
    i = list(days).index(np.datetime64("2020-06-02"))
    sd = solar_day_index(T30, lon)[:, 0]
    dl = np.nonzero((sd == sd[np.searchsorted(T30, np.datetime64("2020-06-02T18:00"))]) & (cz > 0))[0]

    few = base.copy()
    few[dl[10:12]] = np.nan                       # 2 of ~29: under 10%, interpolated
    _, t1, m1 = daily_totals(few, T30, lat, lon)
    assert np.isfinite(t1[i, 0]) and abs(t1[i, 0] / ref[i, 0] - 1) < 0.01

    many = base.copy()
    many[dl[5:10]] = np.nan                       # 5 of ~29: over 10%, rejected
    _, t2, m2 = daily_totals(many, T30, lat, lon)
    assert np.isnan(t2[i, 0]) and m2[i, 0] > 0.1


def test_missing_night_samples_do_not_count():
    lat, lon = np.array([30.0]), np.array([-97.0])
    cz = cos_zenith(T30, lat, lon)[:, 0]
    v = np.where(cz > 0, 600.0, np.nan)           # night entirely missing
    days, tot, miss = daily_totals(v, T30, lat, lon)
    i = list(days).index(np.datetime64("2020-06-02"))
    assert np.isfinite(tot[i, 0]) and miss[i, 0] == 0


def test_day_straddling_series_end_is_kept_when_only_night_is_cut():
    """Dec 31 solar day at 100 W ends 06:40 UTC Jan 1 — night, outside the file."""
    t = pd.date_range("2020-12-30", "2021-01-01", freq="30min", inclusive="left").values
    lat, lon = np.array([30.0]), np.array([-100.0])
    v = np.clip(cos_zenith(t, lat, lon)[:, 0], 0, None) * 700
    days, tot, _ = daily_totals(v, t, lat, lon)
    assert np.isfinite(tot[list(days).index(np.datetime64("2020-12-31")), 0])


def test_interval_end_stamps_shift_by_half_interval():
    t = pd.date_range("2020-06-01", "2020-06-04", freq="5min", inclusive="left").values
    lat, lon = np.array([30.0]), np.array([-97.0])
    v = np.clip(cos_zenith(t, lat, lon)[:, 0], 0, None) * 900
    _, a, _ = daily_totals(v, t, lat, lon, stamp="instant")
    _, b, _ = daily_totals(v, t, lat, lon, stamp="end")
    assert np.nanmax(np.abs(a - b)) < 0.05        # same energy, different binning

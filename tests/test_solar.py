"""Analytic TOA and solar geometry: the TOA that defines kt must be right."""
from __future__ import annotations

import numpy as np

from common.solar import cos_zenith, declination, sun_position, toa_daily_kwh


def test_fao56_example_8():
    """FAO-56 Example 8: 20 S, 3 September (J=246) -> Ra = 32.2 MJ/m2/day."""
    ra = toa_daily_kwh(-20.0, 246, gsc=1367.0, method="fao56") * 3.6
    assert abs(ra - 32.2) < 0.05


def test_matches_live_power_toa():
    """POWER v10 TOA_SW_DWN at (29.5 N, -99.5), 2020-06-01 was 11.3179."""
    assert abs(toa_daily_kwh(29.5, 153) / 11.3179 - 1) < 0.01


def test_polar_night_and_day():
    assert toa_daily_kwh(80.0, 355) == 0.0                       # polar night
    assert toa_daily_kwh(80.0, 172) > toa_daily_kwh(0.0, 172)    # polar day beats equator


def test_broadcasts_over_a_latitude_field():
    lat = np.linspace(27, 35, 12).reshape(3, 4)
    out = toa_daily_kwh(lat, 100)
    assert out.shape == (3, 4) and np.all(np.diff(out.ravel()) < 0)  # March: falls poleward


def test_solar_noon_zenith():
    """At true solar noon cos z = cos(lat - dec)."""
    lon = -90.0                                    # solar noon ~ 18:00 UTC
    t = np.array(["2020-06-21T18:00"], dtype="datetime64[ns]")
    cz = cos_zenith(t, [30.0], [lon])[0, 0]
    dec = declination(173)
    assert abs(cz - np.cos(np.deg2rad(30.0) - dec)) < 0.01


def test_sun_azimuth_is_south_at_noon_and_east_in_morning():
    t = np.array(["2020-03-20T18:00", "2020-03-20T13:00"], dtype="datetime64[ns]")
    _, az = sun_position(t, [30.0], [-90.0])
    assert abs(np.degrees(az[0, 0]) - 180) < 5
    assert 60 < np.degrees(az[1, 0]) < 120

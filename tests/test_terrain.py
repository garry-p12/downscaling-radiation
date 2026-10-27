"""Terrain geometry sanity on analytic surfaces."""
from __future__ import annotations

import numpy as np

from data.terrain import beam_shading, horizon_angles, sky_view_factor, slope_aspect


def test_flat_ground_is_open_sky():
    dem = np.full((20, 20), 300.0)
    hz = horizon_angles(dem, 1000, n_az=8, max_dist=5000)
    assert np.allclose(hz, 0) and np.allclose(sky_view_factor(hz), 1)
    lat, lon = np.full((20, 20), 30.0), np.full((20, 20), -98.0)
    assert np.allclose(beam_shading(hz, lat, lon, step_min=60), 1)


def test_plane_slope_and_aspect():
    x = np.arange(30) * 100.0
    dem = np.tile(-0.1 * x, (30, 1))                # falls to the EAST, 10% grade
    s, a = slope_aspect(dem, 100.0)
    assert np.allclose(s[5:-5, 5:-5], np.arctan(0.1))
    assert np.allclose(a[5:-5, 5:-5], np.pi / 2)    # downslope = east


def test_pit_sees_raised_horizon_and_shading():
    yy, xx = np.mgrid[0:41, 0:41]
    dem = 2.0 * np.hypot(yy - 20, xx - 20) ** 2     # bowl
    hz = horizon_angles(dem, 100.0, n_az=8, max_dist=2000)
    assert hz[:, 20, 20].min() > 0.3
    assert sky_view_factor(hz)[20, 20] < 0.9
    lat, lon = np.full(dem.shape, 30.0), np.full(dem.shape, -98.0)
    f = beam_shading(hz[:, 18:23, 18:23], lat[:5, :5], lon[:5, :5], step_min=60)
    assert (f[:, 2, 2] < 1).all() and (f >= 0).all()

"""Terrain radiation geometry: slope, aspect, horizon angles, sky-view, shading.

Pure numpy on a projected, regularly spaced DEM (north-up, metres), so it runs
without richdem / whitebox. Computed on a ~1 km DEM and block-averaged to the
10 km grid by data/statics.py: at 10 km the relief that casts shadows is
already averaged away, so deriving horizons from a 10 km DEM would report a
flat Texas everywhere.

Conventions: azimuth clockwise from north (0 = N, pi/2 = E); horizon angle is
elevation above the horizontal (rad, >= 0); aspect is the downslope direction.
"""
from __future__ import annotations

import numpy as np

from common.solar import sun_position


def slope_aspect(dem: np.ndarray, dx: float):
    """(slope rad, aspect rad clockwise from north, downslope direction)."""
    z = np.pad(dem.astype("float64"), 1, mode="edge")
    dzdx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * dx)          # +x = east
    dzdy = (z[:-2, 1:-1] - z[2:, 1:-1]) / (2 * dx)          # +y = north (row 0 north)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.mod(np.arctan2(-dzdx, -dzdy), 2 * np.pi)    # downslope bearing
    return slope, aspect


def horizon_angles(dem: np.ndarray, dx: float, n_az: int = 16,
                   max_dist: float = 30_000.0) -> np.ndarray:
    """Horizon elevation angle (rad, >= 0) per azimuth: (n_az, ny, nx).

    Ray march: for each azimuth, step outward one cell length at a time to
    ``max_dist`` and keep the max elevation angle of the terrain seen. Edge
    cells see the clamped edge (an under-estimate there, never an over-).
    """
    z = dem.astype("float64")
    ny, nx = z.shape
    rr, cc = np.mgrid[0:ny, 0:nx]
    n_steps = max(1, int(max_dist // dx))
    out = np.zeros((n_az, ny, nx))
    for a in range(n_az):
        az = 2 * np.pi * a / n_az
        sx, sy = np.sin(az), -np.cos(az)                     # rows grow south
        best = np.zeros((ny, nx))
        for k in range(1, n_steps + 1):
            r = np.clip(np.rint(rr + sy * k).astype(int), 0, ny - 1)
            c = np.clip(np.rint(cc + sx * k).astype(int), 0, nx - 1)
            ang = np.arctan2(z[r, c] - z, k * dx)
            np.maximum(best, ang, out=best)
        out[a] = best
    return out


def sky_view_factor(horizons: np.ndarray) -> np.ndarray:
    """Isotropic sky-view factor of a HORIZONTAL surface from horizon angles.

    Dozier & Frew (1990) with slope 0: V = mean_phi sin^2(H), H the horizon
    ZENITH angle = pi/2 - h, i.e. V = mean cos^2(h). 1 on open plains.
    """
    return np.mean(np.cos(np.clip(horizons, 0, np.pi / 2)) ** 2, axis=0)


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Average non-overlapping f x f blocks over the trailing two axes."""
    *lead, ny, nx = a.shape
    a = a[..., : ny - ny % f, : nx - nx % f]
    return a.reshape(*lead, ny // f, f, nx // f, f).mean(axis=(-3, -1))


def beam_shading(horizons: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 step_min: int = 20, year: int = 2001) -> np.ndarray:
    """Daily fraction of horizontal beam energy NOT blocked by terrain.

    f(doy, y, x) = sum cos(z) * [elev > horizon(az)] / sum cos(z), over the
    day's sunlit steps. cos(z) weighting is the horizontal-plane beam weight
    under a uniform atmosphere; the B2 baseline multiplies the direct part of
    GHI by f. Returns (366, ny, nx) float32 indexed by dayofyear - 1, the same
    convention as every DOY climatology here (computed for a non-leap
    ``year``; slot 366 copies 365, leap years are off by <= 1 day).
    """
    n_az, ny, nx = horizons.shape
    la, lo = lat.ravel(), lon.ravel()
    hz = horizons.reshape(n_az, -1)
    hz_wrap = np.concatenate([hz, hz[:1]], axis=0)          # close the circle
    out = np.ones((366, ny * nx), dtype="float32")
    steps = np.arange(0, 24 * 60, step_min)
    for d in range(365):
        day = np.datetime64(f"{year}-01-01") + np.timedelta64(d, "D")
        t = day + steps.astype("timedelta64[m]")
        elev, az = sun_position(t, la, lo)                   # (S, N)
        up = elev > 0
        if not up.any():
            continue
        pos = az / (2 * np.pi) * n_az
        i0 = np.floor(pos).astype(int) % n_az
        w = pos - np.floor(pos)
        cols = np.arange(la.size)[None, :]
        h_sun = (1 - w) * hz_wrap[i0, cols] + w * hz_wrap[i0 + 1, cols]
        cz = np.where(up, np.sin(elev), 0.0)
        lit = cz * (elev > h_sun)
        tot = cz.sum(0)
        out[d] = np.where(tot > 0, lit.sum(0) / np.maximum(tot, 1e-12), 1.0)
    out[365] = out[364]                                      # doy 366
    return out.reshape(366, ny, nx)

"""Sparse mappings: bilinear exactness, conservative binning, NaN handling."""
from __future__ import annotations

import numpy as np

from common import build_target_grid
from common import regrid as rg

DOM = {"lat_min": 29.0, "lat_max": 31.0, "lon_min": -99.0, "lon_max": -97.0}
GRID = {"target_crs": "EPSG:5070", "target_res_m": 10000}


def test_bilinear_is_exact_on_linear_fields(rng):
    clat = np.arange(27.5, 33.6, 1.0)
    clon = np.arange(-101.5, -94.4, 1.0)
    f = lambda la, lo: 2.0 * la - 0.5 * lo + 3  # noqa: E731
    CL, CO = np.meshgrid(clat, clon, indexing="ij")
    dlat, dlon = rng.uniform(28, 33, 50), rng.uniform(-101, -95, 50)
    W = rg.bilinear_matrix(clat, clon, dlat, dlon)
    assert np.allclose(np.asarray(W.sum(1)).ravel(), 1)
    out = rg.apply(W, f(CL, CO).ravel()[None])[0]
    assert np.allclose(out, f(dlat, dlon))


def test_enclosing_index_finds_the_containing_cell():
    clat, clon = np.array([29.5, 30.5]), np.array([-98.5, -97.5])
    idx = rg.enclosing_index(clat, clon, [29.9, 30.1, 30.9], [-98.9, -97.1, -98.0001])
    assert idx.tolist() == [0, 3, 2]


def test_binning_is_conservative_and_nan_aware(rng):
    from pyproj import Transformer

    grid = build_target_grid(DOM, GRID)
    lat = rng.uniform(29, 31, 5000)
    lon = rng.uniform(-99, -97, 5000)
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(lon, lat)
    W, counts = rg.binning_matrix(x, y, grid)
    assert counts.sum() <= 5000 and counts.max() > 0
    const = rg.apply(W, np.full((1, 5000), 7.0), out_shape=grid.shape)[0]
    assert np.allclose(const[counts > 0], 7.0) and np.isnan(const[counts == 0]).all()
    v = rng.normal(size=5000)
    v[::3] = np.nan
    out = rg.apply(W, v[None], out_shape=grid.shape)[0]
    # each cell = mean of its finite points
    col = np.floor((x - (grid.x[0] - 5000)) / 10000).astype(int)
    row = np.floor(((grid.y[0] + 5000) - y) / 10000).astype(int)
    r, c = np.argwhere(counts > 3)[0]
    sel = (row == r) & (col == c) & np.isfinite(v)
    assert np.isclose(out[r, c], v[sel].mean())


def test_coarse_mean():
    fine = np.array([[1.0, 3.0, 5.0, np.nan]])
    assert np.allclose(rg.coarse_mean(fine, np.array([0, 0, 1, 1]), 3)[0, :2], [2.0, 5.0])


def test_linear_matrix_exact_on_planes_and_nan_outside(rng):
    grid = build_target_grid(DOM, {"target_crs": "EPSG:5070", "target_res_m": 1000})
    xx, yy = grid.meshgrid()
    # scattered points covering only the western half of the grid
    px = rng.uniform(xx.min() - 5e3, xx.mean(), 3000)
    py = rng.uniform(yy.min() - 5e3, yy.max() + 5e3, 3000)
    W, counts = rg.linear_matrix(px, py, grid)
    f = lambda x, y: 1e-4 * x - 3e-5 * y + 2  # noqa: E731
    out = rg.apply(W, f(px, py)[None], out_shape=grid.shape)[0]
    inside = counts > 0
    assert inside[:, : grid.shape[1] // 3].all() and not inside[:, -1].any()
    assert np.allclose(out[inside], f(xx, yy)[inside])
    assert np.isnan(out[~inside]).all()


def test_point_matrix_dispatch(rng):
    grid = build_target_grid(DOM, GRID)
    xx, yy = grid.meshgrid()
    px, py = xx.ravel() + 100, yy.ravel() - 100
    Wb, cb = rg.point_matrix(px, py, grid, "bin")
    assert (cb == 1).all()
    _, cl = rg.point_matrix(px, py, grid, "linear")
    assert cl.sum() > 0

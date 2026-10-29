"""Precomputed coarse<->fine and point->fine mappings as sparse matrices.

Plan 3.3: "Precompute the coarse-to-fine index mapping once and cache it."
Every mapping here is a scipy.sparse matrix ``W`` of shape (n_dst, n_src), so
regridding a whole (T, n_src) stack is one sparse-dense product and the
weights are computed exactly once per (source grid, fine grid) pair.

  bilinear_matrix   coarse regular lat/lon -> fine cell centres (interp)
  enclosing_index   fine cell -> the coarse cell that contains it
  binning_matrix    scattered fine-scale points (NSRDB ~4 km) -> fine cells,
                    area-weighted MEAN (conservative), never bilinear
  linear_matrix     scattered points -> cell centres, linear on a Delaunay
                    triangulation: for grids FINER than the points (1 km vs
                    NSRDB 4 km / GOES 2-4 km), where binning leaves most cells empty
  point_matrix      binning or linear by name (config grid.point_remap)
  coarse_mean       fine -> coarse block means through enclosing_index
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _axis_weights(src: np.ndarray, dst: np.ndarray):
    """Lower index and fractional weight of ``dst`` on ascending ``src``.

    Points outside the axis are clamped to the edge (constant extrapolation):
    POWER is downloaded with a pad around the domain, so this only triggers on
    a misconfigured pad, and a constant is the least-surprising fallback.
    """
    if src.size == 1:
        return np.zeros(dst.shape, int), np.zeros(dst.shape)
    if not np.all(np.diff(src) > 0):
        raise ValueError("source axis must be strictly ascending")
    x = np.clip(dst, src[0], src[-1])
    i = np.clip(np.searchsorted(src, x, side="right") - 1, 0, src.size - 2)
    w = (x - src[i]) / (src[i + 1] - src[i])
    return i, w


def bilinear_matrix(src_lat, src_lon, dst_lat, dst_lon) -> sp.csr_matrix:
    """Bilinear weights from a regular (lat, lon) grid onto arbitrary points.

    Source fields are flattened C-order as (lat, lon); dst points are flat.
    Interpolation is in lat/lon space, which is what "bilinear POWER" means in
    every baseline the literature reports.
    """
    src_lat, src_lon = np.asarray(src_lat, float), np.asarray(src_lon, float)
    dlat, dlon = np.ravel(dst_lat).astype(float), np.ravel(dst_lon).astype(float)
    nlon = src_lon.size
    i, wy = _axis_weights(src_lat, dlat)
    j, wx = _axis_weights(src_lon, dlon)
    i1 = np.minimum(i + 1, src_lat.size - 1)
    j1 = np.minimum(j + 1, nlon - 1)
    rows = np.repeat(np.arange(dlat.size), 4)
    cols = np.stack([i * nlon + j, i * nlon + j1, i1 * nlon + j, i1 * nlon + j1], 1)
    vals = np.stack([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx], 1)
    W = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())),
                      shape=(dlat.size, src_lat.size * nlon))
    W.sum_duplicates()
    return W


def enclosing_index(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Flat index of the coarse cell whose centre is nearest each fine point.

    On a regular grid of cell centres, nearest centre == enclosing cell.
    """
    src_lat, src_lon = np.asarray(src_lat, float), np.asarray(src_lon, float)
    dlat, dlon = np.ravel(dst_lat), np.ravel(dst_lon)
    i = np.abs(dlat[:, None] - src_lat[None, :]).argmin(1)
    j = np.abs(dlon[:, None] - src_lon[None, :]).argmin(1)
    return i * src_lon.size + j


def binning_matrix(pt_x, pt_y, grid, pt_weight=None, min_count: int = 1):
    """Area-weighted mean of scattered points falling inside each fine cell.

    ``pt_x``/``pt_y`` are point coordinates in the fine grid's CRS. NSRDB points
    are centres of ~4 km cells, i.e. ~6 per 10 km cell, so averaging the points
    inside a cell is the conservative (area-weighted) remap the plan asks for;
    ``pt_weight`` should be each source cell's area (cos(lat) for a regular
    lat/lon source). Cells with fewer than ``min_count`` points get an empty
    row (-> NaN after ``apply``).

    Returns (W (n_fine, n_pts) row-normalised, counts (ny, nx)).
    """
    ny, nx = grid.shape
    res = grid.res_m
    x0 = grid.x[0] - res / 2
    y0 = grid.y[0] + res / 2                     # north edge (y descending)
    col = np.floor((np.asarray(pt_x) - x0) / res).astype(int)
    row = np.floor((y0 - np.asarray(pt_y)) / res).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    w = np.ones(col.size) if pt_weight is None else np.asarray(pt_weight, float)
    cell = row * nx + col
    pts = np.nonzero(inside)[0]
    counts = np.bincount(cell[pts], minlength=ny * nx)
    keep = pts[counts[cell[pts]] >= min_count]
    W = sp.csr_matrix((w[keep], (cell[keep], keep)), shape=(ny * nx, col.size))
    rs = np.asarray(W.sum(1)).ravel()
    W = sp.diags(np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1), 0.0)) @ W
    return W.tocsr(), counts.reshape(ny, nx)


def linear_matrix(pt_x, pt_y, grid):
    """Barycentric (linear) interpolation of scattered points at cell centres.

    Returns (W (n_fine, n_pts), counts (ny, nx)); counts is the number of
    contributing points (3 inside the points' hull, 0 outside -> NaN rows).
    Points outside the grid's padded extent are dropped before triangulating.
    """
    from scipy.spatial import Delaunay

    ny, nx = grid.shape
    px, py = np.asarray(pt_x, float), np.asarray(pt_y, float)
    pad = 10 * grid.res_m + 20e3
    near = np.nonzero((px > grid.x.min() - pad) & (px < grid.x.max() + pad)
                      & (py > grid.y.min() - pad) & (py < grid.y.max() + pad))[0]
    tri = Delaunay(np.c_[px[near], py[near]])
    xx, yy = grid.meshgrid()
    q = np.c_[xx.ravel(), yy.ravel()]
    simp = tri.find_simplex(q)
    inside = np.nonzero(simp >= 0)[0]
    T = tri.transform[simp[inside]]                     # (n, 3, 2)
    b = np.einsum("nij,nj->ni", T[:, :2], q[inside] - T[:, 2])
    bary = np.c_[b, 1 - b.sum(1)]
    cols = near[tri.simplices[simp[inside]]]            # (n, 3) -> original point index
    W = sp.csr_matrix((bary.ravel(), (np.repeat(inside, 3), cols.ravel())),
                      shape=(ny * nx, px.size))
    counts = np.zeros(ny * nx, int)
    counts[inside] = 3
    return W, counts.reshape(ny, nx)


def point_matrix(pt_x, pt_y, grid, mode: str = "bin", pt_weight=None, min_count: int = 1):
    """``mode`` 'bin' -> binning_matrix, 'linear' -> linear_matrix."""
    if mode == "linear":
        return linear_matrix(pt_x, pt_y, grid)
    if mode != "bin":
        raise ValueError(f"point_remap {mode!r}: use 'bin' or 'linear'")
    return binning_matrix(pt_x, pt_y, grid, pt_weight=pt_weight, min_count=min_count)


def apply(W: sp.spmatrix, src: np.ndarray, out_shape=None) -> np.ndarray:
    """Apply a (n_dst, n_src) mapping to a (..., n_src) stack.

    NaN-aware: weights are renormalised over finite source values, so a single
    missing NSRDB point inside a cell does not blank the whole cell, and rows
    with no finite contributor return NaN. Flatten 2-D source fields first
    (``field.reshape(T, -1)``).
    """
    n_src = W.shape[1]
    src = np.asarray(src, dtype="float64")
    if src.shape[-1] != n_src:
        raise ValueError(f"last axis {src.shape[-1]} != mapping width {n_src}")
    lead = src.shape[:-1]
    flat = src.reshape(-1, n_src)
    good = np.isfinite(flat)
    num = W @ np.where(good, flat, 0.0).T                 # (n_dst, K)
    den = W @ good.T.astype("float64")
    has_rows = np.asarray(W.getnnz(axis=1) > 0)[:, None]
    out = np.where((den > 1e-12) & has_rows, num / np.maximum(den, 1e-12), np.nan).T
    if out_shape is not None:
        return out.reshape(*lead, *out_shape)
    return out.reshape(*lead, W.shape[0])


def coarse_mean(fine: np.ndarray, enclosing: np.ndarray, n_coarse: int) -> np.ndarray:
    """Mean of fine cells per enclosing coarse cell. fine (..., n_fine)."""
    flat = fine.reshape(-1, enclosing.size)
    out = np.full((flat.shape[0], n_coarse), np.nan)
    for k, row in enumerate(flat):
        g = np.isfinite(row)
        s = np.bincount(enclosing[g], weights=row[g], minlength=n_coarse)
        c = np.bincount(enclosing[g], minlength=n_coarse)
        out[k] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out.reshape(*fine.shape[:-1], n_coarse)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def grid_key(*arrays) -> str:
    """Stable hash of the coordinate arrays that define a mapping."""
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype="float64"))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def cached(path: Path, build):
    """Load a sparse matrix from ``path`` or build and save it."""
    path = Path(path)
    if path.exists():
        return sp.load_npz(path).tocsr()
    W = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, W.tocsr())
    return W

"""Anchored target carried from a coarse store to a nested fine one."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from evaluation.station_anchor import transfer_target


def _store(path, x, y, nsrdb, corr=None):
    t = pd.date_range("2018-01-01", periods=nsrdb.shape[0])
    data = {"nsrdb_ghi": (("time", "y", "x"), nsrdb.astype("float32"))}
    if corr is not None:
        data["nsrdb_ghi_corr_a"] = (("time", "y", "x"), corr.astype("float32"))
    xr.Dataset(data, coords=dict(time=t, x=x, y=y)).chunk({"time": 2}).to_zarr(path, mode="w")


def test_transfer_adds_interpolated_shift(tmp_path):
    # coarse 10 km, 4 x 4 cells; fine 5 km nested in its interior
    cx = 5e3 + 10e3 * np.arange(4)
    cy = 35e3 - 10e3 * np.arange(4)
    T = 3
    base = np.full((T, 4, 4), 5.0)
    shift = (-0.01 * cx[None, None, :] + 0 * cy[None, :, None]) * np.ones((T, 1, 1)) / 1e3
    shift[:, 0, 0] = np.nan                         # a coarse gap -> nearest fill
    _store(tmp_path / "c.zarr", cx, cy, base, base + shift)
    fx = 12.5e3 + 5e3 * np.arange(3)
    fy = 27.5e3 - 5e3 * np.arange(3)
    fine = np.full((T, 3, 3), 4.0)
    _store(tmp_path / "f.zarr", fx, fy, fine)
    transfer_target({"dataset": {"out_zarr": str(tmp_path / "f.zarr")}}, str(tmp_path / "c.zarr"), "a")
    out = xr.open_zarr(tmp_path / "f.zarr")["nsrdb_ghi_corr_a"].values
    want = 4.0 - 0.01 * fx / 1e3                     # shift is linear in x -> exact
    assert np.allclose(out, np.broadcast_to(want, out.shape), atol=1e-5)

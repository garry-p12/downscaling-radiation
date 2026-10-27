"""MODIS cloud-state decoding and 1 km -> 10 km aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from common import build_target_grid
from data.modis_cloud import aggregate_file, decode_state


def test_decode_bits():
    v = np.array([0, 1, 2, 3, 0b1101, 65535], dtype="uint16")   # 0b1101: shadow bit + cloudy
    out = decode_state(v)
    assert np.allclose(out[:5], [0, 1, 0.5, 0, 1]) and np.isnan(out[5])


def test_aggregate_fraction_and_coverage(tmp_path, synth_cfg):
    grid = build_target_grid(synth_cfg["domain"], synth_cfg["grid"])
    lat = np.arange(synth_cfg["domain"]["lat_max"] + 0.1, synth_cfg["domain"]["lat_min"] - 0.1, -0.01)
    lon = np.arange(synth_cfg["domain"]["lon_min"] - 0.1, synth_cfg["domain"]["lon_max"] + 0.1, 0.01)
    v = np.zeros((2, lat.size, lon.size), "uint16")
    v[0, :, : lon.size // 2] = 1                  # west half cloudy on day 0
    v[1] = 65535                                  # day 1 not observed
    ds = xr.Dataset({"state_1km_1": (("time", "lat", "lon"), v)},
                    coords=dict(time=pd.date_range("2020-06-01", periods=2), lat=lat, lon=lon))
    ds.to_netcdf(tmp_path / "m.nc")
    times, out, _ = aggregate_file(tmp_path / "m.nc", grid)
    assert out.shape == (2, *grid.shape) and np.isnan(out[1]).all()
    lat2d, lon2d = grid.latlon()
    mid = (synth_cfg["domain"]["lon_min"] + synth_cfg["domain"]["lon_max"]) / 2
    assert np.nanmean(out[0][lon2d < mid - 0.3]) > 0.95
    assert np.nanmean(out[0][lon2d > mid + 0.3]) < 0.05

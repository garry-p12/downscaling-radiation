"""Ingest paths against fake files with the real on-disk layouts.

Phase 1 exit check: "daily NSRDB total matches manual calc for 3 random
pixels" — here through the actual HDF5 reader, fill-flag handling and
aggregation, not just the aggregation function.
"""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from common.solar import cos_zenith


@pytest.fixture
def fake_nsrdb(tmp_path, rng):
    times = pd.date_range("2020-01-01", "2020-12-31 23:30", freq="30min")
    lat = np.r_[rng.uniform(28, 34, 6), 45.0, 10.0]          # last two outside domain
    lon = np.r_[rng.uniform(-103, -95, 6), -100.0, -100.0]
    order = np.argsort(lon, kind="stable")                    # file is lon-ordered
    lat, lon = lat[order], lon[order]
    cz = cos_zenith(times.values, lat, lon)
    ghi = np.round(np.clip(cz, 0, None) * rng.uniform(300, 1000, cz.shape)).astype("uint16")
    flag = np.zeros(ghi.shape, "uint8")
    flag[1000:1010, 0] = 100                                  # fully gap-filled run
    flag[2000:2010, 0] = 36                                   # partly filled: kept
    meta = np.zeros(lat.size, dtype=[("latitude", "<f4"), ("longitude", "<f4")])
    meta["latitude"], meta["longitude"] = lat, lon
    path = tmp_path / "nsrdb_2020.h5"
    with h5py.File(path, "w") as f:
        f["time_index"] = np.array([t.strftime("%Y-%m-%d %H:%M:%S+00:00").encode()
                                    for t in times])
        f["meta"] = meta
        d = f.create_dataset("ghi", data=ghi)
        d.attrs.update(psm_scale_factor=1.0, physical_max=1350.0, physical_min=0.0)
        f.create_dataset("fill_flag", data=flag)
    return path, times, lat.astype("float32"), lon.astype("float32"), ghi, flag


def _manual_day(v, times, lat, lon, day):
    sec = (times.values - np.datetime64("1970-01-01")).astype("timedelta64[s]").astype(float)
    dnum = np.floor((sec + lon * 240) / 86400)
    target = (np.datetime64(day) - np.datetime64("1970-01-01")).astype(int)
    up = cos_zenith(times.values, [lat], [lon])[:, 0] > 0
    sel = (dnum == target) & up
    return v[sel].astype(float).sum() * 0.5 / 1000


def test_nsrdb_daily_matches_manual_for_three_random_pixels(fake_nsrdb, tmp_path, synth_cfg, rng):
    from data.download_nsrdb import process_year

    path, times, lat, lon, ghi, _ = fake_nsrdb
    cfg = copy.deepcopy(synth_cfg)
    cfg["domain"] = {"name": "t", "lat_min": 27, "lat_max": 35, "lon_min": -104, "lon_max": -94}
    cfg["sources"]["nsrdb"].update(out_dir=str(tmp_path / "out"), variables=["ghi"])
    (out,) = process_year(cfg, 2020, uri=str(path), max_sites=4)
    da = xr.open_dataarray(out)
    assert da.sizes["point"] == 6                             # bbox selection
    for i in rng.choice(np.arange(1, 6), 3, replace=False):   # site 0 has flags
        k = int(np.nonzero((lat == da.lat.values[i]) & (lon == da.lon.values[i]))[0][0])
        for day in ("2020-03-10", "2020-07-04", "2020-11-20"):
            got = float(da.sel(time=day).isel(point=i))
            assert got == pytest.approx(_manual_day(ghi[:, k], times, lat[k], lon[k], day), rel=1e-6)


def test_nsrdb_flagged_samples_are_dropped(fake_nsrdb, tmp_path, synth_cfg):
    from data.download_nsrdb import read_variable

    path, *_ , flag = fake_nsrdb
    with h5py.File(path) as f:
        v = read_variable(f, "ghi", [(0, 3)], drop_at=100)
        w = read_variable(f, "ghi", [(0, 3)], drop_at=None)
    assert np.isnan(v[1000:1010, 0]).all() and np.isfinite(w[1000:1010, 0]).all()
    assert np.isfinite(v[2000:2010, 0]).all()


def test_power_stitch_and_units(tmp_path):
    from common.units import to_kwh_day
    from data.download_power import open_parameter

    t = pd.date_range("2020-01-01", periods=3)
    for i, (la, lo) in enumerate([((29.5, 30.5), (-99.5, -98.5)), ((29.5, 30.5), (-97.5, -96.5))]):
        ds = xr.Dataset({"ALLSKY_SFC_SW_DWN": (("time", "lat", "lon"),
                                               np.full((3, 2, 2), 18.0 + i))},
                        coords=dict(time=t, lat=list(la), lon=list(lo)))
        ds["ALLSKY_SFC_SW_DWN"].attrs["units"] = "MJ/m^2/day"
        ds.to_netcdf(tmp_path / f"power_ALLSKY_SFC_SW_DWN_x_2020_t{i:02d}.nc")
    da = open_parameter(tmp_path, "ALLSKY_SFC_SW_DWN")
    assert da.shape == (3, 2, 4) and not np.isnan(da.values).any()
    assert to_kwh_day(da.values, da.attrs["units"])[0, 0, 0] == pytest.approx(5.0)

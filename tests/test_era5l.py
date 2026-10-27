"""ERA5-Land ssrd de-accumulation: accumulated since 00 UTC -> hourly mean flux."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from data.download_era5_land import hourly_flux


def test_deaccumulation_recovers_hourly_flux(rng):
    t = pd.date_range("2020-06-01 00:00", "2020-06-03 00:00", freq="h")
    flux = rng.uniform(0, 900, (t.size, 2))                # W/m2, hour ENDING at t
    acc = np.zeros_like(flux)
    for i, ts in enumerate(t):                             # ERA5-Land convention
        if ts.hour == 0:
            acc[i] = flux[i - 23:i + 1].sum(0) * 3600 if i >= 23 else np.nan
        else:
            start = i - ts.hour + 1                        # first hour after 00 UTC
            acc[i] = flux[start:i + 1].sum(0) * 3600
    da = xr.DataArray(acc, dims=("time", "p"), coords=dict(time=t))
    out = hourly_flux(da).values
    ok = np.isfinite(out[:, 0])
    assert ok[1:].all()
    assert np.allclose(out[ok], flux[ok], rtol=1e-4)

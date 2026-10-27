"""Source-change check inside the study period (plan 2.3, 3.4).

Known breaks: POWER SRB -> CERES SYN1deg on 2001-01-01 (excluded by the time
window), POWER CERES -> FLASHFlux in the most recent months, NSRDB GOES-East
generation change in 2018. A break shows up as a STEP in the yearly mean of
(NSRDB block-mean - POWER) over matched coarse cells, which a model trained
before the step would carry into the test years as bias.

Flags years whose offset sits more than ``k`` robust SDs (MAD) from the median.

    python -m data.check_discontinuity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import load_config
from common import regrid as rg
from data.build_store import open_store


def yearly_offsets(ds, k: float = 3.0) -> pd.DataFrame:
    lat, lon = ds.lat.values, ds.lon.values
    clat, clon = ds.clat.values, ds.clon.values
    enc = rg.enclosing_index(clat, clon, lat, lon)
    nc = clat.size * clon.size
    valid = ds.valid.values.astype(bool).ravel()
    tgt = ds.nsrdb_ghi.values.reshape(ds.sizes["time"], -1)
    tgt = np.where(valid[None], tgt, np.nan)
    block = rg.coarse_mean(tgt, enc, nc)                       # (T, nc)
    cnt = np.bincount(enc[valid], minlength=nc)
    full = cnt >= 0.5 * np.bincount(enc, minlength=nc).max()   # well-covered cells
    pw = ds.power_ghi.values.reshape(ds.sizes["time"], -1)
    diff = np.where(full[None], block - pw, np.nan)
    years = pd.DatetimeIndex(ds.time.values).year
    df = pd.DataFrame({"year": years, "offset": np.nanmean(diff, 1),
                       "ratio": np.nanmean(np.where(full[None], block, np.nan), 1)
                       / np.nanmean(np.where(full[None], pw, np.nan), 1)})
    y = df.groupby("year").mean()
    med = y.offset.median()
    mad = 1.4826 * np.median(np.abs(y.offset - med)) or 1e-9
    y["z"] = (y.offset - med) / mad
    y["flag"] = np.abs(y.z) > k
    return y


def main() -> None:
    cfg = load_config("data")
    y = yearly_offsets(open_store(cfg))
    print("NSRDB - POWER, yearly mean over well-covered coarse cells (kWh/m2/day)")
    print(y.round(3).to_string())
    if y.flag.any():
        print(f"\nFLAGGED years: {y.index[y.flag].tolist()} — check source versions "
              "before trusting a model trained across them.")


if __name__ == "__main__":
    main()

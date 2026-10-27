"""Temporal splits, the spatial holdout, and the three test cells.

Same philosophy as the temperature paper (plan 6):
  * contiguous year blocks, never random days (daily fields autocorrelate);
  * a region held out entirely (Austin), plus a buffer of >= one COARSE cell
    in which no pixel contributes a training target, climatology or
    normalisation statistic, so nothing about the holdout leaks in through a
    neighbouring POWER cell;
  * three test cells: temporal-only, spatial-only, both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def assign_splits(times, time_cfg: dict) -> np.ndarray:
    """Label each day train/val/test, 'none' outside every window.

    Identical to downscaling-temperature data/build_dataset.assign_splits.
    """
    times = pd.DatetimeIndex(times)
    split = np.full(len(times), "none", dtype="<U5")
    for name in SPLITS:
        lo, hi = time_cfg[name]
        m = (times >= pd.Timestamp(lo)) & (times <= pd.Timestamp(hi))
        split[m] = name
    return split


def box_mask(lat: np.ndarray, lon: np.ndarray, box: dict, pad: float = 0.0):
    return ((lat >= box["lat_min"] - pad) & (lat <= box["lat_max"] + pad)
            & (lon >= box["lon_min"] - pad) & (lon <= box["lon_max"] + pad))


def holdout_masks(lat: np.ndarray, lon: np.ndarray, holdout: dict | None):
    """(holdout, exclusion) boolean (ny, nx) masks.

    ``exclusion`` = holdout + buffer: pixels that must never contribute to
    training. Buffer defaults to 1.0 deg = one POWER solar cell.
    """
    if not holdout:
        z = np.zeros(lat.shape, bool)
        return z, z
    hold = box_mask(lat, lon, holdout)
    excl = box_mask(lat, lon, holdout, pad=float(holdout.get("buffer_deg", 1.0)))
    return hold, excl


def train_pixels(valid: np.ndarray, exclusion: np.ndarray) -> np.ndarray:
    """Pixels allowed to contribute targets / statistics: valid and not excluded."""
    return valid & ~exclusion


def evaluation_cells(split: np.ndarray, holdout: np.ndarray, exclusion: np.ndarray,
                     valid: np.ndarray, spatial_split: str = "train",
                     spatial_day_stride: int = 1) -> dict:
    """The three evaluation cells as {name: (day_mask (T,), pixel_mask (ny,nx))}.

    temporal  test years      x training-region pixels
    spatial   ``spatial_split`` years (default train) x holdout pixels, i.e.
              days the model has seen elsewhere at pixels it has never seen.
              ``spatial_day_stride`` subsamples those days (it is ~15 years).
    both      test years      x holdout pixels  (the headline number)
    """
    test = split == "test"
    sp_days = split == spatial_split
    if spatial_day_stride > 1:
        idx = np.nonzero(sp_days)[0]
        sp_days = np.zeros_like(sp_days)
        sp_days[idx[::spatial_day_stride]] = True
    region = valid & ~exclusion
    hold = valid & holdout
    return {"temporal": (test, region),
            "spatial": (sp_days, hold),
            "both": (test, hold)}

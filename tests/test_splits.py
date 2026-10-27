"""Temporal blocks, holdout + buffer, and the three test cells."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import load_config
from common.splits import assign_splits, evaluation_cells, holdout_masks
from tests.conftest import REPO_ROOT


def test_shipped_split_is_contiguous_ceres_era_and_gapless():
    cfg = load_config(str(REPO_ROOT / "configs/data.yaml"))
    t = cfg["time"]
    assert pd.Timestamp(t["train"][0]) >= pd.Timestamp("2001-01-01"), "pre-CERES (SRB) data"
    times = pd.date_range(t["train"][0], t["test"][1], freq="D")
    s = assign_splits(times, t)
    assert "none" not in set(s)
    assert int((s[1:] != s[:-1]).sum()) == 2


def test_buffer_contains_holdout():
    lat, lon = np.meshgrid(np.linspace(27, 35, 81), np.linspace(-104, -94, 101), indexing="ij")
    hold, excl = holdout_masks(lat, lon, {"lat_min": 29.7, "lat_max": 31, "lon_min": -98.6,
                                          "lon_max": -97, "buffer_deg": 1.0})
    assert hold.any() and (excl | ~hold).all() and excl.sum() > hold.sum()


def evaluation_cells_are_disjoint_in_the_right_way():
    split = np.array(["train"] * 10 + ["val"] * 3 + ["test"] * 4)
    hold = np.zeros((6, 6), bool); hold[2:4, 2:4] = True
    excl = np.zeros((6, 6), bool); excl[1:5, 1:5] = True
    valid = np.ones((6, 6), bool)
    c = evaluation_cells(split, hold, excl, valid, spatial_day_stride=2)
    assert not (c["temporal"][1] & excl).any()
    assert (c["both"][1] == hold).all() and (c["spatial"][1] == hold).all()
    assert c["both"][0].sum() == 4 and c["spatial"][0].sum() == 5
    assert not (c["spatial"][0] & (split != "train")).any()

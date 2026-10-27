"""Metrics and unit conversion."""
from __future__ import annotations

import numpy as np
import pytest

from common.units import to_kwh_day
from evaluation import metrics as M


def test_perfect_prediction(rng):
    o = rng.uniform(1, 8, 100)
    s = M.summary(o, o)
    assert s["rmse"] == 0 and s["mbe"] == 0 and s["kge"] == pytest.approx(1) and s["r"] == pytest.approx(1)


def test_skill_of_reference_is_zero_and_nan_pairs_dropped(rng):
    o = rng.uniform(1, 8, 100)
    ref = o + 1
    p = o.copy(); p[0] = np.nan
    assert M.skill(ref, o, ref) == 0
    assert M.skill(p, o, ref) == 1
    assert M.mbe(o + 0.5, o) == pytest.approx(0.5)


def test_units():
    assert to_kwh_day(3.6, "MJ/m^2/day") == pytest.approx(1.0)
    assert to_kwh_day(5.0, "kW-hr/m^2/day") == 5.0
    assert to_kwh_day(250.0, "W/m^2") == pytest.approx(6.0)
    with pytest.raises(ValueError):
        to_kwh_day(1.0, "furlongs")

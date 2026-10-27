"""Unit harmonisation. Everything is carried in kWh/m2/day (plan 3.2).

POWER's units depend on the ``community`` request parameter (RE/SB return
kW-hr/m^2/day, AG returns MJ/m^2/day), so units are read from each file's
``units`` attribute and converted here rather than assumed.
"""
from __future__ import annotations

import numpy as np

_FACTORS = {
    # -> kWh/m2/day
    "kw-hr/m^2/day": 1.0,
    "kwh/m^2/day": 1.0,
    "kwh/m2/day": 1.0,
    "kwh m-2 day-1": 1.0,
    "mj/m^2/day": 1.0 / 3.6,
    "mj/m2/day": 1.0 / 3.6,
    "mj m-2 day-1": 1.0 / 3.6,
    # A daily-MEAN flux in W/m2 times 24 h.
    "w/m^2": 24.0 / 1000.0,
    "w/m2": 24.0 / 1000.0,
    "w m-2": 24.0 / 1000.0,
}

DIMENSIONLESS = {"dimensionless", "1", "", "unitless", "-", "fraction"}


def to_kwh_day(values, units: str):
    """Convert an irradiation field to kWh/m2/day given its unit string."""
    key = (units or "").strip().lower()
    if key not in _FACTORS:
        raise ValueError(f"unknown irradiation units {units!r}; "
                         f"known: {sorted(_FACTORS)}")
    return np.asarray(values) * _FACTORS[key]


def is_dimensionless(units: str | None) -> bool:
    return (units or "").strip().lower() in DIMENSIONLESS

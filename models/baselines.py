"""Physics and climatology baselines B0-B3 (plan 5). All return GHI, kWh/m2/day.

B0  bilinear POWER GHI                                  floor
B1  bilinear POWER kt x analytic fine TOA               does the kt reframing help?
B2  B1 + terrain: beam part x daily unblocked fraction,
    diffuse part x sky-view factor                       physics-only
B3  B1 x fine NSRDB/POWER kt climatology ratio          static bias map; the
                                                         baseline a model must beat

B3's ratio map is fit on training years and training pixels only and filled
into the holdout + buffer from surrounding training pixels
(features/builder.py), so it is a legitimate predictor at held-out pixels.
"""
from __future__ import annotations

import numpy as np

from features.builder import FeatureBuilder

BASELINES = ("B0", "B1", "B2", "B3")


def b0(fb: FeatureBuilder, t: int) -> np.ndarray:
    return fb.bilinear(fb.c["power_ghi"][t])


def b1(fb: FeatureBuilder, t: int) -> np.ndarray:
    return np.clip(fb.kt_bilinear_toa(t), 0, 1) * fb.toa(fb.doy[t])


def b2(fb: FeatureBuilder, t: int) -> np.ndarray:
    ghi = b1(fb, t)
    if fb.shade is None or "svf" not in fb.static or fb.c["dhi_frac"] is None:
        return ghi
    fd = fb.bilinear(fb.c["dhi_frac"][t])
    diffuse = fd * ghi
    beam = ghi - diffuse
    return beam * fb.shade[fb.doy[t] - 1] + diffuse * fb.static["svf"]


def b3(fb: FeatureBuilder, t: int) -> np.ndarray:
    kt = fb.kt_bilinear(t) * fb.b3_ratio[fb.doy[t] - 1]
    return fb.clip(kt, t) * fb.norm(t)


def predictor(name: str):
    return {"B0": b0, "B1": b1, "B2": b2, "B3": b3}[name]

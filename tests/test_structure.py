"""Structure / texture metrics and the band-amplitude correction."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from evaluation import structure as S


def _fields(rng, T=12, H=40, W=40):
    return np.stack([gaussian_filter(rng.normal(size=(H, W)), 2) for _ in range(T)])


def test_decompose_reconstructs_on_valid(rng):
    x = _fields(rng)
    valid = np.ones(x.shape[1:], bool); valid[:5, :5] = False
    bands, coarse = S.decompose(x, valid, 10.0)
    assert np.allclose((sum(bands) + coarse)[:, valid], x[:, valid])


def test_perfect_prediction_scores_one(rng):
    x = _fields(rng) + 5
    valid = np.ones(x.shape[1:], bool)
    m = S.structure_metrics(x, x, np.zeros_like(x), valid, valid, 10.0)
    assert np.isclose(m["ssim"], 1) and np.isclose(m["resid_corr"], 1)
    assert all(np.isclose(m[k], 1) for k in m if k.startswith(("amp_", "corr_")))


def test_smoothed_prediction_loses_fine_amplitude_and_gains_restore_it(rng):
    x = _fields(rng, T=30)
    valid = np.ones(x.shape[1:], bool)
    sm = np.stack([gaussian_filter(f, 1.5) for f in x])
    m = S.structure_metrics(sm, x, np.zeros_like(x), valid, valid, 10.0)
    assert m["amp_<30km"] < 0.6
    g = S.fit_gains(sm, x, valid, valid, 10.0)
    assert g[0] > 1.5
    fixed = S.apply_gains(sm, g, valid, 10.0)
    m2 = S.structure_metrics(fixed, x, np.zeros_like(x), valid, valid, 10.0)
    assert abs(m2["amp_<30km"] - 1) < abs(m["amp_<30km"] - 1)
    assert np.isclose(m2["corr_<30km"], m["corr_<30km"], atol=0.02)   # scaling cannot move phase

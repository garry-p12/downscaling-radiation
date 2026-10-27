"""Swin wrapper shapes and losses; baselines on the synthetic store."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.swin import RadiationSwin, consistency_penalty, weighted_loss

TINY = {"patch_size": 2, "encoder": {"embed_dim": 16, "depths": [1, 1], "num_heads": [2, 2],
                                     "window_size": 4, "mlp_ratio": 2.0, "drop_path": 0.0}}


@pytest.mark.parametrize("hw", [(32, 32), (23, 30)])
def test_swin_shapes_and_zero_init(hw):
    m = RadiationSwin(5, TINY, n_out=3).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 5, *hw))
    assert y.shape == (2, 3, *hw)
    assert torch.allclose(y, torch.zeros_like(y)), "head starts at 'bilinear is right'"


def test_losses():
    y = torch.zeros(1, 1, 4, 4)
    m = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    w = torch.ones(1, 1, 4, 4)
    p = torch.full((1, 3, 4, 4), 0.1)
    assert weighted_loss(p, y, m, w, None) == pytest.approx(0.01)
    # pinball at q=0.1/0.5/0.9 with error -0.1 -> 0.09, 0.05, 0.01
    assert weighted_loss(p, y, m, w, [0.1, 0.5, 0.9]) == pytest.approx(0.05, rel=1e-5)
    y[0, 0, 0, 0] = float("nan")                         # NaN target masked out
    assert torch.isfinite(weighted_loss(p, y, m, w, None))
    enc = torch.zeros(1, 4, 4, dtype=torch.long)
    assert consistency_penalty(p[:, :1], p[:, :1], enc, m) == 0


def test_baselines_on_synthetic(synth_store, train_cfg):
    from features.builder import FeatureBuilder
    from models.baselines import BASELINES, predictor

    fb = FeatureBuilder(synth_store, train_cfg)
    t = int(np.nonzero(fb.split == "test")[0][0])
    for b in BASELINES:
        g = predictor(b)(fb, t)
        assert g.shape == (fb.ny, fb.nx) and np.nanmin(g) >= 0
        assert np.nanmax(g) <= fb.toa(fb.doy[t]).max() + 1e-6

"""Row sampling: in-mask, unbiased in time, stratified on the INPUT."""
from __future__ import annotations

import numpy as np

from training.sampling import candidate_rows, sky_class, stratified_choice


def test_candidates_respect_masks_and_cover_time(rng):
    days = np.zeros(1000, bool); days[100:900] = True
    pix = np.zeros((20, 20), bool); pix[5:15, 5:15] = True
    t, p = candidate_rows(days, pix, 5000, rng)            # sparse-draw branch
    assert days[t].all() and pix.ravel()[p].all()
    assert abs(t.mean() - 500) < 15, "truncation must not favour early days"


def test_equal_allocation_with_short_class(rng):
    cls = np.r_[np.zeros(50, int), np.ones(5000, int), np.full(5000, 2)]
    idx = stratified_choice(cls, 3000, rng)
    counts = np.bincount(cls[idx], minlength=3)
    assert counts[0] == 50 and counts.sum() == 3000 and abs(counts[1] - counts[2]) <= 1


def test_sky_class_bins():
    assert sky_class(np.array([0.1, 0.3, 0.64, 0.65, np.nan])).tolist() == [0, 1, 1, 2, -1]

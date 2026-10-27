"""Row sampling for the tabular model (plan 8: subsample days, stratified by sky).

~20 years x ~9000 pixels is ~7e7 pixel-days; XGBoost trains on a few million.
Stratification is on the POWER (INPUT) sky class: oversampling partly-cloudy
rows by an input variable re-weights the loss across feature space without
biasing E[y | X]. Stratifying on the TARGET's sky class would bias it.
"""
from __future__ import annotations

import numpy as np


def candidate_rows(day_mask: np.ndarray, pix_mask: np.ndarray, n: int,
                   rng: np.random.Generator):
    """Uniform random (day, flat pixel) pairs from day_mask x pix_mask."""
    days = np.nonzero(day_mask)[0]
    pix = np.nonzero(pix_mask.ravel())[0]
    if days.size == 0 or pix.size == 0:
        raise ValueError("empty day or pixel set")
    total = days.size * pix.size
    n = min(n, total)
    if total <= 50 * n:
        flat = rng.choice(total, size=n, replace=False)
    else:   # sparse draw: dedupe, then shuffle so truncation is not biased low
        flat = rng.permutation(np.unique(rng.integers(0, total, size=int(n * 1.05))))[:n]
    return days[flat // pix.size], pix[flat % pix.size]


def sky_class(kt: np.ndarray, bins=(0.3, 0.65)) -> np.ndarray:
    """0 overcast (< 0.3), 1 partly cloudy, 2 clear (>= 0.65); -1 for NaN."""
    c = np.digitize(kt, bins)
    return np.where(np.isfinite(kt), c, -1)


def stratified_choice(classes: np.ndarray, n: int, rng: np.random.Generator,
                      n_classes: int = 3) -> np.ndarray:
    """Indices with equal allocation per class; short classes give all they
    have and the remainder is spread over the others."""
    pools = [np.nonzero(classes == c)[0] for c in range(n_classes)]
    quota = {c: 0 for c in range(n_classes)}
    left, open_ = n, [c for c in range(n_classes) if pools[c].size]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for c in open_:
            take = min(share, pools[c].size - quota[c], left)
            quota[c] += take
            left -= take
            if quota[c] < pools[c].size:
                nxt.append(c)
            if left == 0:
                break
        open_ = nxt
    out = [rng.choice(pools[c], size=quota[c], replace=False) for c in range(n_classes)
           if quota[c]]
    return np.concatenate(out) if out else np.array([], int)

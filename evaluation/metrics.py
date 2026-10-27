"""Point metrics in kWh/m2/day (plan 7.1): RMSE, MBE, nRMSE, r, KGE, skill.

All NaN-aware over matched pairs: a pair counts only if both values are finite.
"""
from __future__ import annotations

import numpy as np


def _pairs(pred, obs):
    p = np.asarray(pred, dtype="float64").ravel()
    o = np.asarray(obs, dtype="float64").ravel()
    m = np.isfinite(p) & np.isfinite(o)
    return p[m], o[m]


def rmse(pred, obs) -> float:
    p, o = _pairs(pred, obs)
    return float(np.sqrt(np.mean((p - o) ** 2))) if p.size else np.nan


def mbe(pred, obs) -> float:
    p, o = _pairs(pred, obs)
    return float(np.mean(p - o)) if p.size else np.nan


def nrmse(pred, obs) -> float:
    """RMSE as % of the observed mean."""
    p, o = _pairs(pred, obs)
    return float(100 * np.sqrt(np.mean((p - o) ** 2)) / np.mean(o)) if p.size else np.nan


def pearson(pred, obs) -> float:
    p, o = _pairs(pred, obs)
    if p.size < 3 or p.std() == 0 or o.std() == 0:
        return np.nan
    return float(np.corrcoef(p, o)[0, 1])


def kge(pred, obs) -> float:
    """Kling-Gupta efficiency (Gupta et al. 2009): 1 - ||(r-1, a-1, b-1)||."""
    p, o = _pairs(pred, obs)
    if p.size < 3 or o.std() == 0 or o.mean() == 0:
        return np.nan
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 else 0.0
    alpha = p.std() / o.std()
    beta = p.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def skill(pred, obs, ref) -> float:
    """MSE skill score vs a reference prediction: 1 - MSE / MSE_ref.

    Evaluated on pairs where all three are finite, so model and reference are
    scored on identical samples.
    """
    p = np.asarray(pred, float).ravel()
    o = np.asarray(obs, float).ravel()
    r = np.asarray(ref, float).ravel()
    m = np.isfinite(p) & np.isfinite(o) & np.isfinite(r)
    if not m.any():
        return np.nan
    den = np.mean((r[m] - o[m]) ** 2)
    return float(1 - np.mean((p[m] - o[m]) ** 2) / den) if den > 0 else np.nan


def summary(pred, obs) -> dict:
    p, o = _pairs(pred, obs)
    return {"n": int(p.size), "rmse": rmse(p, o), "mbe": mbe(p, o),
            "nrmse_pct": nrmse(p, o), "r": pearson(p, o), "kge": kge(p, o),
            "obs_mean": float(o.mean()) if o.size else np.nan}

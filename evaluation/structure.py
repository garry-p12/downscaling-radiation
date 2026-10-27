"""Structure and texture metrics, and the post-hoc band-amplitude correction.

RMSE rewards smoothing: under MSE, a scale predicted with correlation rho is
reproduced at amplitude rho, so a model minimises its loss by SHRINKING what it
cannot place. The spectra of the first real run (eval/spectra.json) show every
model losing variance below ~300 km. These metrics measure what RMSE hides:

  ssim        structural similarity of the daily field (Gaussian window,
              data_range = that day's target range over the scored cells)
  resid_corr  spatial correlation of (pred - B1) with (target - B1) per day:
              skill on the structure interpolation does not already give
  band_amp    per scale band, sd(pred) / sd(target)   1.0 = right texture amount
  band_corr   per scale band, corr(pred, target)      1.0 = texture in the right place

Bands are differences of Gaussians (as in downscaling-temperature
evaluation/texture.py), computed with MASKED (normalised) convolution so the
Gulf and the domain edge do not leak into the land bands.

The correction (``fit_gains`` / ``apply_gains``) rescales each band's
day-to-day anomaly so its variance matches the target's. It is fitted on DEV
days x TRAINING pixels only — out of sample in time, which is where shrinkage
shows, and never on holdout truth (stricter than the temperature version).
Amplitude is all it can fix: coherence is invariant to scaling, so a band with
low band_corr gets louder, not more right.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

EDGES_KM = (30.0, 60.0, 120.0, 240.0)       # finest first; 10 km grid -> 3 cells+


def band_labels(edges_km=EDGES_KM) -> list[str]:
    return [f"<{edges_km[0]:.0f}km"] + [f"{edges_km[i - 1]:.0f}-{edges_km[i]:.0f}km"
                                         for i in range(1, len(edges_km))]


def _sigma_cells(cutoff_km: float, dx_km: float) -> float:
    """Gaussian sigma (cells) whose high-pass half-amplitude is at cutoff_km."""
    return float(np.sqrt(np.log(2)) / (np.pi * np.sqrt(2)) * cutoff_km / dx_km)


def masked_smooth(stack: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised Gaussian smoothing of (T, H, W) over ``valid`` cells only."""
    m = valid.astype("float64")
    den = gaussian_filter(m, sigma, mode="nearest")
    out = np.empty(stack.shape, dtype="float64")
    for k in range(stack.shape[0]):
        num = gaussian_filter(np.where(valid, np.nan_to_num(stack[k]), 0.0), sigma,
                              mode="nearest")
        out[k] = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    return out


def decompose(stack: np.ndarray, valid: np.ndarray, dx_km: float, edges_km=EDGES_KM):
    """(bands finest-first, coarse remainder); sum reconstructs ``stack`` on valid."""
    sm = [masked_smooth(stack, valid, _sigma_cells(e, dx_km)) for e in edges_km]
    bands = [stack - sm[0]] + [sm[i - 1] - sm[i] for i in range(1, len(sm))]
    return bands, sm[-1]


def ssim_day(pred: np.ndarray, target: np.ndarray, score: np.ndarray,
             valid: np.ndarray, sigma: float = 1.5, ws: int = 11) -> float:
    """SSIM of one (H, W) day averaged over ``score`` cells. Off-``valid`` cells
    are filled with the target's valid mean in BOTH fields (common mode)."""
    if score.sum() < 4:
        return np.nan
    fill = np.nanmean(target[valid])
    p = np.where(valid & np.isfinite(pred), pred, fill).astype("float64")
    t = np.where(valid & np.isfinite(target), target, fill).astype("float64")
    dr = float(np.nanmax(target[score]) - np.nanmin(target[score]))
    if not np.isfinite(dr) or dr <= 0:
        return np.nan
    trunc = ((ws - 1) / 2) / sigma
    g = lambda a: gaussian_filter(a, sigma, truncate=trunc, mode="nearest")  # noqa: E731
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mp, mt = g(p), g(t)
    sp, st, spt = g(p * p) - mp * mp, g(t * t) - mt * mt, g(p * t) - mp * mt
    s = ((2 * mp * mt + c1) * (2 * spt + c2)) / ((mp ** 2 + mt ** 2 + c1) * (sp + st + c2))
    return float(np.nanmean(s[score]))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def structure_metrics(pred: np.ndarray, target: np.ndarray, base: np.ndarray,
                      score: np.ndarray, valid: np.ndarray, dx_km: float,
                      bands_t=None, edges_km=EDGES_KM) -> dict:
    """All structure metrics for (T, H, W) stacks scored over ``score`` cells.

    ``bands_t`` (the target's decomposition) can be passed in to avoid
    recomputing it for every model.
    """
    T = pred.shape[0]
    out = {"ssim": float(np.nanmean([ssim_day(pred[k], target[k], score, valid)
                                     for k in range(T)])),
           "resid_corr": float(np.nanmean([_corr((pred[k] - base[k])[score],
                                                 (target[k] - base[k])[score])
                                           for k in range(T)]))}
    bp, _ = decompose(pred, valid, dx_km, edges_km)
    bt = bands_t if bands_t is not None else decompose(target, valid, dx_km, edges_km)[0]
    for lab, p, t in zip(band_labels(edges_km), bp, bt):
        ps, ts = p[:, score], t[:, score]
        ok = np.isfinite(ps) & np.isfinite(ts)
        out[f"amp_{lab}"] = float(ps[ok].std() / ts[ok].std()) if ok.any() else np.nan
        out[f"corr_{lab}"] = _corr(ps, ts)
    return out


def fit_gains(pred: np.ndarray, target: np.ndarray, fit_cells: np.ndarray,
              valid: np.ndarray, dx_km: float, edges_km=EDGES_KM,
              clip=(1.0, 4.0)) -> list[float]:
    """Per-band gains matching the prediction's day-to-day band variance to the
    target's over ``fit_cells`` (dev days x training pixels). Clipped below at
    1: a band the model over-produces is left alone, not damped."""
    bp, _ = decompose(pred, valid, dx_km, edges_km)
    bt, _ = decompose(target, valid, dx_km, edges_km)
    gains = []
    for p, t in zip(bp, bt):
        pa = (p - np.nanmean(p, 0))[:, fit_cells]
        ta = (t - np.nanmean(t, 0))[:, fit_cells]
        sp, st = np.nanstd(pa), np.nanstd(ta)
        gains.append(float(np.clip(st / sp if sp > 0 else 1.0, *clip)))
    return gains


def apply_gains(stack: np.ndarray, gains, valid: np.ndarray, dx_km: float,
                edges_km=EDGES_KM) -> np.ndarray:
    """Scale each band's day-to-day anomaly about the stack's OWN per-cell time
    mean (no truth used), and reassemble. Off-``valid`` cells pass through."""
    bands, coarse = decompose(stack, valid, dx_km, edges_km)
    out = coarse.copy()
    for b, g in zip(bands, gains):
        bm = np.nanmean(b, 0)
        out += bm + g * (b - bm)
    return np.where(valid[None], out, stack).astype("float32")

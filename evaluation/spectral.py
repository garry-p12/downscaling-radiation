"""Power spectra of predicted vs target fields (plan 7.4).

Does the model put real variance at fine scales, or smooth it away? Radial
amplitude ratio sqrt(PSD_pred / PSD_target) and coherence per wavelength.

radial_spectrum / effective_resolution / coherence_resolution are vendored
verbatim from downscaling-temperature@1d2c3bb (evaluation/spectral.py), so the
two variables' effective-resolution numbers are computed identically.
"""
from __future__ import annotations

import numpy as np

AMP_FLOOR = np.sqrt(0.75)     # the paper's effective-resolution threshold


def radial_spectrum(pred: np.ndarray, truth: np.ndarray,
                    dx_km: float, window: bool = True) -> list[dict]:
    """Per-wavenumber amplitude ratio and coherence for (T, H, W) stacks.

    `dx_km` HAS NO DEFAULT ON PURPOSE. It used to default to 10.0 and no call
    site ever overrode it, so the value was really "whatever grid the project
    happened to be on". On a 5 km grid that default reports every wavelength
    twice as coarse as it is, and because the curves keep their shape, nothing
    about the output looks wrong. Pass `store.dx_km`.
    """
    _, ny, nx = pred.shape
    w = np.outer(np.hanning(ny), np.hanning(nx)) if window else 1.0
    fp, ft = np.fft.rfft2(pred * w), np.fft.rfft2(truth * w)

    ky = np.fft.fftfreq(ny, dx_km)[:, None]
    kx = np.fft.rfftfreq(nx, dx_km)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    nb = max(2, min(ny, nx) // 2)
    edges = np.linspace(0, kr.max(), nb + 1)

    out = []
    for b in range(nb):
        m = (kr >= edges[b]) & (kr < edges[b + 1])
        if not m.any():
            continue
        pp = (np.abs(fp) ** 2)[:, m].sum(1).mean()
        tt = (np.abs(ft) ** 2)[:, m].sum(1).mean()
        ct = np.real(fp * np.conj(ft))[:, m].sum(1).mean()
        k = 0.5 * (edges[b] + edges[b + 1])
        if k <= 0 or tt <= 0:
            continue
        out.append({"wavelength_km": float(1.0 / k),
                    "amplitude_ratio": float(np.sqrt(pp / tt)),
                    "coherence": float(ct / np.sqrt(pp * tt))})
    return out


def _cross(rows, key, level, falling=True):
    """Wavelength where `key` crosses `level`, interpolated in log-wavelength.

    Returns NaN if it never crosses within the measured bands.
    """
    r = sorted(rows, key=lambda x: -x["wavelength_km"])
    for i in range(1, len(r)):
        a, b = r[i - 1][key], r[i][key]
        hit = (b < level <= a) if falling else (b > level >= a)
        if hit and a != b:
            w0, w1 = np.log(r[i - 1]["wavelength_km"]), np.log(r[i]["wavelength_km"])
            f = (a - level) / (a - b)
            return float(np.exp(w0 + f * (w1 - w0)))
    return float("nan")


def effective_resolution(rows: list[dict], interpolate: bool = True) -> float | None:
    """Coarsest wavelength at which the amplitude ratio is still above the floor.

    Scanned coarse to fine, stopping at the FIRST crossing. Scanning for the
    last one would be wrong: the amplitude ratio rises again near the Nyquist
    corner, but with coherence ~0 that is uncorrelated noise, not recovered
    signal — the paper calls this the noise-based regime.

    INTERPOLATE, DO NOT SNAP TO A BAND CENTRE. The radial bands are linear in
    wavenumber, so above 60 km on a 37x35 crop there are only four of them —
    495, 165, 99 and 71 km. Returning the centre of the first band below the
    floor gave an identical 165 km for four arms whose curves are plainly
    different, and it understates the true crossing by roughly 2.5x because a
    curve that is 0.88 at 495 km and 0.78 at 165 km crosses 0.866 near 405.
    The metric could neither distinguish the arms nor state the right number.
    `interpolate=False` restores the old behaviour for reproducing figures that
    were published with it.
    """
    if interpolate:
        v = _cross(rows, "amplitude_ratio", AMP_FLOOR)
        return None if np.isnan(v) else v
    for r in sorted(rows, key=lambda r: -r["wavelength_km"]):
        if r["amplitude_ratio"] < AMP_FLOOR:
            return r["wavelength_km"]
    return None


def coherence_resolution(rows: list[dict], level: float = 0.5) -> float | None:
    """Wavelength where coherence with truth falls below `level`.

    Better behaved than the amplitude crossing and usually the more honest
    summary: the amplitude ratio is non-monotone (it can rise again at fine
    scales where the model is injecting uncorrelated texture), so a
    first-crossing statistic on it is fragile. Coherence falls monotonically in
    practice, and "the scale below which the model stops placing structure
    correctly" is the question a reader actually has.

    None means it never dropped below `level` within the measured bands — a
    lower bound on skill, not a failure.
    """
    v = _cross(rows, "coherence", level)
    return None if np.isnan(v) else v



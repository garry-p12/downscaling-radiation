"""Per-variable z-score normalization with optional per-season stats.

Vendored verbatim from downscaling-temperature@1d2c3bb (common/normalize.py).

Stats are fit on the training split only and persisted alongside the dataset
so train/val/test/inference all use identical constants.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Normalizer:
    """z-score per variable. ``log1p=True`` pre-transforms skewed,
    non-negative variables (unused by the temperature build; kept because it
    is part of the persisted stats format)."""

    def __init__(self, stats: dict | None = None):
        self.stats: dict = stats or {}

    def fit(self, var: str, values: np.ndarray, *, log1p: bool = False,
            season: str | None = None) -> None:
        v = np.asarray(values, dtype="float64")
        v = v[np.isfinite(v)]
        if log1p:
            v = np.log1p(np.clip(v, 0, None))
        key = self._key(var, season)
        self.stats[key] = {
            "mean": float(v.mean()),
            "std": float(v.std() + 1e-8),
            "log1p": bool(log1p),
        }

    def transform(self, var: str, values, season: str | None = None):
        s = self.stats[self._key(var, season)]
        v = values
        if s["log1p"]:
            v = np.log1p(np.clip(v, 0, None))
        return (v - s["mean"]) / s["std"]

    def inverse(self, var: str, values, season: str | None = None):
        s = self.stats[self._key(var, season)]
        v = values * s["std"] + s["mean"]
        if s["log1p"]:
            v = np.expm1(v)
        return v

    @staticmethod
    def _key(var: str, season: str | None) -> str:
        return var if season is None else f"{var}__{season}"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        return cls(json.loads(Path(path).read_text()))

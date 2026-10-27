"""M1: XGBoost on the clearness-index residual kt_nsrdb - kt_bilinear.

Prediction: kt_hat = clip(kt_bilinear + r_hat) and GHI = kt_hat x TOA_fine,
with the physical clip of plan 2.2 (features/builder.physical_clip).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features.builder import FeatureBuilder

LOSS_WEIGHTS = ("none", "toa", "energy")


def loss_weight(kind: str, toa: np.ndarray) -> np.ndarray | None:
    """Sample weights for MSE in kt space.

    none    plain MSE on kt
    toa     errors weighted by TOA: proportional to energy impact (plan 5)
    energy  TOA^2, i.e. exactly MSE in kWh/m2/day space
    """
    if kind == "none":
        return None
    w = toa / np.mean(toa)
    return w if kind == "toa" else w ** 2


class XGBPredictor:
    """Load a trained run and predict full GHI fields day by day."""

    def __init__(self, run_dir: str | Path, fb: FeatureBuilder):
        import xgboost as xgb

        run_dir = Path(run_dir)
        self.meta = json.loads((run_dir / "meta.json").read_text())
        self.names = self.meta["features"]
        self.booster = xgb.Booster()
        self.booster.load_model(str(run_dir / "model.json"))
        self.fb = fb

    def residual(self, t: int) -> np.ndarray:
        import xgboost as xgb

        S = self.fb.stack(t, self.names).reshape(len(self.names), -1).T
        r = self.booster.predict(xgb.DMatrix(S, feature_names=self.names))
        return r.reshape(self.fb.ny, self.fb.nx)

    def __call__(self, t: int) -> np.ndarray:
        kt = self.fb.clip(self.fb.kt_bilinear(t) + self.residual(t), t)
        return kt * self.fb.toa(self.fb.doy[t])

"""Train M1 (XGBoost residual-kt model) for Track A or B.

    python -m training.train_xgb --track A
    python -m training.train_xgb --track B --loss-weight toa

Rows: training years x training pixels (holdout + buffer excluded), finite
target, subsampled with POWER-sky-class stratification. Early stopping on
dev (val years x training pixels). With ``loss_weight: auto`` all three loss
weightings are trained and the one with the lowest dev RMSE in kWh/m2/day is
kept (plan 5: "compare ... on the dev set and keep whichever wins").

Writes runs/<store>/xgb_<track>/: model.json, meta.json (features, dev scores
per weighting), importance.csv (gain and mean |SHAP|). SHAP values come from
XGBoost's own TreeSHAP (pred_contribs), so the shap package is not needed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.tracking import Tracker
from evaluation.metrics import rmse
from models.xgb import LOSS_WEIGHTS, loss_weight
from training.common import builder, dump, load_all, run_dir
from training.sampling import candidate_rows, sky_class, stratified_choice


def sample(fb, day_mask, n, rng, stratify: bool):
    pool_t, pool_p = candidate_rows(day_mask, fb.train_pix, n * (3 if stratify else 1), rng)
    kt_target = fb.kt.reshape(len(fb.times), -1)[pool_t, pool_p]
    keep = np.isfinite(kt_target)
    pool_t, pool_p = pool_t[keep], pool_p[keep]
    if stratify:
        cls = sky_class(fb.gather_kt_bilinear(pool_t, pool_p))
        idx = stratified_choice(cls, min(n, pool_t.size), rng)
        pool_t, pool_p = pool_t[idx], pool_p[idx]
    return pool_t[:n], pool_p[:n]


def main(track: str, weight: str | None, n_rows: int | None, tag: str | None = None,
         drop: list[str] | None = None) -> None:
    import xgboost as xgb

    dcfg, tcfg = load_all()
    xc = tcfg["xgboost"]
    rng = np.random.default_rng(int(xc.get("seed", 0)))
    fb = builder(dcfg, tcfg)
    names = [n for n in fb.track(track) if n not in set(drop or [])]
    out = run_dir(dcfg, tcfg, f"xgb_{track}" + (f"_{tag}" if tag else ""))
    print(f"[xgb] track {track}: {len(names)} features -> {out}")

    strat = xc.get("stratify_on") is not None
    tr_t, tr_p = sample(fb, fb.train_days, int(n_rows or xc["n_rows"]), rng, strat)
    dv_t, dv_p = sample(fb, fb.split == "val", int(xc["n_dev_rows"]), rng, False)
    Xtr, ytr, atr = fb.table(tr_t, tr_p, names)
    Xdv, ydv, adv = fb.table(dv_t, dv_p, names)
    print(f"[xgb] train rows {len(ytr)}, dev rows {len(ydv)}; train sky mix "
          f"{np.bincount(sky_class(atr['kt_bil']) + 1, minlength=4)[1:].tolist()}")

    tracker = Tracker(tcfg["logging"]["wandb"], dict(tcfg), {"model": f"xgb_{track}"})
    weight = weight or xc.get("loss_weight", "auto")
    kinds = LOSS_WEIGHTS if weight == "auto" else (weight,)
    params = dict(xc["params"], objective="reg:squarederror", eval_metric="rmse",
                  seed=int(xc.get("seed", 0)))
    ddv = xgb.DMatrix(Xdv, ydv, feature_names=names)
    results, best = {}, None
    for kind in kinds:
        dtr = xgb.DMatrix(Xtr, ytr, weight=loss_weight(kind, atr["toa"]),
                          feature_names=names)
        bst = xgb.train(params, dtr, int(xc["num_boost_round"]),
                        evals=[(ddv, "dev")],
                        early_stopping_rounds=int(xc["early_stopping_rounds"]),
                        verbose_eval=200)
        r = bst.predict(ddv, iteration_range=(0, bst.best_iteration + 1))
        ghi = fb.reconstruct(adv["kt_bil"], r, adv["toa"], adv["kt_ceil"])
        score = rmse(ghi, adv["ghi"])
        results[kind] = {"dev_rmse_kwh": score, "best_iteration": bst.best_iteration}
        print(f"[xgb] weight={kind}: dev RMSE {score:.4f} kWh/m2/day "
              f"@ {bst.best_iteration} trees")
        tracker.log({f"dev_rmse_kwh/{kind}": score})
        if best is None or score < results[best[0]]["dev_rmse_kwh"]:
            best = (kind, bst)
    kind, bst = best
    bst = bst[: bst.best_iteration + 1]
    bst.save_model(str(out / "model.json"))

    gain = bst.get_score(importance_type="gain")
    ns = min(int(xc.get("shap_rows", 20000)), len(ydv))
    contrib = bst.predict(xgb.DMatrix(Xdv[:ns], feature_names=names), pred_contribs=True)
    imp = pd.DataFrame({"feature": names,
                        "gain": [gain.get(n, 0.0) for n in names],
                        "mean_abs_shap": np.abs(contrib[:, :-1]).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out / "importance.csv", index=False)
    print(imp.head(12).to_string(index=False))

    dump(out / "meta.json", {"track": track, "features": names, "loss_weight": kind,
                             "dev": results, "n_train": int(len(ytr))})
    tracker.summary("dev_rmse_kwh", results[kind]["dev_rmse_kwh"])
    tracker.finish()
    print(f"[xgb] kept weight={kind}; wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["A", "B", "C"], default="A")
    ap.add_argument("--loss-weight", choices=[*LOSS_WEIGHTS, "auto"], default=None)
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--tag", default=None, help="run-name suffix")
    ap.add_argument("--drop", nargs="+", default=None,
                    help="features to remove from the track (ablations)")
    a = ap.parse_args()
    main(a.track, a.loss_weight, a.n_rows, a.tag, a.drop)

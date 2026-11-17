"""1 km radiation fields on two case days, for the resolution figure of Paper 1.

Same window, models and cache layout as evaluation/austin.py, but only on the requested dates (a full 1 km test-period
cache would be several GB). Models are the 1 km Swin runs (B + COD track) on the NSRDB target and on the anchored
targets; any model whose checkpoint is missing is skipped.

    sbatch -A ATM23014 -p gh-dev -t 00:40:00 -J rad_case1 --wrap \
      "DOWNSCALE_CONFIG_data=configs/data_vista_1km.yaml $WORK/miniforge3/envs/dsc/bin/python -m evaluation.austin_1km_case"

Writes runs/figures/paper_rad_case_1km.npz: for each date d and field v, r1_<d>_<v>; plus r1_lat, r1_lon, r1_valid.
"""
from __future__ import annotations
import contextlib, sys
import numpy as np
import pandas as pd
import evaluation.austin as A

DATES = sys.argv[1:] or ["2021-03-14", "2021-07-08"]
MODELS = {k: (g.replace("data_store_goes", "data_store_1km"), n) for k, (g, n) in A.MODELS_10KM.items()}


def main():
    with contextlib.suppress(ImportError):
        import xgboost  # noqa: F401
    from data.build_store import open_store
    from evaluation.evaluate import predictor
    from training.common import builder, load_all
    from training.train_swin import SwinPredictor

    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    ds = open_store(dcfg)
    lat, lon = A._latlon(dcfg)
    B = A.BOX
    inbox = (lat >= B[0]) & (lat <= B[1]) & (lon >= B[2]) & (lon <= B[3])
    rr, cc = np.nonzero(inbox)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    times = pd.DatetimeIndex(fb.times)
    preds = {"B1": predictor("B1")}
    for k, (grp, name) in MODELS.items():
        ck = A.RUNS / grp / name / "best.pt"
        if ck.exists():
            preds[k] = SwinPredictor.from_checkpoint(ck, fb)
        else:
            print(f"[1km] skip {k}: no {ck}", flush=True)
    out = {"r1_lat": lat[r0:r1, c0:c1].astype("float32"), "r1_lon": lon[r0:r1, c0:c1].astype("float32"),
           "r1_valid": ds["valid"].values[r0:r1, c0:c1].astype(bool)}
    for d in DATES:
        t = int(np.where(times == pd.Timestamp(d))[0][0])
        for k, p in preds.items():
            f = p(fb, t) if k == "B1" else p(t)
            out[f"r1_{d}_{k}"] = np.asarray(f[r0:r1, c0:c1], "float32")
        out[f"r1_{d}_NSRDB"] = np.asarray(fb.ghi[t][r0:r1, c0:c1], "float32")
        out[f"r1_{d}_POWER"] = fb.c["power_ghi"][t].ravel()[fb.enc].reshape(fb.ny, fb.nx)[r0:r1, c0:c1].astype("float32")
        for v, key in (("tcld", "goes_tcld"),):
            with contextlib.suppress(Exception):
                out[f"r1_{d}_{v}"] = np.asarray(fb.dyn[key][t][r0:r1, c0:c1], "float32")
        print(f"[1km] {d} done", flush=True)
    p = A.RUNS / "figures" / "paper_rad_case_1km.npz"
    np.savez_compressed(p, **out)
    print(f"[1km] wrote {p}")


if __name__ == "__main__":
    main()

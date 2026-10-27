"""Probabilistic evaluation of the residual-diffusion ensembles (method C).

    python -m evaluation.probabilistic --runs diff_B --members 16

Scored on the test years, against NSRDB (cells temporal and both) and against
ground stations (independent truth). For each diffusion run, and for two
reference forecasts built from the SAME mean model:

    mean        the deterministic mean model (CRPS of a point = MAE)
    mean+gauss  the mean with Gaussian spread whose sd is the mean model's
                dev-year RMSE in index units, x the day's normaliser (a fair,
                cheap probabilistic baseline: if diffusion cannot beat it, the
                generative part adds nothing probabilistic)

Metrics (kWh/m2/day): CRPS, RMSE of the ensemble mean, spread (mean member sd)
and spread/skill ratio (1 = calibrated), rank histogram vs NSRDB, and the
texture of ONE member (structure.structure_metrics: SSIM, band placement and
amplitude) -- the realism a single draw should have, unlike a mean.

Writes runs/<group>/eval/prob_metrics.csv, prob_structure.csv,
prob_stations.csv, rank_hist.json and ens_<run>.npz (first 8 members, test
days, float16) for the figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as gauss

from common.splits import evaluation_cells
from evaluation import structure as S
from training.common import builder, load_all, run_dir, runs_root
from training.train_diffusion import DiffusionPredictor, crps_ensemble


def crps_gauss(mu, sd, obs):
    z = (obs - mu) / sd
    return sd * (z * (2 * gauss.cdf(z) - 1) + 2 * gauss.pdf(z) - 1 / np.sqrt(np.pi))


def rank_hist(ens: np.ndarray, obs: np.ndarray) -> list[int]:
    """Counts of obs rank among M members (M+1 bins), ties broken at random."""
    rng = np.random.default_rng(0)
    below = (ens < obs[None]).sum(0)
    ties = (ens == obs[None]).sum(0)
    r = below + np.floor(rng.random(obs.shape) * (ties + 1)).astype(int)
    return np.bincount(r, minlength=ens.shape[0] + 1).tolist()


def station_pixels(fb, dcfg):
    root = Path(dcfg["sources"]["stations"]["out_dir"])
    if not (root / "stations.csv").exists():
        return pd.DataFrame(columns=["station_id", "row", "col", "region"]), pd.DataFrame()
    meta = pd.read_csv(root / "stations.csv")
    daily = pd.read_csv(root / "station_daily.csv", parse_dates=["date"])
    meta = meta[meta.in_grid]
    meta = meta[[bool(fb.valid[r, c]) for r, c in zip(meta.row, meta.col)]]
    return meta, daily


def main(runs: list[str], members: int) -> None:
    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    ecfg = tcfg["eval"]
    out = run_dir(dcfg, tcfg, "eval")
    cells = evaluation_cells(fb.split, fb.holdout, fb.exclusion, fb.valid,
                             ecfg.get("spatial_split", "train"),
                             int(ecfg.get("spatial_day_stride", 1)))
    days = np.nonzero(cells["temporal"][0])[0]
    obs = fb.ghi[days]
    bt, _ = S.decompose(obs, fb.valid, fb.res_km)
    from models.baselines import b1

    base = np.stack([b1(fb, int(t)) for t in days])
    meta, daily = station_pixels(fb, dcfg)
    tindex = pd.DatetimeIndex(fb.times)

    rows, srows, strows, hists = [], [], [], {}
    for run in runs:
        ck = runs_root(dcfg, tcfg) / run / "best.pt"
        pred = DiffusionPredictor.from_checkpoint(ck, fb, dcfg, tcfg, days)
        mean_pred = pred.c.mean
        ens = np.zeros((len(days), members, fb.ny, fb.nx), np.float32)
        for i, t in enumerate(days):
            ens[i] = pred.ensemble(int(t), members)
            if i % 100 == 0:
                print(f"[prob] {run}: day {i}/{len(days)}", flush=True)
        mu = np.stack([mean_pred(int(t)) for t in days])
        # Gaussian baseline: dev-year RMSE of the mean model, index units
        val = np.nonzero(fb.split == "val")[0]
        dk = np.concatenate([((mean_pred(int(t)) - fb.ghi[t]) / fb.norm(int(t)))[fb.train_pix]
                             for t in val])
        sd_k = float(np.sqrt(np.nanmean(dk ** 2)))
        sd = np.stack([sd_k * fb.norm(int(t)) for t in days])
        np.savez_compressed(out / f"ens_{run}.npz", days=days, ens=ens[:, :8].astype(np.float16))

        for cell in ("temporal", "both"):
            pix = cells[cell][1]
            o = obs[:, pix].ravel()
            E = ens[:, :, pix].transpose(1, 0, 2).reshape(members, -1)
            m_ = mu[:, pix].ravel()
            s_ = sd[:, pix].ravel()
            ok = np.isfinite(o) & np.isfinite(E).all(0)
            o, E, m_, s_ = o[ok], E[:, ok], m_[ok], s_[ok]
            em = E.mean(0)
            spread = float(np.sqrt(np.mean(E.var(0, ddof=1))))
            skill = float(np.sqrt(np.mean((em - o) ** 2)))
            rows += [
                {"model": run, "cell": cell, "crps": crps_ensemble(E, o), "rmse_mean": skill,
                 "rmse_member": float(np.sqrt(np.mean((E[0] - o) ** 2))),
                 "spread": spread, "spread_skill": spread / skill * np.sqrt((members + 1) / members)},
                {"model": f"{run}:mean", "cell": cell, "crps": float(np.mean(np.abs(m_ - o))),
                 "rmse_mean": float(np.sqrt(np.mean((m_ - o) ** 2)))},
                {"model": f"{run}:mean+gauss", "cell": cell,
                 "crps": float(np.mean(crps_gauss(m_, s_, o))),
                 "rmse_mean": float(np.sqrt(np.mean((m_ - o) ** 2))),
                 "spread": float(np.sqrt(np.mean(s_ ** 2)))},
            ]
            if cell == "both":
                hists[run] = rank_hist(E, o)
            for lab, stack in ((f"{run}:member", ens[:, 0]), (f"{run}:ensmean", ens.mean(1)),
                               (f"{run}:mean", mu)):
                srows.append({"model": lab, "cell": cell,
                              **S.structure_metrics(stack, obs, base, pix, fb.valid,
                                                    fb.res_km, bands_t=bt)})

        # Stations: CRPS against the ground, test days.
        pos = {int(t): i for i, t in enumerate(days)}
        for _, s in meta.iterrows():
            d = daily[daily.station_id == s.station_id]
            ti = tindex.get_indexer(d.date)
            keep = np.array([t in pos for t in ti])
            if not keep.any():
                continue
            k = np.array([pos[t] for t in ti[keep]])
            ob = d.ghi_kwh.values[keep]
            E = ens[k, :, s.row, s.col].T
            ok = np.isfinite(ob) & np.isfinite(E).all(0)
            if ok.sum() < 30:
                continue
            ob, E = ob[ok], E[:, ok]
            m_, s_ = mu[k, s.row, s.col][ok], sd[k, s.row, s.col][ok]
            nsr = fb.refs["NSRDB"][days[k], s.row, s.col][ok] if "NSRDB" in fb.refs else None
            rec = {"model": run, "station_id": s.station_id, "network": s.get("network", ""),
                   "region": s.region, "n": int(ok.sum()),
                   "crps_ens": crps_ensemble(E, ob), "crps_gauss": float(np.mean(crps_gauss(m_, s_, ob))),
                   "mae_mean": float(np.mean(np.abs(m_ - ob))),
                   "mae_nsrdb": float(np.nanmean(np.abs(nsr - ob))) if nsr is not None else np.nan}
            # Bias-removed versions: each forecast shifted by its own mean error here
            b_e, b_m = float(np.mean(E.mean(0) - ob)), float(np.mean(m_ - ob))
            rec |= {"crps_ens_c": crps_ensemble(E - b_e, ob),
                    "crps_gauss_c": float(np.mean(crps_gauss(m_ - b_m, s_, ob))),
                    "mae_mean_c": float(np.mean(np.abs(m_ - b_m - ob)))}
            strows.append(rec)

    pd.DataFrame(rows).to_csv(out / "prob_metrics.csv", index=False)
    pd.DataFrame(srows).to_csv(out / "prob_structure.csv", index=False)
    st = pd.DataFrame(strows)
    st.to_csv(out / "prob_stations.csv", index=False)
    (out / "rank_hist.json").write_text(json.dumps(hists))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    cols = ["model", "cell", "ssim"] + [c for c in pd.DataFrame(srows).columns if c.startswith(("amp_", "corr_"))]
    print(pd.DataFrame(srows)[cols].round(3).to_string(index=False))
    if not st.empty:
        w = st.n.values
        agg = {c: float(np.average(st[c], weights=w)) for c in st.columns if c.startswith(("crps", "mae"))}
        print("stations (n-weighted):", {k: round(v, 4) for k, v in agg.items()})
    print(f"[prob] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["diff_B"])
    ap.add_argument("--members", type=int, default=None)
    a = ap.parse_args()
    _, tc = load_all()
    main(a.runs, int(a.members or tc.get("diffusion", {}).get("members", 16)))

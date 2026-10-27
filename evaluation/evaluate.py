"""Phase 6: score every model on the three test cells, stratified, plus maps,
spectra and the independent station comparison.

    python -m evaluation.evaluate                 # all baselines + every trained run
    python -m evaluation.evaluate --models B0 B3 xgb_A

Outputs (runs/<store>/eval/):
    metrics.csv     model x cell x stratum: n, RMSE, MBE, nRMSE, r, KGE,
                    skill vs B0, skill vs B3                        (plan 7.1-7.2)
    maps.npz/.png   per-pixel RMSE and correlation over test years   (plan 7.4)
    spectra.json    amplitude ratio + coherence per wavelength       (plan 7.4)
    stations.csv    model vs station, model vs NSRDB, NSRDB vs station (plan 7.3)

Strata: sky class from the TARGET kt (overcast < 0.3 <= partly < 0.65 <=
clear), season, terrain (rough where 1 km elevation std > threshold: Hill
Country escarpment, Trans-Pecos), coastal vs inland.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common.splits import evaluation_cells
from evaluation import metrics as M
from evaluation import structure as S
from evaluation.spectral import coherence_resolution, effective_resolution, radial_spectrum
from models.baselines import BASELINES, predictor
from training.common import builder, load_all, run_dir, runs_root

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def load_predictors(fb, dcfg, tcfg, wanted: list[str] | None):
    # xgboost BEFORE torch: each ships its own OpenMP runtime, and on macOS
    # loading torch's first makes xgboost segfault.
    with contextlib.suppress(ImportError):
        import xgboost  # noqa: F401
    preds = {b: functools.partial(predictor(b), fb) for b in BASELINES}
    root = runs_root(dcfg, tcfg)
    for d in sorted(root.glob("xgb_*")):
        if (d / "model.json").exists():
            from models.xgb import XGBPredictor

            preds[d.name] = XGBPredictor(d, fb)
    for d in sorted(root.glob("swin_*")):
        if (d / "best.pt").exists():
            from training.train_swin import SwinPredictor

            preds[d.name] = SwinPredictor.from_checkpoint(d / "best.pt", fb)
    if wanted:
        preds = {k: v for k, v in preds.items() if k in wanted or k in ("B0", "B3")}
    return preds


def stratum_labels(fb, days: np.ndarray, ecfg) -> dict[str, np.ndarray]:
    """(D, ny, nx) string labels per stratification axis."""
    D, sh = days.size, (fb.ny, fb.nx)
    lo, hi = ecfg.get("sky_bins", [0.3, 0.65])
    kt = fb.ghi[days] / np.stack([fb.toa(fb.doy[t]) for t in days])   # TOA kt, any mode
    sky = np.where(kt < lo, "overcast", np.where(kt < hi, "partly", "clear"))
    sky = np.where(np.isfinite(kt), sky, "none")
    months = pd.DatetimeIndex(fb.times[days]).month
    season = np.broadcast_to(np.array([SEASONS[m] for m in months])[:, None, None], (D, *sh))
    out = {"sky": sky, "season": season}
    if "elev_std" in fb.static:
        rough = fb.static["elev_std"] > float(ecfg.get("rough_elev_std_m", 60))
        out["terrain"] = np.broadcast_to(np.where(rough, "rough", "flat"), (D, *sh))
    if "coastal_dist" in fb.static:
        coast = fb.static["coastal_dist"] < float(ecfg.get("coastal_km", 50))
        out["coast"] = np.broadcast_to(np.where(coast, "coastal", "inland"), (D, *sh))
    return out


def score(obs, preds: dict, labels: dict, cell: str) -> list[dict]:
    rows = []
    groups = [("all", "all", np.ones(obs.shape, bool))]
    for axis, lab in labels.items():
        for v in np.unique(lab):
            if v != "none":
                groups.append((axis, str(v), lab == v))
    for axis, val, sel in groups:
        o = obs[sel]
        if np.isfinite(o).sum() < 10:
            continue
        for name, p in preds.items():
            r = {"model": name, "cell": cell, "stratum": axis, "value": val,
                 **M.summary(p[sel], o),
                 "skill_vs_B0": M.skill(p[sel], o, preds["B0"][sel]),
                 "skill_vs_B3": M.skill(p[sel], o, preds["B3"][sel])}
            rows.append(r)
    return rows


def pixel_maps(obs: np.ndarray, pred: np.ndarray):
    """Per-pixel RMSE and temporal correlation over a (D, ny, nx) stack."""
    good = np.isfinite(obs) & np.isfinite(pred)
    n = good.sum(0)
    o = np.where(good, obs, np.nan)
    p = np.where(good, pred, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rm = np.sqrt(np.nanmean((p - o) ** 2, axis=0))
        oa = o - np.nanmean(o, 0)
        pa = p - np.nanmean(p, 0)
        corr = np.nansum(oa * pa, 0) / np.sqrt(np.nansum(oa ** 2, 0) * np.nansum(pa ** 2, 0))
    rm[n < 10] = np.nan
    corr[n < 10] = np.nan
    return rm, corr


def spectra(obs: np.ndarray, pred: np.ndarray, valid: np.ndarray, dx_km: float):
    """Spectra of daily ANOMALY fields over the valid-pixel bounding box.

    Each day's valid-area mean is removed and invalid cells (ocean) set to 0,
    identically for prediction and target.
    """
    rows_ = np.nonzero(valid.any(1))[0]
    cols_ = np.nonzero(valid.any(0))[0]
    sl = (slice(None), slice(rows_[0], rows_[-1] + 1), slice(cols_[0], cols_[-1] + 1))
    o, p, v = obs[sl], pred[sl], valid[sl[1:]]

    def anom(a):
        a = np.where(v[None] & np.isfinite(a), a, np.nan)
        a = a - np.nanmean(a, axis=(1, 2), keepdims=True)
        return np.nan_to_num(a)

    return radial_spectrum(anom(p), anom(o), dx_km=dx_km)


def station_eval(fb, dcfg, preds_test: dict, test_days: np.ndarray):
    """Independent-truth check at ground stations over the test days.

    Returns (per-station table, pooled summary). For every model, at the
    station pixels:
      rmse_stn / mbe_stn        model vs station
      rmse_nsrdb                model vs NSRDB (same pixel-days)
      skill_stn_vs_B1           1 - MSE_stn(model) / MSE_stn(B1)
      skill_nsrdb_vs_B1         the same skill measured against NSRDB
      nsrdb_mimicry             corr(model - station, NSRDB - station)

    The last three answer "is this model downscaling, or reproducing NSRDB?"
    A model that learns NSRDB rather than reality gains skill_nsrdb but not
    skill_stn, and its station errors line up with NSRDB's own (mimicry -> 1).
    NSRDB itself is reported as a model, so its station error is the target-
    quality floor (plan 7.3).
    """
    refs = getattr(fb, "refs", None) or {"NSRDB": fb.ghi}
    root = Path(dcfg["sources"]["stations"]["out_dir"])
    if not (root / "stations.csv").exists():
        warnings.warn("[eval] no station table; run python -m data.stations build")
        return None, None
    meta = pd.read_csv(root / "stations.csv")
    daily = pd.read_csv(root / "station_daily.csv", parse_dates=["date"])
    pos = {int(t): i for i, t in enumerate(test_days)}
    tindex = pd.DatetimeIndex(fb.times)
    pairs = []                                   # one row per station-day
    for _, s in meta[meta.in_grid].iterrows():
        if not fb.valid[s.row, s.col]:
            continue
        d = daily[daily.station_id == s.station_id]
        ti = tindex.get_indexer(d.date)
        keep = np.array([t in pos for t in ti])
        if not keep.any():
            continue
        ti, obs = ti[keep], d.ghi_kwh.values[keep]
        row = {"station_id": np.repeat(s.station_id, ti.size),
               "network": np.repeat(s.get("network", ""), ti.size),
               "region": np.repeat(s.region, ti.size),
               "date": tindex[ti], "obs": obs}
        for lab, arr in refs.items():
            row[lab] = arr[ti, s.row, s.col]
        for name, P in preds_test.items():
            row[name] = P[[pos[t] for t in ti], s.row, s.col]
        pairs.append(pd.DataFrame(row))
    if not pairs:
        return None, None
    df = pd.concat(pairs, ignore_index=True)
    models = [*refs, *preds_test]

    def centred(err: np.ndarray, sid: np.ndarray) -> np.ndarray:
        """Errors with each station's mean removed: day-to-day error only."""
        out = err.copy()
        for k in np.unique(sid):
            m = sid == k
            out[m] -= np.nanmean(err[m])
        return out

    def summarise(g: pd.DataFrame, boot: int = 0) -> list[dict]:
        o = g.obs.values
        sid = g.station_id.values
        ok = np.isfinite(o)
        b1 = g["B1"].values if "B1" in g else None
        if b1 is not None:
            ok &= np.isfinite(b1)
        # Week blocks shared by every station: days are autocorrelated and
        # stations see the same weather, so resampling single station-days
        # would understate the uncertainty.
        week = (pd.DatetimeIndex(g.date.values).asi8 // (7 * 86400 * 10 ** 9)) if boot else None
        out = []
        for m in models:
            p = g[m].values
            k = ok & np.isfinite(p)
            if k.sum() < 10:
                continue
            c = centred(p[k] - o[k], sid[k])
            r = {"model": m, "n": int(k.sum()),
                 "rmse_stn": M.rmse(p[k], o[k]), "mbe_stn": M.mbe(p[k], o[k]),
                 "crmse_stn": float(np.sqrt(np.mean(c ** 2)))}
            for lab in refs:
                ref = g[lab].values
                kk = k & np.isfinite(ref)
                lo = lab.lower()
                r[f"rmse_{lo}"] = M.rmse(p[kk], ref[kk])
                r[f"{lo}_mimicry"] = M.pearson(p[kk] - o[kk], ref[kk] - o[kk]) if m != lab else 1.0
            if b1 is not None:
                cb = centred(b1[k] - o[k], sid[k])
                r["skill_stn_vs_B1"] = M.skill(p[k], o[k], b1[k])
                for lab in refs:
                    ref = g[lab].values
                    kk = k & np.isfinite(ref)
                    r[f"skill_{lab.lower()}_vs_B1"] = M.skill(p[kk], ref[kk], b1[kk]) \
                        if m != lab else np.nan
                r["skill_crmse_vs_B1"] = float(1 - np.mean(c ** 2) / np.mean(cb ** 2))
                if boot:
                    wk = week[k]
                    blocks = np.unique(wk)
                    idx = {b: np.nonzero(wk == b)[0] for b in blocks}
                    rng = np.random.default_rng(0)
                    vals = []
                    for _ in range(boot):
                        pick = np.concatenate([idx[b] for b in rng.choice(blocks, blocks.size)])
                        vals.append(1 - np.mean(c[pick] ** 2) / np.mean(cb[pick] ** 2))
                    r["skill_crmse_lo90"], r["skill_crmse_hi90"] = np.percentile(vals, [5, 95])
            out.append(r)
        return out

    rows = []
    for sid_, g in df.groupby("station_id"):
        rows += [{"station_id": sid_, "region": g.region.iloc[0], **r} for r in summarise(g)]
    per_station = pd.DataFrame(rows)
    summ = []
    for region, g in [("all", df), *df.groupby("region")]:
        summ += [{"region": region, "stations": g.station_id.nunique(), **r}
                 for r in summarise(g, boot=500)]
    per_station.attrs["pairs"] = df
    return per_station, pd.DataFrame(summ)


def plot_maps(maps: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(maps)
    fig, ax = plt.subplots(2, len(names), figsize=(3 * len(names), 6), squeeze=False)
    vmax = np.nanpercentile(np.concatenate([m[0].ravel() for m in maps.values()]), 98)
    for j, n in enumerate(names):
        a = ax[0, j].imshow(maps[n][0], vmin=0, vmax=vmax, cmap="viridis")
        ax[0, j].set_title(f"{n} RMSE")
        b = ax[1, j].imshow(maps[n][1], vmin=0.5, vmax=1, cmap="magma")
        ax[1, j].set_title(f"{n} r")
        for k in (0, 1):
            ax[k, j].set_xticks([]); ax[k, j].set_yticks([])
    fig.colorbar(a, ax=ax[0].tolist(), shrink=0.8, label="kWh/m2/day")
    fig.colorbar(b, ax=ax[1].tolist(), shrink=0.8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def add_amplitude_corrected(fb, preds, P, Ps, test_days, sp_days, out) -> None:
    """Add '<model>+amp' stacks: per-band amplitude gains fitted on DEV days x
    TRAINING pixels (evaluation/structure.py), applied to each stack using only
    the stack's own time mean, then clipped to the physical kt range."""
    import json

    dev_days = np.nonzero(fb.split == "val")[0]
    gains = {}
    kt_max = float(fb.cfg["kt"]["max"])
    for n in [m for m in preds if not m.startswith("B")]:
        dv = np.stack([preds[n](int(t)) for t in dev_days])
        g = S.fit_gains(dv, fb.ghi[dev_days], fb.train_pix, fb.valid, fb.res_km)
        gains[n] = g
        for stack, days in ((P, test_days), (Ps, sp_days)):
            if n in stack:
                toa = np.stack([fb.toa(fb.doy[t]) for t in days])
                corr = S.apply_gains(stack[n], g, fb.valid, fb.res_km)
                stack[f"{n}+amp"] = np.clip(corr, 0, kt_max * toa).astype("float32")
        print(f"[eval] {n}+amp gains {dict(zip(S.band_labels(), np.round(g, 3)))}")
    (out / "amp_gains.json").write_text(json.dumps(gains, indent=2))


def main(models: list[str] | None, amp: bool = False, out_name: str = "eval") -> None:
    dcfg, tcfg = load_all()
    ecfg = tcfg["eval"]
    fb = builder(dcfg, tcfg)
    out = run_dir(dcfg, tcfg, out_name)
    preds = load_predictors(fb, dcfg, tcfg, models)
    print(f"[eval] models: {list(preds)}")
    cells = evaluation_cells(fb.split, fb.holdout, fb.exclusion, fb.valid,
                             ecfg.get("spatial_split", "train"),
                             int(ecfg.get("spatial_day_stride", 1)))

    test_days = np.nonzero(cells["temporal"][0])[0]
    sp_days = np.nonzero(cells["spatial"][0])[0]
    P = {n: np.stack([f(int(t)) for t in test_days]).astype("float32")
         for n, f in preds.items()}
    Ps = {n: np.stack([f(int(t)) for t in sp_days]).astype("float32")
          for n, f in preds.items()} if sp_days.size else {}
    if amp:
        add_amplitude_corrected(fb, preds, P, Ps, test_days, sp_days, out)
    obs = fb.ghi[test_days]
    lab = stratum_labels(fb, test_days, ecfg)
    rows = []
    for cell in ("temporal", "both"):
        pix = cells[cell][1]
        rows += score(obs[:, pix], {n: p[:, pix] for n, p in P.items()},
                      {k: v[:, pix] for k, v in lab.items()}, cell)

    pix = cells["spatial"][1]
    if sp_days.size and pix.any():
        ls = stratum_labels(fb, sp_days, ecfg)
        rows += score(fb.ghi[sp_days][:, pix], {n: p[:, pix] for n, p in Ps.items()},
                      {k: v[:, pix] for k, v in ls.items()}, "spatial")

    srows = []
    for cell, days, stack in (("temporal", test_days, P), ("both", test_days, P),
                              ("spatial", sp_days, Ps)):
        if not stack:
            continue
        tgt = fb.ghi[days]
        bt, _ = S.decompose(tgt, fb.valid, fb.res_km)
        for n, pr in stack.items():
            srows.append({"model": n, "cell": cell,
                          **S.structure_metrics(pr, tgt, stack["B1"], cells[cell][1],
                                                fb.valid, fb.res_km, bands_t=bt)})
    st_tab = pd.DataFrame(srows)
    st_tab.to_csv(out / "structure.csv", index=False)
    cols = ["ssim", "resid_corr"] + [c for c in st_tab.columns if c.startswith("amp_")]
    print("structure, cell=both:")
    print(st_tab[st_tab.cell == "both"].set_index("model")[cols].round(3).to_string())
    table = pd.DataFrame(rows)
    table.to_csv(out / "metrics.csv", index=False)
    head = table[table.stratum == "all"].pivot(index="model", columns="cell",
                                               values=["rmse", "mbe", "skill_vs_B3"])
    print(head.round(4).to_string())
    sky = table[(table.stratum == "sky") & (table.cell == "both")]
    if not sky.empty:
        print(sky.pivot(index="model", columns="value", values="rmse").round(4).to_string())

    maps = {n: pixel_maps(obs, p) for n, p in P.items()}
    np.savez(out / "maps.npz", **{f"{n}_{k}": m[i] for n, m in maps.items()
                                  for i, k in enumerate(("rmse", "corr"))})
    plot_maps(maps, out / "maps.png")

    dx = fb.res_km
    spec = {}
    for n, p in P.items():
        r = spectra(obs, p, fb.valid, dx)
        spec[n] = {"bands": r, "effective_resolution_km": effective_resolution(r),
                   "coherence_resolution_km": coherence_resolution(r)}
    pd.Series({n: s["effective_resolution_km"] for n, s in spec.items()}).to_csv(
        out / "effective_resolution.csv")
    import json

    (out / "spectra.json").write_text(json.dumps(spec, indent=2, default=float))

    st, summ = station_eval(fb, dcfg, P, test_days)
    if st is not None and not st.empty:
        st.to_csv(out / "stations.csv", index=False)
        st.attrs["pairs"].to_csv(out / "station_pairs.csv", index=False)
        summ.to_csv(out / "stations_summary.csv", index=False)
        print(summ.round(3).to_string(index=False))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    # Off by default: on the real store (2026-10-22) every gain hit its clip
    # (4.0) because band coherence below 60 km is ~0.0-0.08, so the correction
    # amplified misplaced texture: SSIM 0.41 -> 0.29, RMSE worse than B0.
    ap.add_argument("--amp", action="store_true",
                    help="add post-hoc band-amplitude corrected '+amp' variants")
    ap.add_argument("--out", default="eval",
                    help="output dir under the run group; use another name with --models "
                         "so a partial evaluation never overwrites the full one")
    a = ap.parse_args()
    main(a.models, a.amp, a.out)

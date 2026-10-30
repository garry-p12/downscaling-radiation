"""Uncertainty of the deterministic downscalers: Monte Carlo drop-path ensembles,
station-calibrated intervals, an uncertainty index and worst-case scenarios.

    python -m evaluation.uncertainty sample --members 20      # Vista, per run group
    python -m evaluation.uncertainty analyse --res 5 10 1     # Mac, from the samples

``sample`` keeps the trained Swin's stochastic-depth (drop-path) layers active
at inference, so every forward pass drops a different set of residual blocks;
M passes of the same day give a Monte Carlo ensemble of the model's own
(epistemic) uncertainty with no retraining. It writes, for the test days,
runs/<group>/eval/mc_<run>.npz: the members and the deterministic (drop-path
off) prediction at every station pixel, and full-field members for a few case
days.

``analyse`` builds 90% intervals around the recommended model (Swin, anchored
v2) from four spread estimates and scores them at the stations, each station
by the models that never saw it (as in evaluation.resolution):

    static   a fixed width per calendar month (the cheap baseline)
    mc       the Monte Carlo drop-path spread above
    mm3      the spread of the three Swin models (NSRDB, v1, v2 targets): how
             much the answer depends on which "truth" the model was trained to
    mm6      the same with the three XGBoost models added

Each spread s is turned into an interval centre +/- q*s by split conformal
calibration across years: q is the 90th percentile of |error|/s in one test
year and the interval is scored in the other, then the years swap. The
uncertainty index (green/amber/red) is the tercile of the interval width; it
is useful only if the observed error grows from green to red while coverage
stays near 90% in every class. Also written: KL divergence of each model's
daily-GHI distribution from the stations', and the capture of each station's
darkest and brightest 10% of days per calendar month.

Writes runs/figures/uq_*.csv, fig14_uncertainty and fig15_uq_austin.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from evaluation import resolution as R
from evaluation.figures import (
    C_BASE,
    C_INK,
    C_MUTED,
    C_REF,
    C_SWIN,
    C_XGB,
    RUNS,
    UNITS,
    panel,
    plt,
    save,
    style,
)
from evaluation.probabilistic import station_pixels
from evaluation.station_anchor import FOLDS2
from models.encoder_swin import DropPath
from training.common import builder, load_all, run_dir, runs_root
from training.train_swin import SwinPredictor

# Days kept as full fields: the Austin map day of Figure a4 and the two case days.
CASE_DAYS = ("2020-03-20", "2021-03-14", "2021-04-23")


def mc_mode(model: torch.nn.Module) -> None:
    """Inference mode with only the drop-path layers stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, DropPath):
            m.train()


@torch.no_grad()
def members(pred: SwinPredictor, t: int, m: int, chunk: int = 10) -> np.ndarray:
    """(m, H, W) GHI members for day ``t``; one input, ``m`` drop-path draws."""
    fb = pred.fb
    f = fb.day_fields(t, pred.names)
    x = np.stack([pred.nz.transform(n, f[n]) for n in pred.names])
    x = torch.from_numpy(np.nan_to_num(x).astype("float32"))[None].to(pred.dev)
    r = np.concatenate([pred.model(x.expand(min(chunk, m - i), -1, -1, -1))[:, pred.point].cpu().numpy()
                        for i in range(0, m, chunk)])
    kb, norm = fb.kt_bilinear(t), fb.norm(t)
    return np.stack([fb.clip(kb + ri, t) * norm for ri in r])


def sample(run: str, m: int, max_days: int | None = None) -> None:
    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    pred = SwinPredictor.from_checkpoint(runs_root(dcfg, tcfg) / run / "best.pt", fb)
    days = np.nonzero(fb.split == "test")[0][:max_days]
    meta, _ = station_pixels(fb, dcfg)
    rr, cc = meta.row.values, meta.col.values
    case = [int(i) for i in fb.times.get_indexer(pd.DatetimeIndex(CASE_DAYS)) if i >= 0]
    ens = np.zeros((len(days), m, len(meta)), np.float32)
    det = np.zeros((len(days), len(meta)), np.float32)
    case_ens, case_det = [], []
    for i, t in enumerate(days):
        mc_mode(pred.model)
        e = members(pred, int(t), m)
        pred.model.eval()
        d = pred.ghi(int(t))
        ens[i], det[i] = e[:, rr, cc], d[rr, cc]
        if t in case:
            case_ens.append(e.astype(np.float16))
            case_det.append(d.astype(np.float32))
        if i % 100 == 0:
            print(f"[mc] {run}: day {i}/{len(days)}, mean spread "
                  f"{float(np.nanmean(e.std(0))):.3f} kWh/m2/day", flush=True)
    f = fb.day_fields(int(days[0]), ["lat", "lon"])
    out = run_dir(dcfg, tcfg, "eval") / f"mc_{run}.npz"
    np.savez_compressed(out, dates=fb.times[days].values.astype("datetime64[D]"),
                        station_id=np.array(meta.station_id.astype(str).tolist()), ens=ens, det=det,
                        case_dates=fb.times[[t for t in case if t in set(days)]].values.astype("datetime64[D]"),
                        case_ens=np.array(case_ens, np.float16).reshape(-1, m, fb.ny, fb.nx),
                        case_det=np.array(case_det, np.float32).reshape(-1, fb.ny, fb.nx),
                        lat=f["lat"].astype(np.float32), lon=f["lon"].astype(np.float32),
                        valid=fb.valid, members=m)
    print(f"[mc] wrote {out}")


# ----------------------------------------------------------------------------- analysis
CENTRE = "Swin v2"
SWIN3 = ["Swin nsrdb", "Swin v1", "Swin v2"]
ALL6 = SWIN3 + ["XGB nsrdb", "XGB v1", "XGB v2"]
LEVEL = 0.9
Z = 1.6449                                       # half-width of a 90% normal interval in sd
METHODS = {"static": "Fixed width per month", "mc": "MC drop-path spread",
           "mm3": "Spread of the 3 Swin targets", "mm6": "Spread of 6 models"}
COL = {"static": C_BASE, "mc": C_INK, "mm3": C_SWIN, "mm6": C_XGB}
INDEX = {"green": "#3a9e5f", "amber": "#e3a21a", "red": "#c8463b"}


def mc_spread(res: int) -> pd.DataFrame | None:
    """Drop-path sd at each station from the v2 fold that never saw it."""
    parts = []
    for f in ("a", "b"):
        path = RUNS / f"{R.GROUPS[res]}_corr2_{f}" / "eval" / "mc_swin_B_cod.npz"
        if not path.exists():
            return None
        z = np.load(path)
        keep = np.isin(z["station_id"], FOLDS2[f])
        sd = z["ens"].std(1)[:, keep]
        parts.append(pd.DataFrame({"station_id": np.tile(z["station_id"][keep], len(z["dates"])),
                                   "date": np.repeat(pd.DatetimeIndex(z["dates"]), keep.sum()),
                                   "s_mc": sd.ravel()}))
    return pd.concat(parts)


def frame(res: int) -> pd.DataFrame:
    df = R.frame(res).dropna(subset=["obs", "B1", "NSRDB", *ALL6]).reset_index(drop=True)
    df["s_static"] = 1.0
    df["s_mm3"] = df[SWIN3].std(axis=1, ddof=0)
    df["s_mm6"] = df[ALL6].std(axis=1, ddof=0)
    mc = mc_spread(res)
    if mc is not None:
        df = df.merge(mc, how="left")
    return df


def calibrate(df: pd.DataFrame, method: str) -> tuple[np.ndarray, dict]:
    """Half-width of the 90% interval per station-day, conformal across years."""
    e = (df.obs - df[CENTRE]).abs().values
    s = df[f"s_{method}"].clip(lower=1e-3).values
    year, month = df.date.dt.year.values, df.date.dt.month.values
    w, qs = np.full(len(df), np.nan), {}
    for y in np.unique(year):
        cal, ev = (year != y) & np.isfinite(s), (year == y) & np.isfinite(s)
        if method == "static":
            q = pd.Series(e[cal]).groupby(month[cal]).quantile(LEVEL)
            w[ev] = q.reindex(month[ev]).values
        else:
            q = float(np.quantile(e[cal] / s[cal], LEVEL))
            w[ev] = q * s[ev]
        qs[int(y)] = q
    return w, qs


def _crps_gauss(mu, sd, obs):
    from scipy.stats import norm

    z = (obs - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def score(df: pd.DataFrame, w: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    ok = np.isfinite(w)
    e = (df.obs - df[CENTRE]).values[ok]
    w = w[ok]
    miss = np.maximum(np.abs(e) - w, 0)
    return {"n": int(ok.sum()), "coverage": float(np.mean(np.abs(e) <= w)), "width": float(np.mean(2 * w)),
            "interval_score": float(np.mean(2 * w + 2 / (1 - LEVEL) * miss)),
            "crps": float(np.mean(_crps_gauss(0, w / Z, e))),
            "rank_corr": float(spearmanr(w, np.abs(e))[0]) if np.ptp(w) > 0 else np.nan}


def index_classes(w: np.ndarray) -> np.ndarray:
    """Uncertainty index: tercile of the interval width (thresholds from widths only)."""
    lo, hi = np.nanquantile(w, [1 / 3, 2 / 3])
    return np.where(w <= lo, "green", np.where(w <= hi, "amber", "red"))


KL_BINS = np.arange(0, 9.25, 0.25)                # kWh/m2/day


def kl(obs: np.ndarray, pred: np.ndarray, bins: np.ndarray = KL_BINS) -> float:
    """KL(stations || model) of daily GHI, add-half smoothing."""
    p = np.histogram(np.clip(obs, 0, 9), bins)[0] + 0.5
    q = np.histogram(np.clip(pred, 0, 9), bins)[0] + 0.5
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def extremes(df: pd.DataFrame, model: str) -> dict:
    """Capture of each station's darkest / brightest 10% of days per calendar month."""
    g = df.groupby([df.station_id, df.date.dt.month]).obs
    lo, hi = g.transform(lambda v: v.quantile(0.1)), g.transform(lambda v: v.quantile(0.9))
    out = {}
    for lab, ev, hit in (("dark", df.obs <= lo, df[model] <= lo), ("bright", df.obs >= hi, df[model] >= hi)):
        out[f"{lab}_pod"] = float((ev & hit).sum() / ev.sum())
        out[f"{lab}_far"] = float((~ev & hit).sum() / max(hit.sum(), 1))
        out[f"{lab}_bias"] = float((df[model] - df.obs)[ev].mean())
    return out


def analyse(res_list: list[int]) -> None:
    style()
    out = RUNS / "figures"
    frames = {r: frame(r) for r in res_list}
    main_res = 5 if 5 in frames else res_list[0]
    rows, idx, cal_q = [], [], {}
    df = frames[main_res]
    methods = [m for m in METHODS if f"s_{m}" in df and df[f"s_{m}"].notna().any()]
    for m in methods:
        w, qs = calibrate(df, m)
        cal_q[m] = qs
        df[f"w_{m}"] = w
        raw = df[f"s_{m}"].values
        e = (df.obs - df[CENTRE]).values
        ss = np.sqrt(np.nanmean(raw ** 2)) / np.sqrt(np.mean(e ** 2)) if m != "static" else np.nan
        for net in ["all", *sorted(df.network.unique())]:
            sel = (np.ones(len(df), bool) if net == "all" else (df.network == net).values)
            rows.append({"res_km": main_res, "method": m, "network": net, "raw_spread_skill": ss,
                         **score(df[sel], w[sel])})
        cls = index_classes(w)
        df[f"idx_{m}"] = cls
        for c in INDEX:
            k = cls == c
            ek = np.abs(e[k])
            idx.append({"method": m, "class": c, "share": float(k.mean()), "width": float(np.nanmean(2 * w[k])),
                        "mae": float(ek.mean()), "coverage": float(np.mean(ek <= w[k]))})
    t = pd.DataFrame(rows)
    ix = pd.DataFrame(idx)

    # Grid resolution: the same methods on the station-days every grid shares.
    key = ["station_id", "date"]
    common = frames[res_list[0]][key]
    for r in res_list[1:]:
        common = common.merge(frames[r][key])
    rres, ex_rows = [], []
    for r, f in frames.items():
        fc = f.merge(common).reset_index(drop=True)
        for m in ("static", "mm3", "mm6"):
            w, _ = calibrate(fc, m)
            rres.append({"res_km": r, "method": m, "stations": fc.station_id.nunique(), **score(fc, w)})
        for model in ["B1", "NSRDB", *ALL6]:
            rec = {"res_km": r, "model": model}
            for net in ["all", *sorted(f.network.unique())]:
                sel = f if net == "all" else f[f.network == net]
                rec[f"kl_{net}"] = kl(sel.obs.values, sel[model].values)
            rec |= extremes(f, model)
            ex_rows.append(rec)
    tr = pd.DataFrame(rres)
    ex = pd.DataFrame(ex_rows)
    t.to_csv(out / "uq_intervals.csv", index=False)
    ix.to_csv(out / "uq_index.csv", index=False)
    tr.to_csv(out / "uq_resolution.csv", index=False)
    ex.to_csv(out / "uq_kl_extremes.csv", index=False)
    pd.Series({m: str(q) for m, q in cal_q.items()}).to_csv(out / "uq_calibration.csv")
    print(t.round(3).to_string(index=False))
    print(ix.round(3).to_string(index=False))
    print(tr.round(3).to_string(index=False))
    print(ex.round(3).to_string(index=False))
    fig_stations(t, ix, tr, main_res, methods, out)
    if main_res == 5 and (out / "austin_cache_5km.npz").exists():
        fig_austin(df, cal_q, out)


def fig_stations(t, ix, tr, res, methods, out) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.45, width_ratios=[1.2, 1, 1]))
    a = t[t.network == "all"].set_index("method").loc[methods]
    ax = axes[0]
    y = np.arange(len(methods))[::-1]
    ax.barh(y, a.interval_score, color=[COL[m] for m in methods], height=0.6)
    for yi, m in zip(y, methods):
        ax.text(a.interval_score[m] + 0.02, yi, f"width {a.width[m]:.2f}\n{100 * a.coverage[m]:.0f}% covered",
                va="center", fontsize=5.6, color=C_INK, linespacing=1.1)
    ax.set_yticks(y, [METHODS[m] for m in methods], fontsize=6.5); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Interval score ({UNITS}), lower is better")
    ax.set_xlim(0, a.interval_score.max() * 1.45)
    ax.set_title(f"90% intervals, {res} km", loc="left")
    panel(ax, "a", x=-0.02)
    b = axes[1]
    cls = list(INDEX)
    for k, m in enumerate([m for m in methods if m != "static"]):
        g = ix[ix.method == m].set_index("class").loc[cls]
        off = (k - (len(methods) - 2) / 2) * 0.25
        b.bar(np.arange(3) + off, g.mae, width=0.24, color=[INDEX[c] for c in cls], edgecolor=COL[m], lw=1.2)
    b.set_xticks(range(3), ["Green", "Amber", "Red"])
    b.set_ylabel(f"Mean |error| ({UNITS})")
    b.set_title("Error by uncertainty index", loc="left")
    b.set_ylim(0, ix.mae.max() * 1.35)
    from matplotlib.patches import Patch

    b.legend(handles=[Patch(facecolor="white", edgecolor=COL[m], lw=1.2, label=METHODS[m])
                      for m in methods if m != "static"],
             fontsize=5.5, loc="upper left", handlelength=1, frameon=False)
    panel(b, "b", x=-0.02)
    c = axes[2]
    for m in ("static", "mm3", "mm6"):
        g = tr[tr.method == m].sort_values("res_km", ascending=False)
        c.plot(g.res_km.astype(str), g.width, "o-", color=COL[m], lw=1.2, ms=4, label=METHODS[m])
    c.set_xlabel("Grid spacing (km)")
    c.set_ylabel(f"Mean width ({UNITS})")
    c.set_title(f"Width by grid, {int(tr.stations.iloc[0])} stations", loc="left")
    c.set_ylim(tr.width.min() - 0.01, tr.width.max() + 0.06)
    c.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=1)
    panel(c, "c", x=-0.02)
    fig.text(0.0, -0.1, "Swin on the anchored v2 target, test years 2020–21, each station scored by the fold that "
             "never saw it. Intervals are calibrated in one year and scored in the other (split conformal), so "
             "every method covers about 90%. Interval score = width + 20 x the amount\nby which the observation falls outside. Index: tercile of interval width. Edge colour "
             "in (b) marks the spread estimate.", fontsize=6.2, color=C_MUTED)
    save(fig, out, "fig14_uncertainty")


def fig_austin(df, cal_q, out) -> None:
    from evaluation import austin as A

    A.use_res(5)
    z, w_, _ = A._load(out / "austin_cache_5km.npz")
    day = pd.Timestamp("2021-03-14")
    k = int(np.nonzero(pd.DatetimeIndex(z["days"]) == day)[0][0])
    mem = np.stack([z[c][k] for c in ("swin_nsrdb", "swin_v1", "swin_v2")])
    centre, s = z["swin_v2"][k], mem.std(0)
    half = cal_q["mm3"][2020] * np.maximum(s, 1e-3)          # calibrated on the other year
    lo_t, hi_t = np.nanquantile(df.loc[df.date.dt.year == 2021, "w_mm3"], [1 / 3, 2 / 3])
    cls = np.where(half <= lo_t, 0, np.where(half <= hi_t, 1, 2)).astype(float)
    low = centre - half
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.08))
    vmin, vmax = np.nanpercentile(np.concatenate([centre[w_.valid], low[w_.valid]]), [1, 99])
    im0 = w_.show(axes[0], centre, cmap="cividis", vmin=vmin, vmax=vmax)
    im1 = w_.show(axes[1], half, cities=False, cmap="magma_r", vmin=0)
    from matplotlib.colors import ListedColormap

    im2 = w_.show(axes[2], cls, cities=False, cmap=ListedColormap(list(INDEX.values())), vmin=-0.5, vmax=2.5)
    im3 = w_.show(axes[3], low, cities=False, cmap="cividis", vmin=vmin, vmax=vmax)
    titles = ["Central estimate (Swin v2)", "± 90% half-width", "Uncertainty index", "Low-end scenario (5%)"]
    for ax, ttl, letter in zip(axes, titles, "abcd"):
        ax.set_title(ttl, fontsize=6.6, loc="left")
        panel(ax, letter, x=-0.02)
    # East Austin: the cell nearest 30.26 N, 97.70 W
    x, y = w_.f.project(-97.70, 30.26)
    r = int(np.argmin(np.abs(w_.f.y - y))) - int(z["rc"][0])
    c_ = int(np.argmin(np.abs(w_.f.x - x))) - int(z["rc"][2])
    for ax in axes[1:]:
        ax.scatter(x, y, marker="s", s=10, facecolor="none", edgecolor=C_REF, lw=0.7, zorder=7)
    for ax, im in ((axes[0], im0), (axes[1], im1), (axes[2], im2), (axes[3], im3)):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
        cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.3)
        if im is im2:
            cb.set_ticks([0, 1, 2], labels=["Green", "Amber", "Red"])
    fig.text(0.0, -0.13, f"14 March 2021, 5 km. East Austin (square): central {centre[r, c_]:.2f}, 90% interval "
             f"{low[r, c_]:.2f}–{centre[r, c_] + half[r, c_]:.2f} {UNITS}, index "
             f"{list(INDEX)[int(cls[r, c_])]}. Width from the spread of the three Swin models, scaled by the "
             "factor calibrated at the stations in 2020; index thresholds are the terciles of all 2021 "
             "station-day widths, so a cloudy spring day is mostly amber or red. Low-end scenario = central − half-width, the level "
             "daily GHI falls below on about 1 day in 20.", fontsize=6.2, color=C_MUTED, wrap=True)
    save(fig, out, "fig15_uq_austin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--run", default="swin_B_cod")
    s.add_argument("--members", type=int, default=20)
    s.add_argument("--max-days", type=int, default=None, help="smoke tests only")
    an = sub.add_parser("analyse")
    an.add_argument("--res", type=int, nargs="+", default=[5, 10, 1], choices=sorted(R.GROUPS, reverse=True))
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.run, a.members, a.max_days)
    else:
        analyse(a.res)

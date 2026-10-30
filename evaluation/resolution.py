"""Grid resolution: the same models at 10, 5 (and 1) km, scored at the stations.

    python -m evaluation.resolution                 # 10 vs 5 km
    python -m evaluation.resolution --res 5 10 1    # adds the 1 km subdomain; gains vs 5 km

Every grid is scored on the station-days ALL requested grids have (the 1 km
grid covers a subdomain, so adding it restricts the comparison to its 42
stations). Anchored models are scored at each station by the fold whose
correction never saw it, as in station_anchor.score_networks. Skill is centred
(station mean error removed) relative to B1 on the SAME grid; the paired gain
resamples the same weeks for both grids, so it is the resolution effect alone.
Gains are against the first grid listed.

Writes runs/figures/resolution.csv, resolution_gain.csv and fig13_resolution,
each with the grids as a suffix when they are not the default 10 vs 5 km
(e.g. resolution_10_5_1.csv, fig13_resolution_10_5_1).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from evaluation.figures import (
    C_AQUA,
    C_BASE,
    C_INK,
    C_MUTED,
    C_SWIN,
    C_XGB,
    RUNS,
    panel,
    plt,
    save,
    style,
)
from evaluation.station_anchor import FOLDS, FOLDS2, _centred_mse

# grid -> run-group prefix; the 10 km anchored runs were scored in eval_meso
GROUPS = {10: "data_store_goes", 5: "data_store_5km", 1: "data_store_1km"}
EVALS = ("eval_meso", "eval_swin", "eval")
MODELS = ["NSRDB", "XGB nsrdb", "Swin nsrdb", "XGB v1", "Swin v1", "XGB v2", "Swin v2"]
LAB = {"NSRDB": "NSRDB", "XGB nsrdb": "XGBoost, NSRDB target", "Swin nsrdb": "Swin, NSRDB target",
       "XGB v1": "XGBoost, anchored v1", "Swin v1": "Swin, anchored v1",
       "XGB v2": "XGBoost, anchored v2", "Swin v2": "Swin, anchored v2"}
COL = {10: C_BASE, 5: C_SWIN, 1: C_AQUA}


def _pairs(group: str, col: str) -> pd.DataFrame:
    """The 46-station station_pairs.csv of a run group that has ``col``."""
    for ev in EVALS:
        f = RUNS / group / ev / "station_pairs.csv"
        if f.exists() and col in pd.read_csv(f, nrows=0).columns:
            p = pd.read_csv(f, parse_dates=["date"])
            if p.station_id.nunique() > 7:          # not the early USCRN-only evals
                return p
    raise FileNotFoundError(f"no station_pairs.csv with {col} under runs/{group}")


def _folded(base: pd.DataFrame, grp: str, col: str, folds: dict) -> np.ndarray:
    a = _pairs(grp.format("a"), col)[["station_id", "date", col]].rename(columns={col: "_a"})
    b = _pairs(grp.format("b"), col)[["station_id", "date", col]].rename(columns={col: "_b"})
    m = base[["station_id", "date"]].merge(a, how="left").merge(b, how="left")
    v = m[["_a", "_b"]].mean(axis=1).values
    v = np.where(m.station_id.isin(folds["a"]), m["_a"], v)
    return np.where(m.station_id.isin(folds["b"]), m["_b"], v)


def frame(res: int) -> pd.DataFrame:
    g = GROUPS[res]
    base = _pairs(g, "swin_B_cod")
    df = base[["station_id", "network", "date", "obs", "NSRDB", "B1"]].copy()
    df["XGB nsrdb"], df["Swin nsrdb"] = base["xgb_B_cod"], base["swin_B_cod"]
    for lab, suf, folds in (("v1", "corr_{}", FOLDS), ("v2", "corr2_{}", FOLDS2)):
        for m, col in (("XGB", "xgb_B_cod"), ("Swin", "swin_B_cod")):
            df[f"{m} {lab}"] = _folded(base, f"{g}_{suf}", col, folds)
    return df


def _skill(g: pd.DataFrame, m: str) -> float:
    return 1 - _centred_mse(g, m) / _centred_mse(g, "B1")


def compare(res: list[int], boot: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["station_id", "date"]
    frames = {r: frame(r).dropna(subset=["obs", "B1", *MODELS]) for r in res}
    common = frames[res[0]][key]
    for r in res[1:]:
        common = common.merge(frames[r][key])
    frames = {r: f.merge(common).sort_values(key).reset_index(drop=True) for r, f in frames.items()}
    ref = frames[res[0]]
    rng = np.random.default_rng(0)
    rows, gains = [], []
    for net in ["all", *sorted(ref.network.unique())]:
        sel = np.ones(len(ref), bool) if net == "all" else (ref.network == net).values
        week = ref.date.values[sel].astype("datetime64[W]")
        wk = np.unique(week)
        idx = {w: np.nonzero(week == w)[0] for w in wk}
        draws = [np.concatenate([idx[w] for w in rng.choice(wk, wk.size)]) for _ in range(boot)]
        sub = {r: f[sel].reset_index(drop=True) for r, f in frames.items()}
        for m in [*MODELS, "B1"]:
            sk = {r: np.array([_skill(sub[r].iloc[d], m) for d in draws]) for r in res}
            for r in res:
                e = sub[r][m] - sub[r].obs
                rows.append(dict(network=net, stations=sub[r].station_id.nunique(), n=len(e), res_km=r,
                                 model=m, rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()),
                                 skill=_skill(sub[r], m), lo90=np.percentile(sk[r], 5),
                                 hi90=np.percentile(sk[r], 95)))
            for r in res[1:]:
                d = sk[r] - sk[res[0]]
                gains.append(dict(network=net, model=m, res_km=r, vs_km=res[0],
                                  gain=_skill(sub[r], m) - _skill(sub[res[0]], m),
                                  lo90=np.percentile(d, 5), hi90=np.percentile(d, 95)))
    return pd.DataFrame(rows), pd.DataFrame(gains)


def fig_resolution(t: pd.DataFrame, gains: pd.DataFrame, out, name: str = "fig13_resolution") -> None:
    res = sorted(t.res_km.unique(), reverse=True)
    a = t[(t.network == "all") & t.model.isin(MODELS)].set_index(["model", "res_km"])
    n_st, n = int(a.stations.iloc[0]), int(a.n.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08))
    yy = np.arange(len(MODELS))[::-1]
    ax = axes[0]
    for y, m in zip(yy, MODELS):
        for k, r in enumerate(res):
            off = (k - (len(res) - 1) / 2) * 0.22
            ax.plot(100 * a.loc[(m, r), ["lo90", "hi90"]].values, [y + off] * 2, color=COL[r], lw=1.1,
                    alpha=0.6)
            ax.scatter(100 * a.loc[(m, r), "skill"], y + off, s=26, color=COL[r], edgecolor="white",
                       linewidth=0.5, zorder=3, label=f"{r} km" if y == yy[0] else None)
    ax.axvline(0, color=C_INK, lw=0.8)
    ax.set_xlabel("Day-to-day skill vs B1 (%)")
    ax.set_title(f"Skill per grid\n{n_st} stations, {n:,} station-days", loc="left")
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.1)
    ax.set_yticks(yy, [LAB[m] for m in MODELS]); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    panel(ax, "a", x=-0.03)
    b = axes[1]
    g = gains[(gains.network == "all") & gains.model.isin(MODELS)]
    for k, r in enumerate(sorted(g.res_km.unique(), reverse=True)):
        gg = g[g.res_km == r].set_index("model")
        off = (k - (g.res_km.nunique() - 1) / 2) * 0.22
        for y, m in zip(yy, MODELS):
            fam = C_SWIN if m.startswith("Swin") else C_XGB if m.startswith("XGB") else C_MUTED
            b.plot(100 * gg.loc[m, ["lo90", "hi90"]].values, [y + off] * 2, color=fam, lw=1.5,
                   solid_capstyle="round", alpha=0.85)
            b.scatter(100 * gg.loc[m, "gain"], y + off, s=24,
                      color=fam if r < gg.vs_km.iloc[0] else "white", edgecolor=fam, linewidth=0.9,
                      zorder=3)
    b.axvline(0, color=C_INK, lw=0.8)
    if g.res_km.nunique() > 1:
        vs = int(g.vs_km.iloc[0])
        for r in sorted(g.res_km.unique()):
            b.scatter([], [], s=24, color=C_MUTED if r < vs else "white", edgecolor=C_MUTED,
                      linewidth=0.9, label=f"{r} vs {vs} km")
        b.legend(loc="lower right", fontsize=6.5, handletextpad=0.1)
    b.set_xlabel(f"Skill gain vs {int(g.vs_km.iloc[0])} km (points)")
    b.set_title("Paired gain, same weeks resampled", loc="left")
    b.grid(axis="x"); b.set_axisbelow(True)
    panel(b, "b", x=-0.03)
    fig.text(0.0, -0.07, "Test years 2020–21, GOES window, B + cloud optical depth models. Skill = 1 − "
             "MSE/MSE(B1 on the same grid) after removing each station's mean error. Bars: 90% "
             "week-block bootstrap\nintervals; in (b) both grids are resampled with the same weeks. "
             "Anchored models are scored at each station by the fold whose correction never saw it.",
             fontsize=6.3, color=C_MUTED)
    save(fig, out, name)


def main(res: list[int]) -> None:
    style()
    out = RUNS / "figures"
    t, gains = compare(res)
    suf = "" if res == [10, 5] else "_" + "_".join(map(str, res))
    t.to_csv(out / f"resolution{suf}.csv", index=False)
    gains.to_csv(out / f"resolution_gain{suf}.csv", index=False)
    show = t[t.network == "all"].pivot(index="model", columns="res_km", values=["rmse", "bias", "skill"])
    print(show.reindex([*MODELS, "B1"]).round(3).to_string())
    print(gains[gains.network == "all"].round(3).to_string(index=False))
    fig_resolution(t, gains, out, f"fig13_resolution{suf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[10, 5], choices=sorted(GROUPS, reverse=True))
    main(ap.parse_args().res)

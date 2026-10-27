"""Method A: anchoring to ground stations, scored only on stations held out.

The ERA5-Land target swap showed that a model inherits its training target's
errors at the ground, so the ceiling is the target. Two ways to lift it with
ground truth, each scored leave-one-station-out (LOSO) on the test years:

  errmodel   A2. Learn the TARGET's station error from inputs that exist
             everywhere: d = (station - NSRDB) / norm, regressed on the Track B
             features at the station pixel plus NSRDB's own index, on TRAINING
             years at the OTHER stations (XGBoost). If d-hat improves NSRDB at
             a station the model never saw, a corrected target
             NSRDB + d-hat * norm is worth training on (``build-target``).

  krige      A1. Merge each day's station residuals into the field: for a held
             -out station, the product plus the ordinary-kriged residuals of
             the other stations that day (exponential correlation, length
             scale fitted on training years). Needs station data at
             prediction time, like an operational satellite-gauge merge.

  build-target --exclude S1 S2 ...
             Fit the A2 error model on all stations except the listed ones
             (the evaluation fold) and write <target>_corr_<fold> to the
             store, for training with ``target_var`` set to it.

Metrics per held-out station and pooled: RMSE, bias, centred RMSE (each
product's own station mean error removed) and centred skill vs B1, kWh/m2/day.
Writes runs/<group>/eval/anchor_<mode>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.common import builder, load_all, run_dir

ERR_FEATURES_EXTRA = ("nsrdb_k", "nsrdb_minus_base")


def station_table(fb, dcfg):
    """Station-day rows at in-grid valid stations with a finite target."""
    root = Path(dcfg["sources"]["stations"]["out_dir"])
    meta = pd.read_csv(root / "stations.csv")
    meta = meta[meta.in_grid & meta.apply(lambda s: bool(fb.valid[int(s.row), int(s.col)])
                                          if s.in_grid else False, axis=1)]
    daily = pd.read_csv(root / "station_daily.csv", parse_dates=["date"])
    tindex = pd.DatetimeIndex(fb.times)
    rows = []
    for _, s in meta.iterrows():
        d = daily[daily.station_id == s.station_id]
        ti = tindex.get_indexer(d.date)
        ok = ti >= 0
        pix = int(s.row) * fb.nx + int(s.col)
        rows.append(pd.DataFrame({"station_id": s.station_id, "region": s.region,
                                  "network": s.get("network", ""), "lat": s.lat, "lon": s.lon,
                                  "t": ti[ok], "pix": pix, "obs": d.ghi_kwh.values[ok]}))
    df = pd.concat(rows, ignore_index=True)
    df["target"] = fb.ghi.reshape(len(fb.times), -1)[df.t.values, df.pix.values]
    df = df[np.isfinite(df.target) & np.isfinite(df.obs)].reset_index(drop=True)
    norm = np.empty(len(df), "float32")
    b1 = np.empty(len(df), "float32")
    for t, idx in df.groupby("t").indices.items():
        p = df.pix.values[idx]
        norm[idx] = fb.norm(int(t)).ravel()[p]
        b1[idx] = (np.clip(fb.kt_bilinear_toa(int(t)), 0, 1) * fb.toa(fb.doy[int(t)])).ravel()[p]
    df["norm"], df["B1"] = norm, b1
    df["split"] = fb.split[df.t.values]
    return df


def features(fb, df, names):
    X, _, aux = fb.table(df.t.values, df.pix.values, names)
    k = df.target.values / df.norm.values
    extra = np.c_[k, k - aux["kt_bil"]]
    return np.c_[X, extra].astype("float32")


def err_target(df: pd.DataFrame, centred: bool) -> np.ndarray:
    """NSRDB's station error in index units. ``centred`` removes each
    station's mean error over its TRAINING days, so the model learns only the
    day-to-day part: across networks that part transfers, the absolute offset
    (instrument calibration, about 0.3 kWh/m2/day between USCRN and the
    Mesonet) does not."""
    y = ((df.obs - df.target) / df.norm).values
    if centred:
        tr = (df.split == "train").values
        mu = pd.Series(np.where(tr, y, np.nan)).groupby(df.station_id.values).transform("mean")
        y = y - np.nan_to_num(mu.values)
    return y


def fit_err_model(X, y):
    import xgboost as xgb

    return xgb.train({"max_depth": 4, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                      "min_child_weight": 20, "lambda": 5.0, "objective": "reg:squarederror"},
                     xgb.DMatrix(X, y), 400)


def scores(df, cols, ref="B1"):
    """Per station and pooled scores for prediction columns in ``cols``."""
    out = []
    groups = [("all", df)] + list(df.groupby("station_id"))
    for sid, g in groups:
        e_ref = g[ref] - g.obs
        c_ref = e_ref - (e_ref.groupby(g.station_id).transform("mean"))
        for c in cols:
            e = g[c] - g.obs
            ce = e - e.groupby(g.station_id).transform("mean")
            out.append({"station_id": sid, "model": c, "n": len(g),
                        "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
                        "crmse": float(np.sqrt(np.mean(ce ** 2))),
                        "skill_crmse_vs_B1": float(1 - np.mean(ce ** 2) / np.mean(c_ref ** 2)),
                        "skill_rmse_vs_B1": float(1 - np.mean(e ** 2) / np.mean(e_ref ** 2))})
    return pd.DataFrame(out)


def run_errmodel(fb, dcfg, tcfg, out: Path, fit_net: str | None = None,
                 score_net: str | None = None, centred: bool = False) -> pd.DataFrame:
    """LOSO over every station, or, with ``fit_net``/``score_net``, a
    cross-network test: fit the error model on one network's stations and
    score it on the other's (different instruments and maintenance)."""
    df = station_table(fb, dcfg)
    names = fb.track("B")
    X = features(fb, df, names)
    y = err_target(df, centred)
    test = (df.split == "test").values
    train = (df.split == "train").values
    df["NSRDB+err"] = np.nan
    import xgboost as xgb

    if fit_net:
        fit = (df.network == fit_net).values
        bst = fit_err_model(X[train & fit], y[train & fit])
        sel = test & (df.network == score_net).values
        df.loc[sel, "NSRDB+err"] = (df.target.values[sel]
                                    + bst.predict(xgb.DMatrix(X[sel])) * df.norm.values[sel])
        print(f"[anchor] fit on {fit_net} ({int((train & fit).sum())} station-days), "
              f"scored on {score_net} ({int(sel.sum())} test station-days)", flush=True)
        name = f"anchor_errmodel_{fit_net}_to_{score_net}{'_c' if centred else ''}.csv"
        d = df[sel]
    else:
        for sid in df.station_id.unique():
            held = (df.station_id == sid).values
            bst = fit_err_model(X[train & ~held], y[train & ~held])
            sel = held & test
            if sel.any():
                dh = bst.predict(xgb.DMatrix(X[sel]))
                df.loc[sel, "NSRDB+err"] = df.target.values[sel] + dh * df.norm.values[sel]
            print(f"[anchor] errmodel LOSO {sid}: {int(sel.sum())} test days", flush=True)
        name = f"anchor_errmodel{'_c' if centred else ''}.csv"
        d = df[test]
    d = d.rename(columns={"target": "NSRDB"})
    tab = scores(d, ["NSRDB", "NSRDB+err", "B1"])
    if not fit_net:      # pooled per network too
        tab = pd.concat([tab] + [scores(g, ["NSRDB", "NSRDB+err", "B1"]).query("station_id == 'all'")
                                 .assign(station_id=f"all:{n}") for n, g in d.groupby("network")])
    tab.to_csv(out / name, index=False)
    return tab


def _fit_corr_length(res: pd.DataFrame, dist: np.ndarray, ids: list[str]) -> float:
    """Exponential correlation length (km) of daily residuals between stations."""
    piv = res.pivot_table(index="t", columns="station_id", values="r").reindex(columns=ids)
    c = piv.corr(min_periods=200).values
    iu = np.triu_indices(len(ids), 1)
    h, r = dist[iu], c[iu]
    ok = np.isfinite(r) & (r > 0.02)
    if ok.sum() < 2:
        return 50.0
    # log r = -h / L  (least squares through the origin)
    return float(max(10.0, -np.sum(h[ok] ** 2) / np.sum(h[ok] * np.log(r[ok]))))


def run_krige(fb, dcfg, tcfg, out: Path, pairs: Path | None) -> pd.DataFrame:
    df = station_table(fb, dcfg)
    df = df.rename(columns={"target": "NSRDB"})
    cols = ["NSRDB", "B1"]
    if pairs is not None and pairs.exists():
        pp = pd.read_csv(pairs, parse_dates=["date"])
        pp["t"] = pd.DatetimeIndex(fb.times).get_indexer(pp.date)
        extra = [c for c in pp.columns if c.startswith(("swin_", "xgb_"))]
        df = df.merge(pp[["station_id", "t", *extra]], on=["station_id", "t"], how="left")
        cols += extra
    ids = sorted(df.station_id.unique())
    ll = df.groupby("station_id")[["lat", "lon"]].first().loc[ids].values
    R = 6371.0
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    dist = R * np.arccos(np.clip(np.sin(la[:, None]) * np.sin(la[None]) + np.cos(la[:, None])
                                 * np.cos(la[None]) * np.cos(lo[:, None] - lo[None]), -1, 1))
    tr = df[df.split == "train"].assign(r=lambda g: g.obs - g.NSRDB)
    L = _fit_corr_length(tr, dist, ids)
    print(f"[anchor] residual correlation length {L:.0f} km over {len(ids)} stations")
    test = df[df.split == "test"].copy()
    pos = {s: i for i, s in enumerate(ids)}
    for c in cols:
        test[f"{c}+krige"] = np.nan
        for _, g in test.groupby("t"):
            if len(g) < 2:
                continue
            idx = np.array([pos[s] for s in g.station_id])
            r = (g.obs - g[c]).values
            fin = np.isfinite(r)
            for j in range(len(g)):
                others = np.nonzero(fin & (np.arange(len(g)) != j))[0]
                if not len(others) or not np.isfinite(g[c].values[j]):
                    continue
                C = np.exp(-dist[np.ix_(idx[others], idx[others])] / L) + 0.05 * np.eye(len(others))
                c0 = np.exp(-dist[idx[j], idx[others]] / L)
                # ordinary kriging: weights sum to one (unknown local mean)
                A = np.block([[C, np.ones((len(others), 1))], [np.ones((1, len(others))), 0]])
                w = np.linalg.solve(A, np.r_[c0, 1.0])[:-1]
                test.loc[g.index[j], f"{c}+krige"] = g[c].values[j] + w @ r[others]
    tab = scores(test, cols + [f"{c}+krige" for c in cols])
    tab["corr_length_km"] = L
    tab.to_csv(out / "anchor_krige.csv", index=False)
    return tab


def build_target(fb, dcfg, tcfg, exclude: list[str], fold: str, centred: bool = False) -> None:
    """Corrected target from an error model fit WITHOUT the fold's stations."""
    import xarray as xr
    import xgboost as xgb

    from data.build_store import append_to_store

    df = station_table(fb, dcfg)
    names = fb.track("B")
    keep = (~df.station_id.isin(exclude) & (df.split == "train")).values
    X = features(fb, df[keep], names)
    y = err_target(df, centred)[keep]
    bst = fit_err_model(X, y)
    T = len(fb.times)
    corr = np.full_like(fb.ghi, np.nan)
    vp = np.nonzero(fb.valid.ravel())[0]
    for t in range(T):
        dd = pd.DataFrame({"t": t, "pix": vp, "target": fb.ghi[t].ravel()[vp],
                           "norm": fb.norm(t).ravel()[vp]})
        ok = np.isfinite(dd.target.values)
        if not ok.any():
            continue
        dd = dd[ok]
        dh = bst.predict(xgb.DMatrix(features(fb, dd, names)))
        flat = corr[t].ravel()
        flat[dd.pix.values] = np.maximum(dd.target.values + dh * dd.norm.values, 0)
        corr[t] = flat.reshape(fb.ny, fb.nx)
        if t % 1000 == 0:
            print(f"[anchor] corrected target day {t}/{T}", flush=True)
    var = f"{fb.target_var}_corr_{fold}"
    store = dcfg["dataset"]["out_zarr"]
    st = xr.open_zarr(store)
    out = xr.Dataset({var: (("time", "y", "x"), corr.astype("float32"))}).chunk(
        {"time": st.chunks["time"][0]})
    append_to_store(out, store)
    print(f"[anchor] wrote {var} (error model on {int(keep.sum())} station-days, "
          f"excluded {exclude})")


def _centred_mse(g: pd.DataFrame, col: str) -> float:
    e = g[col] - g.obs
    return float(np.mean((e - e.groupby(g.station_id).transform("mean")) ** 2))


FOLDS = {"a": ["TX_Austin_33_NW", "TX_Monahans_6_ENE", "TX_Palestine_6_WNW",
               "TX_Port_Aransas_32_NNE"],
         "b": ["TX_Bronte_11_NNE", "TX_Muleshoe_19_S", "TX_Panther_Junction_2_N"]}
# v2: all 46 stations of both networks, latitude-interleaved (slurm/fold2_*.txt)
_F2 = Path(__file__).resolve().parents[1] / "slurm"
FOLDS2 = {k: (_F2 / f"fold2_{k}.txt").read_text().split() if (_F2 / f"fold2_{k}.txt").exists()
          else [] for k in ("a", "b")}


def score_folds(runs: Path, model: str = "xgb_B_modis", main_group: str = "data_store",
                boot: int = 500) -> pd.DataFrame:
    # the NSRDB-trained reference: the full eval, or a partial one that has the model
    evals = sorted((runs / main_group).glob("eval*/station_pairs.csv"))
    ref = next(p for p in evals
               if model in pd.read_csv(p, nrows=0).columns and "NSRDB" in pd.read_csv(p, nrows=0).columns)
    """Pool every fold's HELD-OUT stations: the model trained on that fold's
    corrected target vs the same model trained on NSRDB, NSRDB and B1, on the
    same station-days. Centred skill vs B1 with a week-block bootstrap."""
    main = pd.read_csv(ref, parse_dates=["date"])
    parts = []
    for f, ids in FOLDS.items():
        p = pd.read_csv(runs / f"{main_group}_corr_{f}" / "eval" / "station_pairs.csv",
                        parse_dates=["date"])
        p = p[p.station_id.isin(ids)][["station_id", "date", "obs", model]]
        parts.append(p.rename(columns={model: "anchored"}).assign(fold=f))
    df = pd.concat(parts).merge(main[["station_id", "date", "NSRDB", "B1", model]],
                                on=["station_id", "date"]).rename(columns={model: "nsrdb_trained"})
    df = df.dropna(subset=["obs", "NSRDB", "B1", "anchored", "nsrdb_trained"])
    cols = ["NSRDB", "nsrdb_trained", "anchored"]
    tab = scores(df, cols + ["B1"])
    rng = np.random.default_rng(0)
    week = df.date.values.astype("datetime64[W]")
    wk = np.unique(week)
    idx = {w: np.nonzero(week == w)[0] for w in wk}
    draws = {c: [] for c in cols}
    for _ in range(boot):
        sel = np.concatenate([idx[w] for w in rng.choice(wk, wk.size)])
        g = df.iloc[sel]
        ref = _centred_mse(g, "B1")
        for c in cols:
            draws[c].append(1 - _centred_mse(g, c) / ref)
    ci = {c: np.percentile(v, [5, 95]) for c, v in draws.items()}
    tab["skill_lo90"] = [ci[m][0] if (s == "all" and m in ci) else np.nan
                         for s, m in zip(tab.station_id, tab.model)]
    tab["skill_hi90"] = [ci[m][1] if (s == "all" and m in ci) else np.nan
                         for s, m in zip(tab.station_id, tab.model)]
    tab.to_csv(runs / main_group / "eval" / f"anchor_folds_{model}.csv", index=False)
    return tab


def score_networks(runs: Path, ev: str = "eval_meso", boot: int = 500) -> pd.DataFrame:
    """Every model at every station, pooled per network, centred skill vs B1.

    Anchored models: at a fold's own stations the fold's model (its target
    never saw them); at stations used by neither fold's error model (the
    Mesonet, USCRN stations added later) the mean of the two folds."""
    def pairs(group, cols, rename=None):
        p = pd.read_csv(runs / group / ev / "station_pairs.csv", parse_dates=["date"])
        keep = ["station_id", "date"] + cols
        return p[keep].rename(columns=rename or {})

    base = pd.read_csv(runs / "data_store" / ev / "station_pairs.csv", parse_dates=["date"])
    df = base[["station_id", "network", "date", "obs", "NSRDB", "B1", "xgb_B_modis",
               "swin_B_modis"]].rename(columns={"xgb_B_modis": "XGB B+MODIS",
                                                "swin_B_modis": "Swin B+MODIS"})
    goes = pd.read_csv(runs / "data_store_goes" / ev / "station_pairs.csv", nrows=0).columns
    gcols = {c: lab for c, lab in (("xgb_B_cod", "XGB B+COD"), ("swin_B_cod", "Swin B+COD"),
                                   ("swin_C_cod", "Swin C+COD")) if c in goes}
    df = df.merge(pairs("data_store_goes", list(gcols), gcols), on=["station_id", "date"],
                  how="left")
    folded = {"Swin B+MODIS anchored": ("data_store_corr_{}", "swin_B_modis"),
              "XGB B+MODIS anchored": ("data_store_corr_{}", "xgb_B_modis"),
              "XGB B+COD anchored": ("data_store_goes_corr_{}", "xgb_B_cod"),
              "Swin B+COD anchored": ("data_store_goes_corr_{}", "swin_B_cod"),
              "XGB B+COD anchored v2": ("data_store_goes_corr2_{}", "xgb_B_cod"),
              "Swin B+COD anchored v2": ("data_store_goes_corr2_{}", "swin_B_cod"),
              "Swin B+MODIS anchored v2": ("data_store_corr2_{}", "swin_B_modis")}
    def have(g, c):
        f = runs / g / ev / "station_pairs.csv"
        return f.exists() and c in pd.read_csv(f, nrows=0).columns

    folded = {k: v for k, v in folded.items()
              if have(v[0].format("a"), v[1]) and have(v[0].format("b"), v[1])}
    for lab, (grp, col) in folded.items():
        fa = pairs(grp.format("a"), [col], {col: "_a"})
        fb_ = pairs(grp.format("b"), [col], {col: "_b"})
        m = df[["station_id", "date"]].merge(fa, how="left").merge(fb_, how="left")
        v = m[["_a", "_b"]].mean(axis=1).values
        folds = FOLDS2 if "corr2" in grp else FOLDS
        v = np.where(m.station_id.isin(folds["a"]), m["_a"], v)
        v = np.where(m.station_id.isin(folds["b"]), m["_b"], v)
        df[lab] = v
    models = ["NSRDB", "XGB B+MODIS", "Swin B+MODIS", *gcols.values(), *folded]
    df = df.dropna(subset=["obs", "B1"] + models)
    rows, rng = [], np.random.default_rng(0)
    for net, g in [("all", df)] + list(df.groupby("network")):
        tab = scores(g, models + ["B1"])
        tab = tab[tab.station_id == "all"].assign(network=net, stations=g.station_id.nunique())
        week = g.date.values.astype("datetime64[W]")
        wk = np.unique(week)
        idx = {w: np.nonzero(week == w)[0] for w in wk}
        draws = {m: [] for m in models}
        for _ in range(boot):
            sel = g.iloc[np.concatenate([idx[w] for w in rng.choice(wk, wk.size)])]
            ref = _centred_mse(sel, "B1")
            for m in models:
                draws[m].append(1 - _centred_mse(sel, m) / ref)
        tab["skill_lo90"] = [np.percentile(draws[m], 5) if m in draws else np.nan for m in tab.model]
        tab["skill_hi90"] = [np.percentile(draws[m], 95) if m in draws else np.nan for m in tab.model]
        rows.append(tab)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(runs / "data_store" / ev / "networks.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    er = sub.add_parser("errmodel")
    er.add_argument("--fit-net", default=None, help="e.g. uscrn")
    er.add_argument("--score-net", default=None, help="e.g. ok_mesonet")
    er.add_argument("--centred", action="store_true")
    sub.add_parser("networks")
    fo = sub.add_parser("folds")
    fo.add_argument("--model", default="xgb_B_modis")
    fo.add_argument("--group", default="data_store")
    k = sub.add_parser("krige")
    k.add_argument("--pairs", default=None, help="station_pairs.csv with model columns")
    b = sub.add_parser("build-target")
    b.add_argument("--exclude", nargs="+", required=True)
    b.add_argument("--fold", required=True)
    b.add_argument("--centred", action="store_true")
    a = ap.parse_args()
    import xgboost  # noqa: F401  (before torch: macOS OpenMP clash)

    if a.cmd == "networks":
        t = score_networks(Path("runs"))
        print(t[["network", "stations", "model", "n", "rmse", "bias", "skill_crmse_vs_B1",
                 "skill_lo90", "skill_hi90"]].round(3).to_string(index=False))
        raise SystemExit
    if a.cmd == "folds":
        t = score_folds(Path("runs"), a.model, a.group)
        print(t[t.station_id == "all"].round(3).to_string(index=False))
        print(t[t.station_id != "all"].pivot(index="station_id", columns="model",
                                              values="skill_crmse_vs_B1").round(3).to_string())
        raise SystemExit

    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    out = run_dir(dcfg, tcfg, "eval")
    if a.cmd == "errmodel":
        t = run_errmodel(fb, dcfg, tcfg, out, a.fit_net, a.score_net, a.centred)
    elif a.cmd == "krige":
        t = run_krige(fb, dcfg, tcfg, out, Path(a.pairs) if a.pairs else None)
    else:
        build_target(fb, dcfg, tcfg, a.exclude, a.fold, a.centred)
        raise SystemExit
    print(t[t.station_id.str.startswith("all")].round(4).to_string(index=False))
    print(t[t.station_id != "all"].pivot(index="station_id", columns="model",
                                          values="skill_crmse_vs_B1").round(3).to_string())

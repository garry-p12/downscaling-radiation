"""Austin-area inference maps: the spatial holdout up close.

    python -m evaluation.austin                  # cache (GOES config) + figures
    python -m evaluation.austin --res 5          # 5 km: cache built on Vista (below)

    # on Vista, then copy runs/figures/austin_cache_5km.npz back:
    DOWNSCALE_CONFIG_data=configs/data_vista_5km.yaml \
        python -m evaluation.austin --res 5 --cache runs/figures/austin_cache_5km.npz

Austin and its 1 deg buffer are never trained on, and the Austin 33 NW USCRN
station sits in anchoring fold a, so the fold-a models are the ones whose
target never saw it. One inference pass over the test years (2020-21) in the
GOES-window configuration caches, on a window around Austin:

  POWER (enclosing 1 deg cell), B1, NSRDB, the anchored targets (v1, v2 fold a),
  Swin B + COD trained on NSRDB / on anchored v1 / on anchored v2 (fold a),
  GOES cloud transmittance and developed-land fraction.

Figures (runs/figures/):
  a1_austin_cases     four test days, one per season, picked by a fixed rule
                      (largest NSRDB spread over the window in that season)
  a2_austin_maps      test-period mean GHI, anchoring shift and RMSE maps
  a3_austin_station   Austin 33 NW: daily series, scatter and scores
  a4_austin_10v5      one case day and the station scores at 10 km and 5 km
                      (needs both caches)

The 5 km figures carry a _5km suffix. The cache also stores the grid geometry
and the station pixel, so figures need no local store.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.figures import (
    C_AQUA,
    C_BASE,
    C_INK,
    C_MUTED,
    C_REF,
    C_SWIN,
    RUNS,
    UNITS,
    MapFrame,
    panel,
    plt,
    save,
    style,
)

CACHE = RUNS / "figures" / "austin_cache.npz"
RES_KM = 10
TAG = ""
BOX = (28.4, 32.5, -100.3, -95.3)          # lat_min, lat_max, lon_min, lon_max of the window
CITIES = {"Austin": (30.27, -97.74), "San Antonio": (29.42, -98.49), "Waco": (31.55, -97.15)}
STATION = "TX_Austin_33_NW"
MODELS = {  # cache key: (run group, run name)
    "swin_nsrdb": ("data_store_goes", "swin_B_cod"),
    "swin_v1": ("data_store_goes_corr_a", "swin_B_cod"),
    "swin_v2": ("data_store_goes_corr2_a", "swin_B_cod"),
}
MODELS_10KM = dict(MODELS)
MODELS_5KM = {k: (g.replace("data_store_goes", "data_store_5km"), n) for k, (g, n) in MODELS.items()}
LAB = {"POWER": "POWER 1° (input)", "B1": "B1 bilinear", "NSRDB": "NSRDB",
       "swin_nsrdb": "Swin, NSRDB target", "swin_v1": "Swin, anchored v1",
       "swin_v2": "Swin, anchored v2", "anch_v1": "Anchored target v1"}


def use_res(res: int) -> None:
    """Switch the module to the 10 km or 5 km grid."""
    global CACHE, RES_KM, TAG, MODELS
    RES_KM, TAG = res, ("" if res == 10 else f"_{res}km")
    CACHE = RUNS / "figures" / f"austin_cache{TAG}.npz"
    MODELS = MODELS_5KM if res == 5 else MODELS_10KM


def build_cache(path: Path) -> None:
    """Run in a GOES-config process: every test day, cropped to BOX."""
    with contextlib.suppress(ImportError):
        import xgboost  # noqa: F401  (before torch: macOS OpenMP clash)
    from data.build_store import open_store
    from evaluation.evaluate import predictor
    from training.common import builder, load_all
    from training.train_swin import SwinPredictor

    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    ds = open_store(dcfg)
    lat, lon = _latlon(dcfg)
    inbox = (lat >= BOX[0]) & (lat <= BOX[1]) & (lon >= BOX[2]) & (lon <= BOX[3])
    rr, cc = np.nonzero(inbox)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    days = np.nonzero(fb.split == "test")[0]
    crop = (slice(None), slice(r0, r1), slice(c0, c1))
    preds = {"B1": predictor("B1")}
    for k, (grp, name) in MODELS.items():
        preds[k] = SwinPredictor.from_checkpoint(RUNS / grp / name / "best.pt", fb)
    out = {k: np.empty((days.size, r1 - r0, c1 - c0), "float32") for k in preds}
    for i, t in enumerate(days):
        for k, p in preds.items():
            f = p(fb, int(t)) if k == "B1" else p(int(t))
            out[k][i] = f[r0:r1, c0:c1]
        if i % 100 == 0:
            print(f"[austin] day {i}/{days.size}", flush=True)
    out["NSRDB"] = fb.ghi[days][crop]
    out["POWER"] = np.stack([fb.c["power_ghi"][t].ravel()[fb.enc].reshape(fb.ny, fb.nx)
                             for t in days])[crop].astype("float32")
    out["tcld"] = fb.dyn["goes_tcld"][days][crop]
    out["anch_v1"] = ds["nsrdb_ghi_corr_a"].isel(time=days).values[crop]
    out["anch_v2"] = ds["nsrdb_ghi_corr_2a"].isel(time=days).values[crop]
    out["developed"] = fb.static["lc_developed"][r0:r1, c0:c1]
    meta = pd.read_csv(Path(dcfg["sources"]["stations"]["out_dir"]) / "stations.csv")
    s = meta.set_index("station_id").loc[STATION]
    geo = {"fx": ds.x.values, "fy": ds.y.values, "st_rc": np.array([int(s.row), int(s.col)])}
    geo.update({f"f_{k}": ds[k].values.astype(bool) for k in ("valid", "holdout", "exclusion")})
    np.savez_compressed(path, days=fb.times[days], rc=np.array([r0, r1, c0, c1]), **geo,
                        **{k: v.astype("float32") for k, v in out.items()})
    print(f"[austin] wrote {path}")


def _latlon(dcfg):
    from common import build_target_grid

    return build_target_grid(dcfg["domain"], dcfg["grid"]).latlon()


class Window:
    """MapFrame cropped to the cached window, with Austin-area overlays."""

    def __init__(self, frame: MapFrame, rc, station_ll):
        r0, r1, c0, c1 = (int(v) for v in rc)
        self.f, self.sl = frame, (slice(r0, r1), slice(c0, c1))
        dx = abs(frame.x[1] - frame.x[0])
        xs, ys = frame.x[c0:c1], frame.y[r0:r1]
        self.extent = (xs.min() - dx / 2, xs.max() + dx / 2, ys.min() - dx / 2, ys.max() + dx / 2)
        self.valid = frame.valid[self.sl]
        self.st_xy = frame.project(station_ll[1], station_ll[0])
        self.st_rc = (None, None)

    def show(self, ax, field, cities=True, **kw):
        im = ax.imshow(np.where(self.valid, field, np.nan), extent=self.extent, origin="upper",
                       interpolation="nearest", **kw)
        self.f.decorate(ax)
        ax.set_xlim(*self.extent[:2]); ax.set_ylim(*self.extent[2:])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_color(C_MUTED)
        ax.scatter(*self.st_xy, marker="^", s=22, color="white", edgecolor=C_REF, lw=0.7, zorder=6)
        if cities:
            for name, (la, lo) in CITIES.items():
                x, y = self.f.project(lo, la)
                ax.scatter(x, y, s=7, color=C_REF, zorder=6)
                ax.text(x + 4, y + 4, name, fontsize=5.2, zorder=6, color=C_REF,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        return im


def _frame(z):
    """MapFrame from the geometry saved in the cache (or the local 10 km store)."""
    if "fx" not in z:
        from data.build_store import open_store
        from training.common import load_all

        return MapFrame(open_store(load_all()[0]))
    import xarray as xr

    return MapFrame(xr.Dataset({k: (("y", "x"), z[f"f_{k}"]) for k in ("valid", "holdout", "exclusion")},
                               coords={"x": z["fx"], "y": z["fy"]}))


def _load(cache=None):
    z = dict(np.load(cache or CACHE, allow_pickle=True))
    meta = pd.read_csv(Path("data_store/raw/stations/stations.csv")).set_index("station_id")
    s = meta.loc[STATION]
    w = Window(_frame(z), z["rc"], (s.lat, s.lon))
    row, col = z.get("st_rc", (s.row, s.col))
    w.st_rc = (int(row) - int(z["rc"][0]), int(col) - int(z["rc"][2]))
    return z, w, s


def _station_obs(days) -> pd.Series:
    d = pd.read_csv("data_store/raw/stations/station_daily.csv", parse_dates=["date"])
    d = d[d.station_id == STATION].set_index("date").ghi_kwh
    return d.reindex(pd.DatetimeIndex(days))


def pick_days(z) -> list[int]:
    """Largest NSRDB spread over the window, one day per season (fixed rule)."""
    n = z["NSRDB"]
    ok = (np.isnan(n).mean(axis=(1, 2)) < 0.05) & (np.isnan(z["tcld"]).mean(axis=(1, 2)) < 0.2)
    sd = np.where(ok, np.nanstd(n, axis=(1, 2)), -1)
    month = pd.DatetimeIndex(z["days"]).month
    picks = []
    for months in ((3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 1, 2)):
        m = np.isin(month, months)
        picks.append(int(np.argmax(np.where(m, sd, -1))))
    return picks


def fig_cases(z, w, out: Path) -> None:
    days = pick_days(z)
    obs = _station_obs(z["days"])
    cols = ["tcld", "POWER", "B1", "NSRDB", "swin_nsrdb", "swin_v1"]
    titles = ["Cloud transmittance", LAB["POWER"], LAB["B1"], "NSRDB (product)",
              LAB["swin_nsrdb"], LAB["swin_v1"]]
    fig = plt.figure(figsize=(7.6, 5.9))
    gs = fig.add_gridspec(len(days), len(cols) + 1, width_ratios=[1] * len(cols) + [0.05],
                          wspace=0.04, hspace=0.1)
    sr, sc = w.st_rc
    for i, k in enumerate(days):
        v = np.concatenate([z[c][k][w.valid] for c in cols[1:]])
        lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
        o = obs.iloc[k]
        for j, c in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            if c == "tcld":
                w.show(ax, z[c][k], cities=(i == 0), cmap="Greys_r", vmin=0.2, vmax=1)
                ax.set_ylabel(pd.Timestamp(z["days"][k]).strftime("%-d %b %Y"), fontsize=7.5)
            else:
                im = w.show(ax, z[c][k], cities=False, cmap="cividis", vmin=lo, vmax=hi)
                val = z[c][k][sr, sc]
                ax.text(0.03, 0.03, f"{val:.2f}" + (f"  (obs {o:.2f})" if j == 1 and np.isfinite(o) else ""),
                        transform=ax.transAxes, fontsize=5.6, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
            if i == 0:
                ax.set_title(titles[j], fontsize=6.6, loc="left")
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, -1]))
        cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.3)
        if i == len(days) // 2:
            cb.set_label(f"GHI ({UNITS})", fontsize=6.5)
    fig.text(0.0, 0.03, "Austin area, test days (one per season: the largest NSRDB spread over this "
             "window). Solid box: holdout; dashed: 1° buffer; triangle: USCRN Austin 33 NW. Numbers: "
             "each field at the station pixel; obs = station daily total. Left column: white = clear, "
             "dark = optically thick cloud. The anchored model's target was built without this station.",
             fontsize=6.2, color=C_MUTED, wrap=True)
    save(fig, out, f"a1_austin_cases{TAG}")


def _rmse_map(a, b):
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean((a - b) ** 2, axis=0))


def fig_maps(z, w, out: Path) -> None:
    n = z["NSRDB"]
    mean = {k: np.nanmean(z[k], axis=0) for k in ("NSRDB", "B1", "swin_v1", "anch_v1")}
    rm_b1, rm_sw = _rmse_map(z["B1"], n), _rmse_map(z["swin_nsrdb"], n)
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.16)
    lo = min(np.nanpercentile(mean[k][w.valid], 1) for k in ("NSRDB", "B1", "swin_v1"))
    hi = max(np.nanpercentile(mean[k][w.valid], 99) for k in ("NSRDB", "B1", "swin_v1"))

    def cbar(ax, im, label):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label, fontsize=6.3); cb.ax.tick_params(labelsize=5.8, length=1.5)
        cb.outline.set_linewidth(0.3)

    kw = dict(cmap="cividis", vmin=lo, vmax=hi)
    for j, (k, t) in enumerate((("NSRDB", "Mean GHI, NSRDB"), ("B1", "Mean GHI, B1 bilinear"),
                                ("swin_v1", "Mean GHI, Swin anchored v1"))):
        ax = fig.add_subplot(gs[0, j]); im = w.show(ax, mean[k], cities=(j == 0), **kw)
        ax.set_title(t, loc="left", fontsize=7.2); panel(ax, "abc"[j])
        cbar(ax, im, f"GHI ({UNITS})")
    # anchoring shift, B1 RMSE, Swin RMSE vs NSRDB
    sh = mean["anch_v1"] - mean["NSRDB"]
    e = float(np.nanpercentile(np.abs(sh[w.valid]), 98))
    ax = fig.add_subplot(gs[1, 0])
    im = w.show(ax, sh, cities=False, cmap="RdBu_r", vmin=-e, vmax=e)
    ax.set_title("Anchoring shift (target v1 − NSRDB)", loc="left", fontsize=7.2); panel(ax, "d")
    cbar(ax, im, f"mean difference ({UNITS})")
    rmax = float(np.nanpercentile(rm_b1[w.valid], 99))
    for j, (r, t) in enumerate(((rm_b1, "RMSE vs NSRDB, B1"), (rm_sw, "RMSE vs NSRDB, Swin (NSRDB target)")),
                               start=1):
        ax = fig.add_subplot(gs[1, j])
        im = w.show(ax, r, cities=False, cmap="magma_r", vmin=0, vmax=rmax)
        ax.set_title(t, loc="left", fontsize=7.2); panel(ax, "ef"[j - 1])
        m = w.valid & np.isfinite(r)
        ax.text(0.03, 0.03, f"window mean {np.nanmean(r[m]):.3f}", transform=ax.transAxes,
                fontsize=5.8, zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        if j == 2:
            ax.contour(*np.meshgrid(np.linspace(w.extent[0], w.extent[1], r.shape[1]),
                                    np.linspace(w.extent[3], w.extent[2], r.shape[0])),
                       z["developed"], levels=[0.5], colors=[C_AQUA], linewidths=0.9, zorder=5)
        cbar(ax, im, f"daily RMSE ({UNITS})")
    fig.text(0.0, 0.04, "Test years 2020–21. (d) The station-anchored target lowers NSRDB over the whole "
             "area. (f) Green contour: pixels over 50% developed land; with optical depth as input, "
             "their error is no larger than their surroundings'.", fontsize=6.2, color=C_MUTED)
    save(fig, out, f"a2_austin_maps{TAG}")


def _scores(obs, pred, b1):
    ok = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(b1)
    e, e1 = pred[ok] - obs[ok], b1[ok] - obs[ok]
    skill = 1 - np.var(e) / np.var(e1)
    return float(np.sqrt(np.mean(e ** 2))), float(e.mean()), float(skill), int(ok.sum())


def fig_station(z, w, out: Path) -> None:
    sr, sc = w.st_rc
    obs = _station_obs(z["days"]).values
    at = {k: z[k][:, sr, sc] for k in ("B1", "NSRDB", "swin_nsrdb", "swin_v1", "swin_v2")}
    t = pd.DatetimeIndex(z["days"])
    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], width_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.62)
    # (a) two spring months
    ax = fig.add_subplot(gs[0, :])
    m = (t >= "2021-04-01") & (t <= "2021-05-31")
    ax.plot(t[m], obs[m], color=C_REF, lw=1.6, label="Station (USCRN)", zorder=5)
    ax.plot(t[m], at["B1"][m], color=C_BASE, lw=1.0, ls=(0, (3, 2)), label="B1 bilinear POWER")
    ax.plot(t[m], at["NSRDB"][m], color=C_MUTED, lw=1.0, label="NSRDB")
    ax.plot(t[m], at["swin_v1"][m], color=C_AQUA, lw=1.4, label="Swin, anchored v1")
    ax.set_ylabel(f"GHI ({UNITS})"); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.16))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%-d %b"))
    panel(ax, "a", x=-0.05, y=1.04)
    # (b, c) scatter NSRDB and Swin v1 vs obs
    lim = (0, np.nanmax(obs) * 1.05)
    for j, (k, col) in enumerate((("NSRDB", C_MUTED), ("swin_v1", C_AQUA))):
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(obs, at[k], s=3, color=col, alpha=0.45, lw=0)
        ax.plot(lim, lim, color=C_INK, lw=0.7, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        r, b, s, _ = _scores(obs, at[k], at["B1"])
        ax.set_title(f"{LAB.get(k, k)}\nRMSE {r:.2f}, bias {b:+.2f}", loc="left", fontsize=7)
        ax.set_xlabel(f"Station ({UNITS})"); ax.set_ylabel("Estimate" if j == 0 else "")
        panel(ax, "bc"[j], x=-0.08)
    # (d) scores
    ax = fig.add_subplot(gs[1, 2])
    order = [("B1", C_BASE), ("NSRDB", C_MUTED), ("swin_nsrdb", C_SWIN), ("swin_v2", C_AQUA),
             ("swin_v1", C_AQUA)]
    names = {"B1": "B1", "NSRDB": "NSRDB", "swin_nsrdb": "Swin, NSRDB tgt",
             "swin_v2": "Swin, anch. v2", "swin_v1": "Swin, anch. v1"}
    for y, (k, col) in enumerate(order):
        r, b, s, n = _scores(obs, at[k], at["B1"])
        ax.barh(y, r, color=col, height=0.62, alpha=0.55 if k == "swin_v2" else 1)
        ax.text(r + 0.01, y, f"{r:.2f}  ({100 * s:+.0f}%)", va="center", fontsize=5.8, color=C_INK)
    ax.set_yticks(range(len(order)), [names[k] for k, _ in order])
    ax.set_xlim(0, 0.95); ax.set_xlabel(f"RMSE ({UNITS})")
    ax.set_title("2020–21 scores\n(day-to-day skill vs B1)", loc="left", fontsize=7)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x"); ax.set_axisbelow(True)
    panel(ax, "d", x=-0.45)
    fig.text(0.0, -0.02, f"USCRN Austin 33 NW (inside the holdout), test years 2020–21, {n} days. "
             f"Model values at the station's {RES_KM} km pixel. "
             "Anchored targets were built without this "
             "station (fold a).", fontsize=6.2, color=C_MUTED)
    save(fig, out, f"a3_austin_station{TAG}")


def fig_10v5(out: Path) -> None:
    """Same spring case day at both resolutions, plus the station scores."""
    use_res(10); z10, w10, _ = _load()
    use_res(5); z5, w5, _ = _load()
    k = pick_days(z10)[0]
    d5 = int(np.nonzero(z5["days"] == z10["days"][k])[0][0])
    cols = [("NSRDB", "NSRDB (product)"), ("swin_nsrdb", LAB["swin_nsrdb"]),
            ("swin_v1", LAB["swin_v1"]), ("swin_v2", LAB["swin_v2"])]
    v = np.concatenate([z10[c][k][w10.valid] for c, _ in cols])
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    fig = plt.figure(figsize=(7.4, 4.3))
    gs = fig.add_gridspec(2, len(cols) + 2, width_ratios=[1] * len(cols) + [0.05, 1.35],
                          wspace=0.05, hspace=0.08)
    obs10, obs5 = _station_obs(z10["days"]).values, _station_obs(z5["days"]).values
    for i, (z, w, d, res) in enumerate(((z10, w10, k, 10), (z5, w5, d5, 5))):
        for j, (c, t) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            im = w.show(ax, z[c][d], cities=(i == 0 and j == 0), cmap="cividis", vmin=lo, vmax=hi)
            if i == 0:
                ax.set_title(t, fontsize=6.6, loc="left")
            if j == 0:
                ax.set_ylabel(f"{res} km", fontsize=8)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, len(cols)]))
        cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.3)
        cb.set_label(f"GHI ({UNITS})", fontsize=6)
    # station RMSE at both resolutions
    ax = fig.add_subplot(gs[:, -1])
    keys = ["B1", "NSRDB", "swin_nsrdb", "swin_v2", "swin_v1"]
    names = {"B1": "B1", "NSRDB": "NSRDB", "swin_nsrdb": "Swin, NSRDB tgt",
             "swin_v2": "Swin, anch. v2", "swin_v1": "Swin, anch. v1"}
    for y, kk in enumerate(keys):
        for off, (z, w, obs, col) in ((-0.18, (z10, w10, obs10, C_BASE)), (0.18, (z5, w5, obs5, C_AQUA))):
            sr, sc = w.st_rc
            r, _, s, _ = _scores(obs, z[kk][:, sr, sc], z["B1"][:, sr, sc])
            ax.barh(y + off, r, height=0.34, color=col)
            ax.text(r + 0.01, y + off, f"{r:.2f} ({100 * s:+.0f}%)", va="center", fontsize=5.4, color=C_INK)
    ax.set_yticks(range(len(keys)), [names[x] for x in keys]); ax.invert_yaxis()
    ax.set_xlim(0, 0.95); ax.set_xlabel(f"RMSE at Austin 33 NW ({UNITS})")
    ax.barh([], [], color=C_BASE, label="10 km"); ax.barh([], [], color=C_AQUA, label="5 km")
    ax.legend(loc="lower right", fontsize=5.8)
    ax.yaxis.tick_right(); ax.tick_params(axis="y", length=0); ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.text(0.0, 0.0, f"{pd.Timestamp(z10['days'][k]):%-d %b %Y} (spring case day of a1), one colour scale. "
             "Right: 2020–21 daily RMSE at the station pixel, skill vs B1 at the same resolution in brackets.",
             fontsize=6.2, color=C_MUTED)
    save(fig, out, "a4_austin_10v5")


def main(res: int = 10) -> None:
    use_res(res)
    if not CACHE.exists():
        if res != 10:
            raise SystemExit(f"{CACHE} missing: build it on Vista (see module docstring)")
        env = dict(os.environ, DOWNSCALE_CONFIG_data="configs/data_goes.yaml")
        subprocess.run([sys.executable, "-m", "evaluation.austin", "--cache", str(CACHE)],
                       env=env, check=True)
    style()
    out = RUNS / "figures"
    z, w, _ = _load()
    fig_cases(z, w, out)
    fig_maps(z, w, out)
    fig_station(z, w, out)
    caches = [RUNS / "figures" / f"austin_cache{t}.npz" for t in ("", "_5km")]
    if all(c.exists() for c in caches):
        fig_10v5(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=10, choices=(10, 5))
    ap.add_argument("--cache", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cache:
        use_res(a.res)
        build_cache(Path(a.cache))
    else:
        main(a.res)

"""Publication figures for the results write-up.

    python -m evaluation.figures                 # all figures -> runs/figures/
    python -m evaluation.figures --no-cases      # skip the case-day inference

Everything but the case-day maps is drawn from the eval tables already on disk
(runs/<group>/eval*/); the case days run B0, B1, XGBoost B + MODIS and Swin
B + MODIS on two test days picked by a fixed rule (``pick_case_days``), so the
maps are not hand-chosen. Maps are drawn on the native EPSG:5070 grid.
Each figure is written as PNG (300 dpi) and PDF.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

RUNS = Path("runs")
MAIN, GOES, E5 = RUNS / "data_store", RUNS / "data_store_goes", RUNS / "data_store_era5l"
STATES = Path("data_store/raw/naturalearth/ne_50m_admin_1_states_provinces_lakes.shp")

# Fixed entity colours (validated categorical slots 1-3 + neutral greys).
C_SWIN, C_XGB, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_BASE, C_REF, C_INK, C_MUTED = "#8a8985", "#0b0b0b", "#1f1f1e", "#6b6a66"
FAMILY = {"swin": C_SWIN, "xgb": C_XGB}

LABEL = {"B0": "B0 bilinear GHI", "B1": "B1 bilinear kt", "B2": "B2 terrain",
         "B3": "B3 climatology", "xgb_A": "XGBoost A", "xgb_B": "XGBoost B",
         "xgb_B_modis": "XGBoost B + MODIS", "xgb_C": "XGBoost C (GOES)",
         "swin_A": "Swin A", "swin_A_ssim": "Swin A, SSIM", "swin_A_q": "Swin A, quantile",
         "swin_A_amse": "Swin A, AMSE", "swin_A_amse_ssim": "Swin A, AMSE+SSIM",
         "swin_B": "Swin B", "swin_B_amse": "Swin B, AMSE",
         "swin_B_modis": "Swin B + MODIS", "swin_C": "Swin C (GOES)",
         "NSRDB": "NSRDB", "ERA5L": "ERA5-Land"}
UNITS = "kWh m$^{-2}$ day$^{-1}$"


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "text.color": C_INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 110,
        "pdf.fonttype": 42, "mathtext.default": "regular",
    })


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}")


def panel(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


# --------------------------------------------------------------------- maps
class MapFrame:
    """Grid geometry + overlays for the 10 km EPSG:5070 grid."""

    def __init__(self, ds):
        import geopandas as gpd
        from pyproj import Transformer

        self.x = ds.x.values / 1000.0
        self.y = ds.y.values / 1000.0
        dx, dy = abs(self.x[1] - self.x[0]), abs(self.y[1] - self.y[0])
        self.extent = (self.x.min() - dx / 2, self.x.max() + dx / 2,
                       self.y.min() - dy / 2, self.y.max() + dy / 2)
        self.valid = ds.valid.values.astype(bool)
        self.holdout = ds.holdout.values.astype(bool)
        self.exclusion = ds.exclusion.values.astype(bool)
        self.fwd = Transformer.from_crs(4326, 5070, always_xy=True)
        self.states = None
        if STATES.exists():
            st = gpd.read_file(STATES)
            st = st[st["iso_a2"].isin(["US", "MX"])].to_crs(5070)
            self.states = st.geometry.scale(1e-3, 1e-3, origin=(0, 0))   # m -> km

    def project(self, lon, lat):
        x, y = self.fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0

    def show(self, ax, field, **kw):
        f = np.where(self.valid, field, np.nan)
        im = ax.imshow(f, extent=self.extent, origin="upper", interpolation="nearest", **kw)
        self.decorate(ax)
        return im

    def decorate(self, ax, holdout: bool = True, lw: float = 0.45):
        if self.states is not None:
            self.states.boundary.plot(ax=ax, color="#3d3d3a", linewidth=lw, zorder=3)
        if holdout:
            from matplotlib import patheffects as pe

            halo = [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]
            for m, ls in ((self.holdout, "-"), (self.exclusion, (0, (2, 1.5)))):
                cs = ax.contour(self.x, self.y, m.astype(float), levels=[0.5], colors="#0b0b0b",
                                linewidths=0.8, linestyles=[ls], zorder=4)
                cs.set(path_effects=halo)
        ax.set_xlim(*self.extent[:2])
        ax.set_ylim(*self.extent[2:])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_color("#9a9994")


def station_xy(frame: MapFrame, ids) -> pd.DataFrame:
    meta = pd.read_csv("data_store/raw/stations/stations.csv")
    meta = meta[meta.station_id.isin(ids)].copy()
    meta["px"], meta["py"] = frame.project(meta.lon.values, meta.lat.values)
    return meta


def fig_domain(fb, ds, frame: MapFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    dem = ds["dem"].values
    terrain = mcolors.LinearSegmentedColormap.from_list(
        "t", ["#f4f3ee", "#d9d4c4", "#a89d84", "#6e6250", "#3f372d"])
    im = frame.show(ax, dem, cmap=terrain, vmin=0, vmax=float(np.nanpercentile(dem, 99.5)))
    # POWER 1-degree cell edges
    for lon in np.arange(-105, -92):
        lat = np.linspace(26, 37, 60)
        ax.plot(*frame.project(np.full_like(lat, lon), lat), color=C_SWIN, lw=0.5,
                alpha=0.75, zorder=2)
    for lat in np.arange(26, 38):
        lon = np.linspace(-106, -92, 60)
        ax.plot(*frame.project(lon, np.full_like(lon, lat)), color=C_SWIN, lw=0.5,
                alpha=0.75, zorder=2)
    frame.decorate(ax)
    ids = pd.read_csv(MAIN / "eval" / "station_pairs.csv").station_id.unique()
    st = station_xy(frame, ids)
    ax.scatter(st.px, st.py, marker="^", s=34, color=C_XGB, edgecolor="white",
               linewidth=0.7, zorder=6)
    for _, s in st.iterrows():
        name = s.station_id.split("_", 1)[1].replace("_", " ")
        left = s.px > frame.extent[0] + 0.75 * (frame.extent[1] - frame.extent[0])
        ax.annotate(name, (s.px, s.py), xytext=(0 if left else 4, -12 if left else 3),
                    textcoords="offset points", ha="center" if left else "left",
                    fontsize=6, color=C_INK, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Elevation (m)")
    cb.outline.set_linewidth(0.4)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=C_SWIN, lw=0.8, label="POWER 1° cells"),
               Line2D([], [], color="k", lw=0.9, label="Austin holdout"),
               Line2D([], [], color="k", lw=0.9, ls=(0, (2, 1.5)), label="Holdout + 1° buffer"),
               Line2D([], [], marker="^", color=C_XGB, ls="", markeredgecolor="white",
                      markersize=6, label="USCRN station")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.85)
    ax.set_title("Domain: 10 km EPSG:5070 grid, 27–35° N, 104–94° W", loc="left")
    save(fig, out, "fig1_domain")


# ----------------------------------------------------------------- case days
def pick_case_days(fb, n_per: int = 1) -> list[int]:
    """Highest spatial standard deviation of NSRDB GHI over the valid domain,
    one test day from the cool season (Oct-Mar) and one from the warm season
    (Apr-Sep), among days with <5% missing NSRDB and <20% missing MODIS."""
    test = np.nonzero(fb.split == "test")[0]
    g = fb.ghi[test][:, fb.valid]
    miss = np.isnan(g).mean(1)
    cf = fb.dyn.get("modis_cf")
    cmiss = np.isnan(cf[test][:, fb.valid]).mean(1) if cf is not None else np.zeros(len(test))
    sd = np.nanstd(g, 1)
    ok = (miss < 0.05) & (cmiss < 0.2)
    month = fb.times[test].month
    days = []
    for warm in (False, True):
        sel = ok & (((month >= 4) & (month <= 9)) == warm)
        order = np.argsort(-np.where(sel, sd, -np.inf))[:n_per]
        days += [int(test[i]) for i in order]
    return days


def case_predictions(fb, dcfg, tcfg, days: list[int], cache: Path) -> dict:
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["days"]) == days:
            return {k: z[k] for k in z.files}
    from evaluation.evaluate import load_predictors

    want = ["B0", "B1", "xgb_B_modis", "swin_B_modis"]
    preds = load_predictors(fb, dcfg, tcfg, want)
    res = {"days": np.array(days)}
    for n in want:
        res[n] = np.stack([preds[n](t) for t in days]).astype("float32")
    res["NSRDB"] = fb.ghi[days]
    res["POWER"] = np.stack([fb.c["power_ghi"][t].ravel()[fb.enc].reshape(fb.ny, fb.nx)
                             for t in days]).astype("float32")
    res["MODIS"] = fb.dyn["modis_cf"][days] if "modis_cf" in fb.dyn else np.full_like(res["B1"], np.nan)
    np.savez(cache, **res)
    return res


def _rmse(a, b, m):
    d = (a - b)[m]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2)))


def fig_case(fb, frame: MapFrame, res: dict, k: int, out: Path, name: str) -> None:
    t = int(res["days"][k])
    day = pd.Timestamp(fb.times[t])
    obs = res["NSRDB"][k]
    cols = [("POWER", "POWER 1° (input)"), ("B1", "B1 bilinear kt"),
            ("xgb_B_modis", "XGBoost B+MODIS"), ("swin_B_modis", "Swin B+MODIS")]
    v = obs[frame.valid]
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    errs = [res[c][k] - obs for c, _ in cols]
    emax = float(np.nanpercentile(np.abs(np.concatenate([e[frame.valid] for e in errs])), 98))
    emax = np.ceil(emax * 2) / 2

    fig = plt.figure(figsize=(7.4, 3.75))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.055], wspace=0.04, hspace=0.12)
    ghi_kw = dict(cmap="cividis", vmin=lo, vmax=hi)
    err_kw = dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -emax, emax))

    def title(ax, letter, text, color=C_INK):
        ax.set_title(r"$\bf{" + letter + "}$  " + text, loc="left", fontsize=7.5, color=color)

    ax = fig.add_subplot(gs[0, 0]); im_g = frame.show(ax, obs, **ghi_kw)
    title(ax, "a", "NSRDB (target)")
    ax = fig.add_subplot(gs[1, 0])
    im_c = frame.show(ax, res["MODIS"][k], cmap="Greys_r", vmin=0, vmax=1)
    title(ax, "f", "MODIS cloud fraction")
    cax = ax.inset_axes([0.1, -0.1, 0.8, 0.045])
    cb = fig.colorbar(im_c, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["clear", "0.5", "cloudy"])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_linewidth(0.3)

    for j, ((c, ttl), e) in enumerate(zip(cols, errs), start=1):
        ax = fig.add_subplot(gs[0, j]); frame.show(ax, res[c][k], **ghi_kw)
        title(ax, "abcde"[j], ttl)
        ax = fig.add_subplot(gs[1, j]); im_e = frame.show(ax, e, **err_kw)
        r_all = _rmse(res[c][k], obs, frame.valid)
        r_ho = _rmse(res[c][k], obs, frame.valid & frame.holdout)
        ax.text(0.03, 0.04, f"RMSE {r_all:.2f}\nholdout {r_ho:.2f}", transform=ax.transAxes,
                fontsize=6, va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        title(ax, "fghij"[j], "error vs NSRDB", C_MUTED)

    cb = fig.colorbar(im_g, cax=fig.add_subplot(gs[0, 5]))
    cb.set_label(f"GHI ({UNITS})"); cb.outline.set_linewidth(0.4)
    cb = fig.colorbar(im_e, cax=fig.add_subplot(gs[1, 5]))
    cb.set_label(f"Error ({UNITS})"); cb.outline.set_linewidth(0.4)
    season = "warm season" if 4 <= day.month <= 9 else "cool season"
    fig.suptitle(f"{day:%d %B %Y} (test year, {season}): daily all-sky GHI and error vs NSRDB",
                 x=0.12, y=0.955, ha="left", fontsize=9)
    save(fig, out, name)


def fig_rmse_maps(frame: MapFrame, out: Path) -> None:
    z = np.load(MAIN / "eval" / "maps.npz")
    b1, sw = z["B1_rmse"], z["swin_B_modis_rmse"]
    c1, cs = z["B1_corr"], z["swin_B_modis_corr"]
    vmax = float(np.nanpercentile(np.r_[b1[frame.valid], sw[frame.valid]], 99))
    fig = plt.figure(figsize=(7.4, 4.0))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 0.05, 0.32, 1, 0.05], wspace=0.06,
                          hspace=0.16)
    kw = dict(cmap="magma_r", vmin=0, vmax=vmax)
    rows = [("RMSE", b1, sw, kw, "RMSE (" + UNITS + ")", "magma_r"),
            ("r", c1, cs, dict(cmap="viridis", vmin=0.9, vmax=1.0), "Correlation r", None)]
    for i, (lab, a, b, k, cl, _) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0]); im = frame.show(ax, a, **k)
        ax.set_title(f"B1, {lab}", loc="left"); panel(ax, "ad"[i])
        ax = fig.add_subplot(gs[i, 1]); frame.show(ax, b, **k)
        ax.set_title(f"Swin B + MODIS, {lab}", loc="left"); panel(ax, "be"[i])
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, 2])); cb.set_label(cl)
        cb.outline.set_linewidth(0.4)
        d = b - a
        dm = float(np.nanpercentile(np.abs(d[frame.valid]), 98))
        cmap = "RdBu_r" if lab == "RMSE" else "RdBu"
        ax = fig.add_subplot(gs[i, 4])
        im = frame.show(ax, d, cmap=cmap, norm=mcolors.TwoSlopeNorm(0, -dm, dm))
        better = (d < 0) if lab == "RMSE" else (d > 0)
        frac = 100 * float(np.mean(better[frame.valid]))
        ax.set_title(f"Swin − B1 ({frac:.1f}% of pixels improve)", loc="left")
        panel(ax, "cf"[i])
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[i, 5]))
        cb.set_label("Δ" + ("RMSE" if lab == "RMSE" else "r")); cb.outline.set_linewidth(0.4)
    fig.suptitle("Per-pixel daily skill against NSRDB, test years 2020–21 (blue = Swin better)",
                 x=0.12, y=1.0, ha="left", fontsize=9)
    save(fig, out, "fig4_skill_maps")


# --------------------------------------------------------------- scalar plots
def _family(m: str) -> str:
    return C_SWIN if m.startswith("swin") else C_XGB if m.startswith("xgb") else C_BASE


def fig_accuracy(out: Path) -> None:
    m = pd.read_csv(MAIN / "eval" / "metrics.csv")
    m = m[m.stratum == "all"].pivot(index="model", columns="cell", values="rmse")
    s = pd.read_csv(MAIN / "eval" / "structure.csv")
    s = s[s.cell == "both"].set_index("model")
    g = pd.read_csv(GOES / "eval" / "metrics.csv")
    g = g[g.stratum == "all"].pivot(index="model", columns="cell", values="rmse")
    gs_ = pd.read_csv(GOES / "eval" / "structure.csv")
    gs_ = gs_[gs_.cell == "both"].set_index("model")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.55))
    order = ["B1", "B3", "xgb_A", "xgb_B_modis", "swin_A", "swin_B_modis"]
    cells = [("temporal", "Temporal", C_SWIN, "o"), ("spatial", "Spatial", C_XGB, "s"),
             ("both", "Both", C_AQUA, "D")]
    yy = np.arange(len(order))[::-1]
    for y, mod in zip(yy, order):
        vals = [m.loc[mod, c] for c, *_ in cells]
        a.plot([min(vals), max(vals)], [y, y], color="#d8d7d2", lw=1.6, zorder=1,
               solid_capstyle="round")
    for c, lab, col, mk in cells:
        a.scatter(m.loc[order, c], yy, s=26, color=col, marker=mk, label=lab, zorder=3,
                  edgecolor="white", linewidth=0.6)
    a.set_yticks(yy, [LABEL[o] for o in order])
    a.set_xlabel(f"RMSE vs NSRDB ({UNITS})")
    a.grid(axis="x"); a.set_axisbelow(True)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, handletextpad=0.2,
             columnspacing=1.0)
    a.set_xlim(0.27, 0.49)
    a.set_title("Error by test cell, main window", loc="left"); panel(a, "a", x=-0.45)

    main_models = [k for k in m.index if k in s.index and k not in ("B0", "B2")]
    for mod in main_models:
        b.scatter(m.loc[mod, "both"], s.loc[mod, "ssim"], s=26, color=_family(mod),
                  edgecolor="white", linewidth=0.6, zorder=3, marker="o")
    for mod in ("xgb_C", "swin_C"):
        b.scatter(g.loc[mod, "both"], gs_.loc[mod, "ssim"], s=26, facecolor="white",
                  edgecolor=_family(mod), linewidth=1.1, zorder=3, marker="o")
    lab_off = {"B1": (-5, 5, "right"), "B3": (6, -1, "left"),
               "swin_A_amse": (6, -1, "left"), "swin_B_amse": (6, -2, "left"),
               "swin_A_ssim": (6, -1, "left"), "swin_A_amse_ssim": (2, -9, "left"),
               "xgb_B": (-5, 3, "right"),
               "swin_B_modis": (-6, 1, "right"), "xgb_B_modis": (6, -3, "left")}
    for mod, (dx, dy, ha) in lab_off.items():
        b.annotate(LABEL[mod], (m.loc[mod, "both"], s.loc[mod, "ssim"]), xytext=(dx, dy),
                   textcoords="offset points", fontsize=6, ha=ha, va="center", color=C_INK)
    b.annotate("Swin A/B, XGBoost A\n(no MODIS)", (0.4075, 0.405), xytext=(0, -14),
               textcoords="offset points", fontsize=6, ha="center", va="top", color=C_INK)
    for mod, dy in {"swin_C": 1, "xgb_C": -3}.items():
        b.annotate(LABEL[mod] + " †", (g.loc[mod, "both"], gs_.loc[mod, "ssim"]), xytext=(6, dy),
                   textcoords="offset points", fontsize=6, va="center", color=C_INK)
    b.set_xlabel(f"RMSE, holdout × test ({UNITS})  →  worse")
    b.set_ylabel("SSIM vs NSRDB  →  better")
    b.grid(True); b.set_axisbelow(True)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="o", ls="", color=C_SWIN, label="Swin"),
         Line2D([], [], marker="o", ls="", color=C_XGB, label="XGBoost"),
         Line2D([], [], marker="o", ls="", color=C_BASE, label="Baseline"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=C_MUTED, label="† GOES window")]
    b.legend(handles=h, loc="upper right", ncol=1, handletextpad=0.2)
    b.set_title("Accuracy vs structure, holdout × test", loc="left"); panel(b, "b", x=-0.16)
    save(fig, out, "fig5_accuracy_structure")


def fig_scales(out: Path) -> None:
    s = pd.read_csv(MAIN / "eval" / "structure.csv")
    s = s[s.cell == "both"].set_index("model")
    g = pd.read_csv(GOES / "eval" / "structure.csv")
    g = g[g.cell == "both"].set_index("model")
    bands = ["<30km", "30-60km", "60-120km", "120-240km"]
    xt = ["<30", "30–60", "60–120", "120–240"]
    series = [("B1", s, C_BASE, "-", "B1 bilinear kt"),
              ("swin_A", s, "#86b6ef", "-", "Swin A (no clouds)"),
              ("swin_A_amse", s, C_XGB, "--", "Swin A, AMSE loss"),
              ("swin_B_modis", s, C_SWIN, "-", "Swin B + MODIS"),
              ("swin_C", g, C_AQUA, "-", "Swin C + GOES †")]
    spec_m = json.loads((MAIN / "eval" / "spectra.json").read_text())
    spec_g = json.loads((GOES / "eval" / "spectra.json").read_text())

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.38))
    x = np.arange(len(bands))
    for mod, tab, col, ls, lab in series:
        ax[0].plot(x, [tab.loc[mod, f"corr_{b}"] for b in bands], color=col, ls=ls,
                   marker="o", ms=3.5, lw=1.6, label=lab)
        ax[1].plot(x, [tab.loc[mod, f"amp_{b}"] for b in bands], color=col, ls=ls,
                   marker="o", ms=3.5, lw=1.6)
        sp = (spec_g if tab is g else spec_m)[mod]["bands"]
        wl = np.array([r["wavelength_km"] for r in sp])
        ax[2].plot(wl, [r["coherence"] for r in sp], color=col, ls=ls, lw=1.6)
    for a in ax[:2]:
        a.set_xticks(x, xt); a.set_xlabel("Scale band (km)")
        a.grid(axis="y"); a.set_axisbelow(True)
    ax[0].set_ylabel("Placement: band correlation with NSRDB")
    ax[0].set_ylim(0, None)
    ax[1].axhline(1, color=C_MUTED, lw=0.7, ls=":")
    ax[1].text(0.02, 1.0, "target amplitude", fontsize=6, color=C_MUTED, va="bottom",
               transform=ax[1].get_yaxis_transform())
    ax[1].set_ylabel("Amplitude ratio (model / NSRDB)")
    ax[2].set_xscale("log"); ax[2].invert_xaxis()
    ax[2].set_xticks([1000, 300, 100, 30], ["1000", "300", "100", "30"])
    ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].axhline(0.5, color=C_MUTED, lw=0.7, ls=":")
    ax[2].set_xlabel("Wavelength (km)"); ax[2].set_ylabel("Spectral coherence with NSRDB")
    ax[2].set_ylim(0, 1); ax[2].grid(True, which="major"); ax[2].set_axisbelow(True)
    ax[0].legend(loc="upper left", fontsize=6.3, handlelength=2.2)
    ax[0].set_title("Texture in the right place", loc="left")
    ax[1].set_title("Texture amount", loc="left")
    ax[2].set_title("Coherence by wavelength", loc="left")
    for a, letter in zip(ax, "abc"):
        panel(a, letter, x=-0.2)
    fig.text(0.12, -0.06, "Holdout × test years; daily anomalies band-passed with difference-of-"
             "Gaussian filters. † GOES window (train 2018–mid-2019).", fontsize=6.5, color=C_MUTED)
    save(fig, out, "fig6_scale_bands")


def _skill_rows(region: str):
    main = pd.read_csv(MAIN / "eval" / "stations_summary.csv")
    goes = pd.read_csv(GOES / "eval" / "stations_summary.csv")
    e5 = pd.read_csv(E5 / "eval" / "stations_summary.csv")
    def get(df, mod):
        r = df[(df.region == region) & (df.model == mod)].iloc[0]
        return 100 * r.skill_crmse_vs_B1, 100 * r.skill_crmse_lo90, 100 * r.skill_crmse_hi90
    groups = [
        ("Reference products", [("NSRDB", e5, "NSRDB", C_REF), ("ERA5L", e5, "ERA5-Land", C_REF)]),
        ("Trained on NSRDB, main window",
         [(m, main, LABEL[m], _family(m)) for m in
          ("B3", "xgb_A", "xgb_B_modis", "swin_A", "swin_A_amse", "swin_B_modis")]),
        ("Trained on NSRDB, GOES window",
         [(m, goes, LABEL[m], _family(m)) for m in ("swin_A", "xgb_C", "swin_C")]),
        ("Trained on ERA5-Land",
         [(m, e5, lab, _family(m)) for m, lab in
          (("xgb_B", "XGBoost B + MODIS"), ("swin_B", "Swin B + MODIS"))]),
    ]
    rows = []
    for g, items in groups:
        for mod, df, lab, col in items:
            rows.append((g, lab, col, *get(df, mod)))
    return rows


def fig_stations(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    lo_lim = -62
    for ax, (region, title) in zip(axes, (("all", "All 7 USCRN stations (4,723 station-days)"),
                                          ("holdout", "Austin 33 NW, inside the holdout (684)"))):
        rows = _skill_rows(region)
        y, yt, yl, prev = 0, [], [], None
        heads = []
        for g, lab, col, v, lo, hi in rows:
            if g != prev:
                y -= 0.6 if prev else 0
                heads.append((y, g)); y -= 1; prev = g
            yt.append(y); yl.append(lab)
            clipped = lo < lo_lim
            ax.plot([max(lo, lo_lim), hi], [y, y], color=col, lw=1.6, solid_capstyle="round",
                    alpha=0.9)
            if v > lo_lim:
                ax.scatter(v, y, s=30, color=col, edgecolor="white", linewidth=0.7, zorder=3)
            if clipped:
                ax.annotate(f"← {v:.0f}% (off scale)", (lo_lim, y), xytext=(3, 0),
                            textcoords="offset points", fontsize=6, color=C_INK, va="center")
            y -= 1
        ax.axvline(0, color=C_INK, lw=0.8)
        ax.axvspan(lo_lim, 0, color="#f3f2ef", zorder=0)
        ax.set_xlim(lo_lim, 28)
        ax.set_xlabel("Day-to-day skill vs B1 (%), centred RMSE")
        ax.set_title(title, loc="left")
        ax.grid(axis="x"); ax.set_axisbelow(True)
        for hy, g in heads:
            ax.text(lo_lim + 1, hy, g, fontsize=6.6, fontweight="bold", color=C_MUTED, va="center")
        ax.text(0.99, 0.01, "better than POWER →", transform=ax.transAxes, ha="right",
                fontsize=6.3, color=C_MUTED)
    axes[0].set_yticks(yt, yl)
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(y + 0.4, 0.6)
    panel(axes[0], "a", x=-0.02); panel(axes[1], "b", x=-0.02)
    fig.text(0.0, -0.04, "Dots: skill = 1 − MSE/MSE(B1) after removing each station's mean error; "
             "bars: 90% week-block bootstrap interval (500 draws). Test years 2020–21.",
             fontsize=6.5, color=C_MUTED)
    save(fig, out, "fig7_station_skill")


def fig_target_swap(out: Path) -> None:
    cmp_ = pd.read_csv(MAIN / "eval_e5cmp" / "stations_summary.csv")
    e5 = pd.read_csv(E5 / "eval" / "stations_summary.csv")
    cmp_, e5 = cmp_[cmp_.region == "all"].set_index("model"), e5[e5.region == "all"].set_index("model")
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42)
    a = fig.add_subplot(gs[0])
    a.plot([0, 1], [0, 1], color="#d8d7d2", lw=0.8, zorder=0)
    pts = [("NSRDB", cmp_, C_REF, "s", "NSRDB\nproduct", (-6, 2), "right"),
           ("ERA5L", cmp_, C_REF, "s", "ERA5-Land product", (6, -3), "left"),
           ("B1", cmp_, C_BASE, "o", "B1", (-6, -6), "right")]
    b1 = cmp_.loc["B1", ["nsrdb_mimicry", "era5l_mimicry"]].values
    for mod, tab, tgt in (("swin_B_modis", cmp_, "N"), ("xgb_B_modis", cmp_, "N"),
                          ("swin_B", e5, "E"), ("xgb_B", e5, "E")):
        col = C_SWIN if tgt == "N" else C_XGB
        p = tab.loc[mod, ["nsrdb_mimicry", "era5l_mimicry"]].values
        a.annotate("", p, b1, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9,
                                               shrinkA=3, shrinkB=4, mutation_scale=7))
        a.scatter(*p, s=34, color=col, edgecolor="white", linewidth=0.7, zorder=3,
                  marker="o" if mod.startswith("swin") else "D")
    for mod, tab, col, mk, lab, off, ha in pts:
        p = tab.loc[mod, ["nsrdb_mimicry", "era5l_mimicry"]].values
        a.scatter(*p, s=34, color=col, marker=mk, edgecolor="white", linewidth=0.7, zorder=3)
        a.annotate(lab, p, xytext=off, textcoords="offset points", fontsize=6.3, ha=ha)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=C_SWIN, lw=1.4, label="trained on NSRDB"),
         Line2D([], [], color=C_XGB, lw=1.4, label="trained on ERA5-Land"),
         Line2D([], [], marker="o", ls="", color=C_MUTED, label="Swin B + MODIS"),
         Line2D([], [], marker="D", ls="", color=C_MUTED, label="XGBoost B + MODIS")]
    a.legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=6.2,
             handlelength=1.4, borderaxespad=0.2, frameon=True, facecolor="white",
             edgecolor="none", framealpha=0.9)
    a.set_xlim(0.2, 1.05); a.set_ylim(0.2, 1.05)
    a.set_xlabel("NSRDB mimicry"); a.set_ylabel("ERA5-Land mimicry")
    a.set_aspect("equal"); a.grid(True); a.set_axisbelow(True)
    a.set_title("Errors follow the target", loc="left"); panel(a, "a", x=-0.18)

    pm = pd.read_csv(MAIN / "eval_e5cmp" / "station_pairs.csv")
    pe = pd.read_csv(E5 / "eval" / "station_pairs.csv")
    lim = (0, 9)
    for k, (df, col, tag, ttl) in enumerate(((pm, "swin_B_modis", C_SWIN, "Swin B + MODIS, NSRDB target"),
                                            (pe, "swin_B", C_XGB, "Swin B + MODIS, ERA5-Land target"))):
        ax = fig.add_subplot(gs[k + 1])
        d = df[["obs", col]].dropna()
        cmap = mcolors.LinearSegmentedColormap.from_list("c", ["#ffffff", tag])
        ax.hexbin(d.obs, d[col], gridsize=40, extent=(*lim, *lim), cmap=cmap, mincnt=1,
                  bins="log", linewidths=0.1)
        ax.plot(lim, lim, color=C_INK, lw=0.7, ls="--")
        e = d[col] - d.obs
        r = np.corrcoef(d.obs, d[col])[0, 1]
        ax.text(0.04, 0.96, f"bias {e.mean():+.2f}\nRMSE {np.sqrt((e**2).mean()):.2f}\nr {r:.3f}\n"
                f"n {len(d):,}", transform=ax.transAxes, va="top", fontsize=6.3)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"USCRN GHI ({UNITS})")
        if k == 0:
            ax.set_ylabel(f"Model GHI ({UNITS})")
        ax.set_title(ttl, loc="left", fontsize=7.5)
        panel(ax, "bc"[k], x=-0.12)
    fig.text(0.0, 0.02, "Mimicry = correlation of a model's daily station errors with a "
             "product's errors at the same station-days. 7 USCRN stations, 2020–21.",
             fontsize=6.5, color=C_MUTED)
    save(fig, out, "fig8_target_swap")


# ------------------------------------------------------------ round 2
NET_MODELS = [  # (label in networks.csv, short label, colour, marker, filled)
    ("NSRDB", "NSRDB product", C_REF, "s", True),
    ("XGB B+MODIS", "XGBoost B + MODIS", C_XGB, "o", True),
    ("Swin B+MODIS", "Swin B + MODIS", C_SWIN, "o", True),
    ("XGB B+COD", "XGBoost B + COD", C_XGB, "D", True),
    ("Swin B+COD", "Swin B + COD", C_SWIN, "D", True),
    ("Swin C+COD", "Swin C + COD", C_SWIN, "^", True),
    ("XGB B+COD anchored", "XGBoost B + COD, anchored v1", C_XGB, "D", False),
    ("Swin B+COD anchored", "Swin B + COD, anchored v1", C_SWIN, "D", False),
    ("XGB B+COD anchored v2", "XGBoost B + COD, anchored v2", C_XGB, "P", True),
    ("Swin B+COD anchored v2", "Swin B + COD, anchored v2", C_SWIN, "P", True),
    ("Swin B+MODIS anchored v2", "Swin B + MODIS, anchored v2", C_SWIN, "X", True),
]


def fig_networks(out: Path) -> None:
    """Station skill and bias per network (Oklahoma Mesonet vs USCRN)."""
    f = RUNS / "data_store" / "eval_meso" / "networks.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t = t[t.station_id == "all"]
    nets = [("ok_mesonet", "Oklahoma Mesonet"), ("uscrn", "USCRN")]
    rows = [r for r in NET_MODELS if r[0] in set(t.model)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 0.8], wspace=0.1))
    yy = np.arange(len(rows))[::-1]
    for ax, (net, ttl) in zip(axes[:2], nets):
        g = t[t.network == net].set_index("model")
        n = int(g.stations.iloc[0])
        for y, (m, _, col, mk, fill) in zip(yy, rows):
            r = g.loc[m]
            ax.plot([100 * r.skill_lo90, 100 * r.skill_hi90], [y, y], color=col, lw=1.5,
                    solid_capstyle="round", alpha=0.85)
            ax.scatter(100 * r.skill_crmse_vs_B1, y, s=30, marker=mk, zorder=3,
                       facecolor=col if fill else "white", edgecolor=col if not fill else "white",
                       linewidth=1.1 if not fill else 0.6)
        ax.axvline(0, color=C_INK, lw=0.8)
        ax.set_xlim(-10, 55)
        ax.grid(axis="x"); ax.set_axisbelow(True)
        ax.set_xlabel("Day-to-day skill vs B1 (%)")
        ax.set_title(f"{ttl}\n{n} stations, {int(g.n.iloc[0]):,} station-days", loc="left")
    b = axes[2]
    for net, ttl, off, ls in (("ok_mesonet", "Mesonet", 0.15, "o"), ("uscrn", "USCRN", -0.15, "s")):
        g = t[t.network == net].set_index("model")
        for y, (m, *_rest) in zip(yy, rows):
            b.plot([0, g.loc[m, "bias"]], [y + off] * 2, color="#c9c8c3", lw=1.0, zorder=1)
        b.scatter(g.loc[[r[0] for r in rows], "bias"], yy + off, s=18, marker=ls, zorder=3,
                  color=C_MUTED if net == "uscrn" else C_INK, edgecolor="white", linewidth=0.5,
                  label=ttl)
    b.axvline(0, color=C_INK, lw=0.8)
    b.set_xlabel(f"Mean bias ({UNITS})")
    b.set_title("Bias at stations", loc="left")
    b.grid(axis="x"); b.set_axisbelow(True)
    b.legend(loc="lower right", handletextpad=0.1, fontsize=6.3)
    axes[0].set_yticks(yy, [r[1] for r in rows])
    axes[0].tick_params(axis="y", length=0)
    for ax, k in zip(axes, "abc"):
        panel(ax, k, x=-0.03)
    fig.text(0.0, -0.06, "Skill = 1 − MSE/MSE(B1) after removing each station's mean error; bars: "
             "90% week-block bootstrap interval. Anchored models are scored at each station by the "
             "fold whose\ncorrection never saw it (v1: corrections from 7 USCRN stations; v2: "
             "from all 46 stations of both networks). Test years 2020–21, GOES window.",
             fontsize=6.3, color=C_MUTED)
    save(fig, out, "fig11_networks")


def fig_cod_scores(out: Path) -> None:
    """What cloud optical depth adds in the GOES window (2018+), test years."""
    ev = GOES / "eval_meso"
    if not (ev / "metrics.csv").exists():
        return
    m = pd.read_csv(ev / "metrics.csv")
    m = m[m.stratum == "all"].pivot(index="model", columns="cell", values="rmse")
    s = pd.read_csv(ev / "structure.csv")
    s = s[s.cell == "both"].set_index("model")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                               gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.5))
    pairs = [("xgb_B", "xgb_B_cod", "XGBoost B", C_XGB), ("swin_B", "swin_B_cod", "Swin B", C_SWIN),
             ("xgb_C", "xgb_C_cod", "XGBoost C", C_XGB), ("swin_C", "swin_C_cod", "Swin C", C_SWIN)]
    gm = pd.read_csv(GOES / "eval" / "metrics.csv")
    gm = gm[gm.stratum == "all"].pivot(index="model", columns="cell", values="rmse")
    m = m.combine_first(gm)
    pairs = [p for p in pairs if p[0] in m.index and p[1] in m.index]
    cells = [("temporal", "time holdout"), ("spatial", "space holdout"), ("both", "space + time")]
    y = 0
    yt, yl = [], []
    for c, cl in cells:
        a.text(0.205, y - 0.05, cl, fontsize=6.5, fontweight="bold", color=C_MUTED)
        y -= 0.6
        for lo_, hi_, lab, col in pairs:
            v0, v1 = m.loc[lo_, c], m.loc[hi_, c]
            a.annotate("", (v1, y), (v0, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1,
                                                             shrinkA=0, shrinkB=2, mutation_scale=7))
            a.scatter(v0, y, s=16, color="white", edgecolor=col, linewidth=1, zorder=3)
            yt.append(y); yl.append(lab); y -= 1
        y -= 0.4
    a.set_yticks(yt, yl); a.tick_params(axis="y", length=0)
    a.set_xlim(0.2, 0.42)
    a.set_ylim(y + 0.6, 0.6)
    a.set_xlabel(f"RMSE vs NSRDB ({UNITS})")
    a.grid(axis="x"); a.set_axisbelow(True)
    a.set_title("Adding COD (open → arrow head)", loc="left"); panel(a, "a", x=-0.3)

    bands = ["<30km", "30-60km", "60-120km", "120-240km"]
    xs = np.arange(len(bands))
    mods = [("B1", "B1", C_BASE, ":"), ("xgb_B", "XGBoost B", C_XGB, "--"),
            ("xgb_B_cod", "XGBoost B + COD", C_XGB, "-"), ("swin_B", "Swin B", C_SWIN, "--"),
            ("swin_B_cod", "Swin B + COD", C_SWIN, "-")]
    sg = pd.read_csv(GOES / "eval" / "structure.csv")
    s = s.combine_first(sg[sg.cell == "both"].set_index("model"))
    for mod, lab, col, ls in mods:
        if mod in s.index:
            b.plot(xs, [s.loc[mod, f"corr_{k}"] for k in bands], color=col, ls=ls, lw=1.6,
                   marker="o", ms=3.5, label=f"{lab} (SSIM {s.loc[mod, 'ssim']:.2f})")
    b.set_xticks(xs, ["< 30", "30–60", "60–120", "120–240"])
    b.set_xlabel("Spatial scale band (km)")
    b.set_ylabel("Placement: corr. with NSRDB detail")
    b.set_ylim(0, 0.75); b.grid(axis="y"); b.set_axisbelow(True)
    b.legend(loc="upper left", fontsize=6.2, handlelength=2.2)
    b.set_title("Texture in the right place, by scale", loc="left"); panel(b, "b", x=-0.14)
    fig.text(0.0, -0.08, "GOES window (2018–21), test years 2020–21. COD = GOES-16 ABI cloud "
             "optical depth as sunlight-weighted two-stream transmittance (daily, a.m., p.m.). "
             "Panel b: holdout region × test years.", fontsize=6.3, color=C_MUTED)
    save(fig, out, "fig10_cod_scores")


COD_DAY = "2021-04-23"


def cod_case_cache(path: Path) -> None:
    """Run in a GOES-config process (see fig_cod_case): one test day, COD models."""
    from evaluation.evaluate import load_predictors
    from training.common import builder, load_all

    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    t = int(np.nonzero(pd.DatetimeIndex(fb.times) == pd.Timestamp(COD_DAY))[0][0])
    want = ["B1", "xgb_B", "xgb_B_cod", "swin_B_cod"]
    preds = load_predictors(fb, dcfg, tcfg, want)
    res = {k: preds[k](t).astype("float32") for k in want if k in preds}
    res["NSRDB"] = fb.ghi[t]
    res["tcld"] = fb.dyn["goes_tcld"][t] if "goes_tcld" in fb.dyn else np.full_like(fb.ghi[t], np.nan)
    np.savez(path, **res)


def fig_cod_case(frame: MapFrame, out: Path) -> None:
    import os
    import subprocess
    import sys

    cache = out / "cod_case.npz"
    if not cache.exists():
        env = dict(os.environ, DOWNSCALE_CONFIG_data="configs/data_goes.yaml")
        subprocess.run([sys.executable, "-m", "evaluation.figures", "--cod-cache", str(cache)],
                       env=env, check=True)
    r = dict(np.load(cache))
    obs = r["NSRDB"]
    cols = [(k, lab) for k, lab in (("xgb_B", "XGBoost B"), ("xgb_B_cod", "XGBoost B + COD"),
                                    ("swin_B_cod", "Swin B + COD")) if k in r]
    v = obs[frame.valid]
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    errs = [r[c] - obs for c, _ in cols]
    emax = np.ceil(float(np.nanpercentile(np.abs(np.concatenate([e[frame.valid] for e in errs])), 98)) * 2) / 2
    n = len(cols) + 1
    fig = plt.figure(figsize=(6.2, 3.75))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.055], wspace=0.04, hspace=0.12)
    ghi_kw = dict(cmap="cividis", vmin=lo, vmax=hi)
    err_kw = dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -emax, emax))

    def title(ax, letter, text, color=C_INK):
        ax.set_title(r"$\bf{" + letter + "}$  " + text, loc="left", fontsize=7.5, color=color)

    ax = fig.add_subplot(gs[0, 0]); im_g = frame.show(ax, obs, **ghi_kw); title(ax, "a", "NSRDB (target)")
    ax = fig.add_subplot(gs[1, 0])
    im_c = frame.show(ax, r["tcld"], cmap="Greys", vmin=0.2, vmax=1)
    title(ax, chr(ord("a") + n), "COD transmittance")
    cax = ax.inset_axes([0.1, -0.1, 0.8, 0.045])
    cb = fig.colorbar(im_c, cax=cax, orientation="horizontal", ticks=[0.2, 0.6, 1])
    cb.ax.set_xticklabels(["thick", "0.6", "clear"])
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
    for j, ((c, ttl), e) in enumerate(zip(cols, errs), start=1):
        ax = fig.add_subplot(gs[0, j]); frame.show(ax, r[c], **ghi_kw)
        title(ax, chr(ord("a") + j), ttl)
        ax = fig.add_subplot(gs[1, j]); im_e = frame.show(ax, e, **err_kw)
        ax.text(0.03, 0.04, f"RMSE {_rmse(r[c], obs, frame.valid):.2f}\nholdout "
                f"{_rmse(r[c], obs, frame.valid & frame.holdout):.2f}", transform=ax.transAxes,
                fontsize=6, va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        title(ax, chr(ord("a") + n + j), "error vs NSRDB", C_MUTED)
    cb = fig.colorbar(im_g, cax=fig.add_subplot(gs[0, n])); cb.set_label(f"GHI ({UNITS})")
    cb.outline.set_linewidth(0.4)
    cb = fig.colorbar(im_e, cax=fig.add_subplot(gs[1, n])); cb.set_label(f"Error ({UNITS})")
    cb.outline.set_linewidth(0.4)
    fig.suptitle(f"{pd.Timestamp(COD_DAY):%d %B %Y} (test year): what GOES cloud optical depth adds",
                 x=0.12, y=0.955, ha="left", fontsize=9)
    save(fig, out, "fig9_cod_case")


def fig_diffusion(frame: MapFrame, out: Path) -> None:
    """Residual diffusion (CorrDiff-style): samples, spread, calibration, texture."""
    ev = MAIN / "eval"
    ens_f, case_f = ev / "ens_diff_B.npz", out / "case_days.npz"
    if not (ens_f.exists() and case_f.exists() and (ev / "prob_metrics.csv").exists()):
        return
    z, c = np.load(ens_f), np.load(case_f)
    k = 1                                                        # the warm-season case day
    t = int(c["days"][k])
    i = int(np.nonzero(z["days"] == t)[0][0])
    ens = z["ens"][i].astype("float32")                          # (8, ny, nx)
    obs, mean = c["NSRDB"][k], c["swin_B_modis"][k]
    v = obs[frame.valid]
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    ghi_kw = dict(cmap="cividis", vmin=lo, vmax=hi)

    fig = plt.figure(figsize=(7.4, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.32)
    top = gs[0].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.055], wspace=0.04)
    maps = [(obs, "NSRDB (target)"), (mean, "Swin mean"),
            (ens[0], "member 1"), (ens[1], "member 2")]
    for j, (f, ttl) in enumerate(maps):
        ax = fig.add_subplot(top[j]); im = frame.show(ax, f, **ghi_kw)
        ax.set_title(r"$\bf{" + "abcd"[j] + "}$  " + ttl, loc="left", fontsize=7.5)
    ax = fig.add_subplot(top[4])
    im_s = frame.show(ax, ens.std(0, ddof=1), cmap="magma_r", vmin=0)
    ax.set_title(r"$\bf{e}$  ensemble sd (8)", loc="left", fontsize=7.5)
    cax = ax.inset_axes([0.1, -0.1, 0.8, 0.045])
    cb = fig.colorbar(im_s, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
    cb = fig.colorbar(im, cax=fig.add_subplot(top[5])); cb.set_label(f"GHI ({UNITS})")
    cb.outline.set_linewidth(0.4)

    bot = gs[1].subgridspec(1, 3, wspace=0.45)
    a = fig.add_subplot(bot[0])
    pm = pd.read_csv(ev / "prob_metrics.csv")
    labs = [("diff_B:mean", "mean only", C_BASE), ("diff_B:mean+gauss", "mean + Gaussian", C_XGB),
            ("diff_B", "diffusion", C_SWIN)]
    w = 0.26
    for j, (mod, lab, col) in enumerate(labs):
        vals = [pm[(pm.model == mod) & (pm.cell == cc)].crps.iloc[0] for cc in ("temporal", "both")]
        a.bar(np.arange(2) + (j - 1) * w, vals, w * 0.92, color=col, label=lab)
        for x_, v_ in zip(np.arange(2) + (j - 1) * w, vals):
            a.text(x_, v_ + 0.004, f"{v_:.2f}", ha="center", fontsize=5.6, color=C_INK)
    a.set_xticks([0, 1], ["time\nholdout", "space +\ntime"])
    a.set_ylabel(f"CRPS ({UNITS})"); a.set_ylim(0, 0.36)
    a.legend(loc="upper left", fontsize=6, handlelength=1, ncol=2, columnspacing=0.6)
    a.grid(axis="y"); a.set_axisbelow(True)
    a.set_title("Probabilistic skill", loc="left"); panel(a, "f", x=-0.25)

    b = fig.add_subplot(bot[1])
    h = np.array(json.loads((ev / "rank_hist.json").read_text())["diff_B"], float)
    h = h / h.sum() * len(h)
    b.bar(np.arange(len(h)), h, 0.86, color=C_SWIN)
    b.axhline(1, color=C_INK, lw=0.8, ls="--")
    b.set_xlabel("Rank of NSRDB among 16 members"); b.set_ylabel("Relative frequency")
    b.set_xticks([0, 8, 16])
    sk = pm[(pm.model == "diff_B") & (pm.cell == "both")].spread_skill.iloc[0]
    b.text(0.5, 0.95, f"spread/skill {sk:.2f}", transform=b.transAxes, ha="center", va="top",
           fontsize=6.3)
    b.set_title("Calibration (U = too narrow)", loc="left"); panel(b, "g", x=-0.25)

    cc = fig.add_subplot(bot[2])
    st = pd.read_csv(ev / "prob_structure.csv")
    st = st[st.cell == "temporal"].set_index("model")
    bands = ["<30km", "30-60km", "60-120km", "120-240km"]
    xs = np.arange(len(bands))
    for mod, lab, col, ls in (("diff_B:member", "one member", C_SWIN, "-"),
                              ("diff_B:ensmean", "ensemble mean", C_SWIN, "--"),
                              ("diff_B:mean", "Swin mean", C_BASE, ":")):
        cc.plot(xs, [st.loc[mod, f"amp_{k_}"] for k_ in bands], color=col, ls=ls, lw=1.5,
                marker="o", ms=3, label=lab)
    cc.set_xticks(xs, ["<30", "30–60", "60–120", "120–240"])
    cc.set_xlabel("Scale band (km)"); cc.set_ylabel("Detail amplitude / NSRDB")
    cc.set_ylim(0, 0.8); cc.grid(axis="y"); cc.set_axisbelow(True)
    cc.legend(loc="upper left", fontsize=6, handlelength=1.8)
    cc.set_title("Texture amplitude", loc="left"); panel(cc, "h", x=-0.25)
    fig.suptitle("Residual diffusion over the Swin mean: samples, spread and calibration",
                 x=0.12, y=0.93, ha="left", fontsize=9)
    fig.text(0.0, 0.0, "Maps: test day of fig 3. Panels f–h: test years 2020–21, scored against "
             "NSRDB; Gaussian reference uses the mean model's dev-year RMSE as its sd.",
             fontsize=6.3, color=C_MUTED)
    save(fig, out, "fig12_diffusion")


def main(cases: bool) -> None:
    style()
    out = RUNS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_accuracy(out)
    fig_scales(out)
    fig_stations(out)
    fig_target_swap(out)
    fig_networks(out)
    fig_cod_scores(out)

    with contextlib.suppress(ImportError):
        import xgboost  # noqa: F401  (before torch: macOS OpenMP clash)
    from training.common import builder, load_all

    dcfg, tcfg = load_all()
    from data.build_store import open_store

    ds = open_store(dcfg)
    frame = MapFrame(ds)
    fb = builder(dcfg, tcfg) if cases else None
    fig_domain(fb, ds, frame, out)
    fig_rmse_maps(frame, out)
    fig_cod_case(frame, out)
    fig_diffusion(frame, out)
    if cases:
        days = pick_case_days(fb)
        print("[fig] case days:", [str(fb.times[t].date()) for t in days])
        res = case_predictions(fb, dcfg, tcfg, days, out / "case_days.npz")
        for k, nm in enumerate(("fig2_case_cool", "fig3_case_warm")):
            fig_case(fb, frame, res, k, out, nm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cases", action="store_true")
    ap.add_argument("--cod-cache", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cod_cache:
        cod_case_cache(Path(a.cod_cache))
    else:
        main(not a.no_cases)

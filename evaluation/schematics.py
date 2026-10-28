"""Methodology and architecture figures (M1-M6) for the write-up.

    python -m evaluation.schematics            # -> runs/figures/m*.png|pdf

M1 experimental design, M2 pipeline, M3 residual formulation on a real
day, M4 mean models (XGBoost, Swin U-Net), M5 station anchoring, M6
residual diffusion (numbered in the write-up's order). Diagrams are drawn to the configured shapes
(configs/model.yaml, configs/train.yaml); the map panels use the store and
the cached case-day predictions of evaluation.figures (run that first).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from evaluation.figures import (
    C_AQUA,
    C_INK,
    C_MUTED,
    C_SWIN,
    C_XGB,
    RUNS,
    UNITS,
    MapFrame,
    panel,
    save,
    style,
)

# Tints of the entity colours for box fills (text stays in ink).
T_SWIN, T_XGB, T_AQUA, T_GREY = "#dce9f9", "#fce3d6", "#d5f0e5", "#efeeea"
T_TARGET, C_TARGET = "#f6edd5", "#a77b0b"
CASE_DAY = "2021-04-23"


# ------------------------------------------------------------------ drawing
def box(ax, x, y, w, h, text, fc=T_GREY, ec=C_MUTED, fs=5.6, bold_first=True, lw=0.7,
        ha="center", round_=1.2):
    """Rounded box; first line bold (a real font weight, not mathtext), lines
    stacked about the box centre with a line height computed in data units."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    lines = text.split("\n")
    fig = ax.figure
    px_per_unit = abs(ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1])
    lh = fs * 1.32 / 72 * fig.dpi / px_per_unit
    top = y + h / 2 + (len(lines) - 1) / 2 * lh
    tx = x + w / 2 if ha == "center" else x + 1.2
    for i, ln in enumerate(lines):
        ax.text(tx, top - i * lh, ln, ha=ha, va="center", fontsize=fs + (0.3 if i == 0 and bold_first else 0),
                color=C_INK, zorder=3,
                fontweight="bold" if (i == 0 and bold_first) else "normal")


def arrow(ax, p0, p1, color=C_MUTED, lw=0.9, style_="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style_, mutation_scale=7, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1, zorder=1))


def canvas(ax, w=100, h=100):
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")


def col_head(ax, x, y, text):
    ax.text(x, y, text.upper(), fontsize=6.0, color=C_MUTED, fontweight="bold", ha="center",
            va="bottom")


# ------------------------------------------------------------------ M1
def m1_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    canvas(ax, 100, 64)
    for x, t in ((10, "Sources"), (31, "10 km store"), (51.25, "Features & target"),
                 (72.5, "Models"), (91.75, "Products & scoring")):
        col_head(ax, x, 61.5, t)

    src = [("NASA POWER, 1° daily\nGHI, clear-sky, PW, AOD", 52, T_SWIN),
           ("Terrain & surface\nDEM, land cover, albedo", 44.5, T_SWIN),
           ("MODIS Terra / Aqua\n1 km cloud mask, 2001+", 37, T_SWIN),
           ("GOES-16 ABI, 2018+\ncloud mask, optical depth", 29.5, T_SWIN),
           ("NSRDB PSM v4\n4 km, 30 min (target)", 19, T_TARGET),
           ("Ground stations\n7 USCRN + 39 OK Mesonet", 8.5, T_AQUA)]
    for t, y, fc in src:
        box(ax, 0.5, y, 19, 6.3, t, fc=fc)
    box(ax, 23, 19, 16, 39.3, "Regrid & aggregate\n\nEPSG:5070, 10 km\n93 × 100 cells\n27–35°N, 104–94°W\n\n"
        "daily totals on the\nlocal solar day,\n2001–2021\n\nconservative: POWER,\nNSRDB; area fraction:\n"
        "clouds; bilinear:\ncontinuous fields", fc=T_GREY)
    for _, y, _ in src[:5]:
        arrow(ax, (19.6, y + 3.15), (23, y + 3.15))
    box(ax, 42.5, 38, 17.5, 20, "Feature tracks\nA   POWER + static (27)\nB   A + MODIS, PW, AOD\n"
        "C   B + GOES cloud mask\n+cod  optical depth\n+T    days t−1, t+1", fc=T_SWIN, ha="left")
    box(ax, 42.5, 19, 17.5, 15.5, "Residual target\nk = GHI / N\nN = TOA, or clear-sky\n\n"
        r"r = k$_{target}$ − bilinear k$_{POWER}$", fc=T_TARGET)
    box(ax, 42.5, 6.5, 17.5, 9, "Station anchoring\nlearned NSRDB error\n→ corrected target", fc=T_AQUA)
    arrow(ax, (39, 48), (42.5, 48))
    arrow(ax, (39, 26.75), (42.5, 26.75))
    arrow(ax, (19.6, 11.65), (42.5, 11.65))
    arrow(ax, (19.6, 20.5), (42.5, 13.5))
    arrow(ax, (51.25, 15.5), (51.25, 19), color=C_AQUA, lw=1.1)
    # feed bus: features and target into every model
    ax.plot([62, 62], [22.5, 54.5], color=C_MUTED, lw=0.9, zorder=1)
    ax.plot([60, 62], [48, 48], color=C_MUTED, lw=0.9, zorder=1)
    ax.plot([60, 62], [26.75, 26.75], color=C_TARGET, lw=0.9, zorder=1)
    mods = [("Baselines B0–B3\nbilinear GHI / kt,\nterrain, climatology", 50, T_GREY),
            ("XGBoost\nper pixel-day,\n3 M training rows", 40.5, T_XGB),
            ("Swin U-Net\n32 × 32 crops,\nMSE + 0.2 (1 − SSIM)", 31, T_SWIN),
            ("Residual diffusion\nEDM U-Net over the\nSwin mean, 16 members", 19, T_SWIN)]
    for t, y, fc in mods:
        box(ax, 64, y, 17, 8, t, fc=fc)
        arrow(ax, (62, y + 4), (64, y + 4))
    arrow(ax, (72.5, 31), (72.5, 27), color=C_SWIN, lw=1.0)
    ax.text(73.3, 29, "frozen mean", fontsize=5.2, color=C_SWIN, va="center")
    # products & scoring
    box(ax, 84, 41, 15.5, 17, "Reconstruct\n" + r"GHI = (bilinear k$_P$ + $\hat{r}$) · N" + "\nclip to POWER\n"
        "clear-sky ceiling", fc=T_GREY, fs=5.4)
    ax.plot([81.5, 81.5], [23, 54], color=C_MUTED, lw=0.9, zorder=1)
    for _, y, _ in mods:
        ax.plot([81, 81.5], [y + 4, y + 4], color=C_MUTED, lw=0.9, zorder=1)
    arrow(ax, (81.5, 49.5), (84, 49.5))
    box(ax, 84, 22, 15.5, 15.5, "vs NSRDB, grid\nRMSE in 3 test cells\nSSIM; scale-band\n"
        "placement, amplitude\nCRPS, rank histogram", fc=T_TARGET, fs=5.4)
    box(ax, 84, 3, 15.5, 15.5, "vs stations\nday-to-day skill vs B1\nbias, RMSE\n"
        "mimicry of NSRDB\nerrors, per network", fc=T_AQUA, fs=5.4)
    arrow(ax, (91.75, 41), (91.75, 37.5))
    arrow(ax, (91.75, 22), (91.75, 18.5))
    # independent truth: stations straight to scoring, under everything
    ax.plot([10, 10, 82], [8.5, 2.2, 2.2], color=C_AQUA, lw=1.0, ls=(0, (3, 2)), zorder=1)
    arrow(ax, (82, 2.2), (84, 5), color=C_AQUA, lw=1.0)
    ax.text(46, 3.0, "ground truth for scoring (never a model input)", fontsize=5.4, color=C_AQUA,
            ha="center", va="bottom")
    save(fig, out, "m2_pipeline")


# ------------------------------------------------------------------ M2
def residual_case(cache: Path) -> dict:
    if cache.exists():
        return dict(np.load(cache))
    from training.common import builder, load_all

    dcfg, tcfg = load_all()
    fb = builder(dcfg, tcfg)
    t = int(np.nonzero(pd.DatetimeIndex(fb.times) == pd.Timestamp(CASE_DAY))[0][0])
    res = {"t": np.array(t), "N": fb.norm(t).astype("float32"),
           "kb": fb.kt_bilinear_toa(t).astype("float32")}
    np.savez(cache, **res)
    return res


def m2_residual(frame: MapFrame, out: Path) -> None:
    c = np.load(RUNS / "figures" / "case_days.npz")
    rc = residual_case(RUNS / "figures" / "residual_case.npz")
    k = int(np.nonzero(c["days"] == int(rc["t"]))[0][0])
    N, kb = rc["N"], rc["kb"]
    power, nsrdb, swin = c["POWER"][k], c["NSRDB"][k], c["swin_B_modis"][k]
    kn = nsrdb / N
    r = kn - kb
    rh = swin / N - kb
    v = frame.valid
    glo, ghi_ = np.nanpercentile(nsrdb[v], 1), np.nanpercentile(nsrdb[v], 99)
    rmax = float(np.ceil(np.nanpercentile(np.abs(r[v]), 98) * 20) / 20)

    fig = plt.figure(figsize=(7.4, 3.55))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], wspace=0.06, hspace=0.42)
    panels = [
        (0, 0, power, "POWER 1° GHI (input)", dict(cmap="cividis", vmin=glo, vmax=ghi_)),
        (0, 1, kb, r"bilinear k$_{POWER}$ = GHI$_P$/N", dict(cmap="cividis", vmin=0, vmax=0.8)),
        (0, 2, kn, r"NSRDB k$_{target}$", dict(cmap="cividis", vmin=0, vmax=0.8)),
        (0, 3, r, r"target residual r", dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -rmax, rmax))),
        (1, 0, N, "normaliser N (TOA)", dict(cmap="Greys", vmin=np.nanmin(N[v]), vmax=np.nanmax(N[v]))),
        (1, 1, rh, r"Swin prediction $\hat{r}$", dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -rmax, rmax))),
        (1, 2, swin, r"GHI = (k$_P$ + $\hat{r}$) · N", dict(cmap="cividis", vmin=glo, vmax=ghi_)),
        (1, 3, swin - nsrdb, "error vs NSRDB", dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -1.5, 1.5))),
    ]
    ims = {}
    for i, (row, col, f, ttl, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[row, col])
        ims[(row, col)] = frame.show(ax, f, **kw)
        ax.set_title(r"$\bf{" + "abcdefgh"[i] + "}$  " + ttl, loc="left", fontsize=7.2)
        cax = ax.inset_axes([0.08, -0.09, 0.84, 0.04])
        cb = fig.colorbar(ims[(row, col)], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
        cb.locator = plt.MaxNLocator(4); cb.update_ticks()
    fig.suptitle(f"Residual formulation on {pd.Timestamp(CASE_DAY):%d %B %Y}: models predict r, "
                 "the part of the clearness index the bilinear POWER field misses",
                 x=0.11, y=1.0, ha="left", fontsize=8.5)
    fig.text(0.11, -0.02, f"GHI panels (a, g) in {UNITS}; index panels unitless; N in the same "
             "GHI units. r = d − b; g = (b + f) · e, clipped to the POWER clear-sky ceiling.",
             fontsize=6.2, color=C_MUTED)
    save(fig, out, "m3_residual")


# ------------------------------------------------------------------ M3
def m3_mean_models(out: Path) -> None:
    from common.config import load_config

    enc = load_config("model")["encoder"]
    dims = [enc["embed_dim"] * 2 ** i for i in range(len(enc["depths"]))]
    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.78, 2.22], height_ratios=[1, 0.5],
                          wspace=0.04, hspace=0.08)

    # (a) XGBoost
    a = fig.add_subplot(gs[:, 0]); canvas(a, 40, 100)
    a.text(0, 98, "a", fontsize=9, fontweight="bold", va="top")
    a.text(5, 98, "XGBoost, per pixel-day", fontsize=7.6, va="top")
    feats = ["kt_bilinear", "kt_clr_ratio", "kt_roll3, roll7", "dem, slope, svf", "land cover",
             "albedo", "doy, lat, lon", "modis_cf", "PW, AOD", "goes_tcld (+cod)"]
    for i, f in enumerate(feats):
        y = 85 - i * 4.4
        fc = T_SWIN if i < 7 else (T_XGB if i < 9 else T_AQUA)
        a.add_patch(Rectangle((1, y), 16, 4.0, fc=fc, ec="white", lw=0.8))
        a.text(9, y + 2.0, f, fontsize=5.2, ha="center", va="center")
    a.text(9, 41.5, "x (one pixel, one day)", fontsize=5.4, ha="center", color=C_MUTED)
    for x0, yt in ((27, 86), (33, 68)):
        nodes = [(x0, yt), (x0 - 3.5, yt - 6), (x0 + 3.5, yt - 6), (x0 - 5.5, yt - 12), (x0 - 1.5, yt - 12)]
        for p_, q_ in ((0, 1), (0, 2), (1, 3), (1, 4)):
            a.plot(*zip(nodes[p_], nodes[q_]), color=C_XGB, lw=0.8, zorder=2)
        for px, py in nodes:
            a.add_patch(plt.Circle((px, py), 1.0, color=C_XGB, zorder=3))
    a.text(30, 51, "+ … ≤ 2000 trees", fontsize=5.6, ha="center", color=C_MUTED)
    arrow(a, (17.5, 74), (23.5, 79))
    box(a, 21, 30, 18, 11, "output\n" + r"$\hat{r}$ = Σ tree leaves", fc=T_XGB)
    arrow(a, (30, 49), (30, 41.3))
    a.text(1, 25, "depth 8 · η 0.05 · early stop 50\n3 M pixel-days, stratified by\nthe INPUT sky class "
           "(POWER kt),\nnot the target's\nloss weight none / TOA /\nenergy, chosen on dev",
           fontsize=5.4, color=C_MUTED, va="top", linespacing=1.35)

    # (b) Swin U-Net
    b = fig.add_subplot(gs[0, 1]); canvas(b, 120, 62)
    b.text(0, 61, "b", fontsize=9, fontweight="bold", va="top")
    b.text(4, 61, "Swin U-Net (patch 2, window 4)", fontsize=7.6, va="top")
    ys, hs = [41, 25, 9], [11, 9, 8]
    ex, dx, w = 19, 68, 23
    box(b, 0.5, 41, 13, 11, "input crop\nC × 32 × 32\nC channels", fc=T_GREY)
    arrow(b, (13.5, 46.5), (ex, 46.5))
    b.text(0.5, 39, "patch embed:\n2×2 conv, stride 2", fontsize=5, va="top", color=C_MUTED)
    for i, (d, nh, dp) in enumerate(zip(dims, enc["num_heads"], enc["depths"])):
        res = 16 // 2 ** i
        box(b, ex, ys[i], w, hs[i], f"encoder stage {i + 1}\n{dp} blocks · {nh} heads\n{d} × {res} × {res}",
            fc=T_SWIN)
        if i < len(dims) - 1:
            arrow(b, (ex + w / 2, ys[i]), (ex + w / 2, ys[i + 1] + hs[i + 1]))
            b.text(ex + w / 2 + 1, (ys[i] + ys[i + 1] + hs[i + 1]) / 2, "patch merge ↓2",
                   fontsize=5, color=C_MUTED, va="center")
    for i in range(len(dims) - 1):
        res = 16 // 2 ** i
        box(b, dx, ys[i], w, hs[i], f"decoder stage {i + 1}\n{enc['depths'][i]} blocks · "
            f"{enc['num_heads'][i]} heads\n{dims[i]} × {res} × {res}", fc=T_SWIN)
        yc = ys[i] + hs[i] / 2
        arrow(b, (ex + w, yc), (dx, yc), color=C_SWIN, ls=(0, (3, 2)))
        b.text((ex + w + dx) / 2, yc + 0.8, "skip: concat → linear", fontsize=5, ha="center",
               color=C_SWIN, va="bottom")
    arrow(b, (ex + w, ys[2] + hs[2] / 2), (dx + w / 2, ys[1]), rad=0.2)
    b.text(57, 10.5, "patch expand ↑2", fontsize=5, color=C_MUTED, ha="center")
    arrow(b, (dx + w / 2, ys[1] + hs[1]), (dx + w / 2, ys[0]))
    b.text(dx + w / 2 + 1, (ys[0] + ys[1] + hs[1]) / 2, "patch expand ↑2", fontsize=5,
           color=C_MUTED, va="center")
    box(b, 96, 39, 23, 15, "head\nconv 3×3 · GELU\nconv 3×3\npixel shuffle ↑2\n" + r"$\hat{r}$: 1 × 32 × 32",
        fc=T_SWIN)
    arrow(b, (dx + w, 46.5), (96, 46.5))
    b.text(96, 36, "last conv zero-initialised:\ntraining starts at " + r"$\hat{r}$" + " = 0 (= B1)\n\n"
           "loss: masked MSE on k\n   + 0.2 (1 − SSIM)\nAdamW 2e-4, decay 0.05\n40 epochs, batch 16\n"
           "inference: whole grid,\n93 × 100 padded to 96 × 128",
           fontsize=5.2, color=C_MUTED, va="top", linespacing=1.35)

    # (c) Swin block pair
    c = fig.add_subplot(gs[1, 1]); canvas(c, 120, 30)
    c.text(0, 29, "c", fontsize=9, fontweight="bold", va="top")
    c.text(4, 29, "One Swin block pair (regular, then shifted windows)", fontsize=7.6, va="top")
    seq = [("LN", T_GREY), ("W-MSA\nwindow 4×4", T_SWIN), ("+", None), ("LN", T_GREY),
           ("MLP ×4\nGELU", T_SWIN), ("+", None), ("LN", T_GREY), ("SW-MSA\nshift 2", T_SWIN),
           ("+", None), ("LN", T_GREY), ("MLP ×4\nGELU", T_SWIN), ("+", None)]
    x = 1
    for n, (t, fc) in enumerate(seq):
        if fc is None:
            c.add_patch(plt.Circle((x + 1.6, 14), 1.6, fc="white", ec=C_INK, lw=0.7, zorder=3))
            c.text(x + 1.6, 14, "+", fontsize=6.5, ha="center", va="center", zorder=4)
            nxt = x + 3.2
        else:
            ww = 6 if t == "LN" else 13
            box(c, x, 9, ww, 10, t, fc=fc, bold_first=False, fs=5.3)
            nxt = x + ww
        if n < len(seq) - 1:
            arrow(c, (nxt, 14), (nxt + 1.8, 14))
        x = nxt + 1.8
    c.text(1, 4, "Each + adds the input of the preceding (LN, attention) or (LN, MLP) pair back "
           "(residual); stochastic depth rises to 0.1 with depth.\nShifting the 4×4 windows by 2 "
           "on alternate blocks lets neighbouring windows exchange information.",
           fontsize=5.3, color=C_MUTED, va="top", linespacing=1.35)
    save(fig, out, "m4_mean_models")


# ------------------------------------------------------------------ M4
def m4_diffusion(frame: MapFrame, out: Path) -> None:
    from common.config import load_config

    dc = load_config("train").get("diffusion", {})
    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.12)
    a = fig.add_subplot(gs[0]); canvas(a, 120, 60)
    panel(a, "a", x=0.0, y=0.97)
    a.text(3, 57, "Training (EDM, residual of a frozen mean)", fontsize=8, va="bottom")
    box(a, 1, 40, 16, 10, "inputs\nTrack B + MODIS\nchannels", fc=T_SWIN, fs=5.6)
    box(a, 22, 40, 16, 10, "Swin B+MODIS\n(frozen)\n" + r"$r_{mean}$", fc=T_SWIN, fs=5.6)
    box(a, 1, 22, 16, 10, "NSRDB target\n" + r"$r_{true}$", fc=T_TARGET, fs=5.6)
    box(a, 22, 22, 16, 10, "residual\n" + r"e = ($r_{true}$ − $r_{mean}$) / s", fc=T_TARGET, fs=5.6)
    arrow(a, (17, 45), (22, 45))
    arrow(a, (17, 27), (22, 27))
    arrow(a, (30, 40), (30, 32))
    box(a, 43, 22, 17, 10, "add noise\n" + r"$e_\sigma$ = e + σ ε" + "\n" + r"ln σ ~ N(−1.2, 1.2²)",
        fc=T_GREY, fs=5.6)
    arrow(a, (38, 27), (43, 27))
    box(a, 63, 30, 31, 22, "conditional U-Net D\n"
        f"64 / 128 / 256 ch, {2} ResBlocks / level\nFiLM on Fourier(ln σ / 4)\n"
        "EDM preconditioning\n" + r"D = $c_{skip}$ $e_\sigma$ + $c_{out}$ F($c_{in}$ $e_\sigma$; σ, cond)",
        fc=T_SWIN, fs=5.6)
    arrow(a, (60, 27), (63, 36))
    a.plot([9, 9, 50], [50, 53, 53], color=C_SWIN, lw=0.9, ls=(0, (3, 2)), zorder=1)
    arrow(a, (50, 53), (63, 47), color=C_SWIN, ls=(0, (3, 2)))
    arrow(a, (38, 45), (63, 43), color=C_SWIN, ls=(0, (3, 2)))
    a.text(30, 54, "condition = input channels + " + r"$r_{mean}$", fontsize=5.4, color=C_SWIN,
           ha="center", va="bottom")
    box(a, 98, 30, 21.5, 22, "loss\n" + r"λ(σ) ‖D − e‖²" + "\nmasked: holdout,\nbuffer, missing\n"
        f"\n{int(dc.get('steps', 60000)) // 1000}k steps · batch {dc.get('batch', 64)}\n"
        f"crops {dc.get('crop', 32)} · EMA {dc.get('ema', 0.999)}", fc=T_GREY, fs=5.6)
    arrow(a, (94, 41), (98, 41))
    # sampling strip
    a.text(3, 13.5, "Sampling", fontsize=8, va="bottom")
    steps = [(r"z ~ N(0, $σ_{max}$²)", T_GREY), ("18 Heun steps\n(2nd-order ODE)", T_SWIN),
             (r"ê · s + $r_{mean}$", T_TARGET), ("GHI member\n" + r"(k$_P$ + r) · N", T_GREY),
             (f"× {dc.get('members', 16)} members\nCRPS, spread", T_AQUA)]
    x = 3
    for t, fc in steps:
        box(a, x, 1, 19, 10, t, fc=fc, fs=5.6, bold_first=False)
        if x > 3:
            arrow(a, (x - 4, 6), (x, 6))
        x += 23.5

    # (b) real samples
    ev = RUNS / "data_store" / "eval"
    z, c = np.load(ev / "ens_diff_B.npz"), np.load(RUNS / "figures" / "case_days.npz")
    k = 1
    t = int(c["days"][k])
    i = int(np.nonzero(z["days"] == t)[0][0])
    ens = z["ens"][i].astype("float32")
    mean, obs = c["swin_B_modis"][k], c["NSRDB"][k]
    sub = gs[1].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05], wspace=0.05)
    emax = 1.2
    kw = dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -emax, emax))
    fields = [(obs - mean, r"NSRDB − $r_{mean}$ (truth)"), (ens[0] - mean, "sample 1"),
              (ens[1] - mean, "sample 2"), (ens[2] - mean, "sample 3"),
              (ens.mean(0) - mean, "mean of 8 samples")]
    for j, (f, ttl) in enumerate(fields):
        ax = fig.add_subplot(sub[j])
        im = frame.show(ax, f, **kw)
        ax.set_title(r"$\bf{" + ("b" if j == 0 else "") + "}$  " + ttl if j == 0 else ttl,
                     loc="left", fontsize=7)
    cb = fig.colorbar(im, cax=fig.add_subplot(sub[5]))
    cb.set_label(f"residual GHI\n({UNITS})", fontsize=6); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.125, 0.04, f"{pd.Timestamp(CASE_DAY):%d %B %Y}: the "
             "residual the mean misses, and three draws of it. Draws carry texture of the right "
             "amount; their mean collapses toward zero.", fontsize=6.2, color=C_MUTED)
    save(fig, out, "m6_diffusion")


# ------------------------------------------------------------------ M5
def m5_anchoring(frame: MapFrame, out: Path) -> None:
    from training.common import load_all

    dcfg, _ = load_all()
    from data.build_store import open_store

    ds = open_store(dcfg)
    f2 = {k: (Path("slurm") / f"fold2_{k}.txt").read_text().split() for k in ("a", "b")}
    meta = pd.read_csv("data_store/raw/stations/stations.csv")
    meta = meta[meta.in_grid].copy()
    meta["px"], meta["py"] = frame.project(meta.lon.values, meta.lat.values)
    meta["fold"] = np.where(meta.station_id.isin(f2["a"]), "a",
                            np.where(meta.station_id.isin(f2["b"]), "b", "–"))
    test = ds.time.to_index().year.isin([2020, 2021])
    nsr = ds["nsrdb_ghi"].isel(time=np.nonzero(test)[0])
    corr = ds["nsrdb_ghi_corr_2a"].isel(time=np.nonzero(test)[0])
    dmean = (corr - nsr).mean("time").values
    tday = int(np.nonzero(ds.time.to_index() == pd.Timestamp(CASE_DAY))[0][0])
    dday = (ds["nsrdb_ghi_corr_2a"].isel(time=tday) - ds["nsrdb_ghi"].isel(time=tday)).values

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.78], width_ratios=[1.15, 1, 1],
                          wspace=0.1, hspace=0.3)
    # (a) station map
    a = fig.add_subplot(gs[0, 0])
    frame.show(a, np.where(frame.valid, 0.0, np.nan), cmap=mcolors.ListedColormap(["#f6f5f2"]))
    for net, mk, sz in (("ok_mesonet", "o", 16), ("uscrn", "s", 24)):
        for fold, col in (("a", C_SWIN), ("b", C_XGB)):
            m = meta[(meta.network == net) & (meta.fold == fold)]
            a.scatter(m.px, m.py, s=sz, marker=mk, color=col, edgecolor="white", linewidth=0.5,
                      zorder=6)
        m = meta[(meta.network == net) & (meta.fold == "–")]
        a.scatter(m.px, m.py, s=sz, marker=mk, facecolor="white", edgecolor=C_MUTED,
                  linewidth=0.7, zorder=6)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="s", ls="", color=C_SWIN, label="fold a"),
         Line2D([], [], marker="s", ls="", color=C_XGB, label="fold b"),
         Line2D([], [], marker="o", ls="", color=C_MUTED, ms=4, label="OK Mesonet"),
         Line2D([], [], marker="s", ls="", color=C_MUTED, ms=4.5, label="USCRN"),
         Line2D([], [], marker="s", ls="", mfc="white", mec=C_MUTED, ms=4.5, label="not in folds")]
    a.legend(handles=h, loc="lower left", fontsize=5.4, frameon=True, facecolor="white",
             edgecolor="none", framealpha=0.9, handletextpad=0.2, borderaxespad=0.3)
    a.set_title(r"$\bf{a}$  46 stations, two latitude-interleaved folds", loc="left", fontsize=7)

    # (b) correction, one day; (c) mean correction, test years
    vm = float(np.ceil(np.nanpercentile(np.abs(dday[frame.valid]), 98) * 10) / 10)
    for ax_i, (fld, ttl, vmax) in enumerate(((dday, f"correction, {pd.Timestamp(CASE_DAY):%d %b %Y}", vm),
                                             (dmean, "mean correction, 2020–21",
                                              float(np.ceil(np.nanpercentile(np.abs(dmean[frame.valid]), 99) * 10) / 10)))):
        ax = fig.add_subplot(gs[0, ax_i + 1])
        im = frame.show(ax, fld, cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -vmax, vmax))
        ax.scatter(meta.px, meta.py, s=4, color=C_INK, zorder=6, linewidth=0)
        ax.set_title(r"$\bf{" + "bc"[ax_i] + "}$  " + ttl, loc="left", fontsize=7)
        cax = ax.inset_axes([0.08, -0.08, 0.84, 0.04])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
        cb.set_label(f"anchored − NSRDB ({UNITS})", fontsize=5.6, labelpad=1)

    # (d) flow
    d = fig.add_subplot(gs[1, :]); canvas(d, 120, 32)
    panel(d, "d", x=0.0, y=0.98)
    box(d, 1, 14, 18, 13, "station-days\nobs vs NSRDB at\nthe station pixel", fc=T_AQUA, fs=5.6)
    box(d, 23, 14, 20, 13, "error index\n" + r"d = (obs − NSRDB) / N", fc=T_AQUA, fs=5.6)
    box(d, 47, 14, 24, 13, "XGBoost error model\nTrack B features + NSRDB\nindex · depth 4 · 400 trees\n"
        "fit on the OTHER fold", fc=T_XGB, fs=5.6)
    box(d, 75, 14, 20, 13, "anchored target\n" + r"NSRDB + $\hat{d}$ · N" + "\non every cell, every day",
        fc=T_TARGET, fs=5.6)
    box(d, 99, 14, 20, 13, "train downscaler\nscore at the fold's\nown stations only", fc=T_SWIN, fs=5.6)
    for x0, x1 in ((19, 23), (43, 47), (71, 75), (95, 99)):
        arrow(d, (x0, 20.5), (x1, 20.5))
    d.text(1, 8, "Fold a's target is learned without fold a's stations, so its model is scored there "
           "honestly; stations used by neither model's error fit get the mean of the two folds. "
           "v1 used the 7 USCRN stations only;\nv2 uses all 46 (Austin 33 NW forced into fold a, "
           "the holdout's only station).", fontsize=5.6, color=C_MUTED, va="top", linespacing=1.3)
    save(fig, out, "m5_anchoring")


# ------------------------------------------------------------------ M6
def m6_design(frame: MapFrame, out: Path) -> None:
    from training.train_swin import crop_origins

    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 0.9, 1], height_ratios=[1, 1],
                          wspace=0.3, hspace=0.55)
    # (a) timeline
    a = fig.add_subplot(gs[0, 0])
    rows = [("Main window", [(2001, 2019, "train", T_SWIN), (2019, 2020, "dev", T_GREY),
                             (2020, 2022, "test", T_TARGET)]),
            ("GOES window", [(2018, 2019.5, "train", T_SWIN), (2019.5, 2020, "dev", T_GREY),
                             (2020, 2022, "test", T_TARGET)])]
    for j, (lab, segs) in enumerate(rows):
        y = 1 - j
        for x0, x1, t, fc in segs:
            a.add_patch(Rectangle((x0, y - 0.32), x1 - x0, 0.64, fc=fc, ec="white", lw=1))
            a.text((x0 + x1) / 2, y, t, fontsize=5.2, ha="center", va="center",
                   rotation=0 if x1 - x0 >= 2 else 90)
        a.text(2000.7, y, lab, fontsize=6.3, ha="right", va="center")
    for yr, txt, ha in ((2018, "GOES-16 era (NSRDB satellite change) →", "right"),):
        a.axvline(yr, color=C_MUTED, lw=0.6, ls=":")
        a.text(yr + (0.15 if ha == "left" else -0.15), 1.55, txt, fontsize=5.2, color=C_MUTED,
               va="bottom", ha=ha)
    a.set_xlim(2001, 2022); a.set_ylim(-0.5, 1.8)
    a.set_yticks([]); a.spines["left"].set_visible(False)
    a.set_xticks([2001, 2005, 2009, 2013, 2018, 2022])
    a.set_title(r"$\bf{a}$  Time split", loc="left", fontsize=7.2, pad=10)

    # (b) test-cell matrix
    b = fig.add_subplot(gs[1, 0]); canvas(b, 100, 40)
    b.text(0, 38, r"$\bf{b}$  Test cells", fontsize=7.2, va="top")
    cells = [(30, 16, "temporal\ntest years ×\ntraining region", T_TARGET),
             (65, 16, "both (headline)\ntest years ×\nheld-out pixels", "#f2d7b0"),
             (65, 1, "spatial\ntraining years ×\nheld-out pixels", T_SWIN),
             (30, 1, "training data\n(exclusion zone\nnever trained on)", T_GREY)]
    for x, y, t, fc in cells:
        box(b, x, y, 33, 14, t, fc=fc, fs=5.4)
    b.text(46.5, 31.5, "training region", fontsize=5.6, ha="center", color=C_MUTED)
    b.text(81.5, 31.5, "Austin holdout", fontsize=5.6, ha="center", color=C_MUTED)
    b.text(28, 23, "test\nyears", fontsize=5.6, ha="right", va="center", color=C_MUTED)
    b.text(28, 8, "train\nyears", fontsize=5.6, ha="right", va="center", color=C_MUTED)

    # (c) holdout + buffer + crops
    c = fig.add_subplot(gs[:, 1])
    frame.show(c, np.where(frame.valid, np.where(frame.exclusion, 1.0, 0.0), np.nan),
               cmap=mcolors.ListedColormap(["#f3f2ee", "#e3c58a"]), vmin=0, vmax=1)
    org = crop_origins(frame.exclusion, 32)
    rng = np.random.default_rng(3)
    dx = abs(frame.x[1] - frame.x[0])
    for r0, c0 in org[rng.choice(len(org), 6, replace=False)]:
        x0, y0 = frame.x[c0] - dx / 2, frame.y[r0 + 31] - dx / 2
        c.add_patch(Rectangle((x0, y0), 32 * dx, 32 * dx, fill=False, ec=C_SWIN, lw=0.8, zorder=7))
    c.set_title(r"$\bf{c}$  Holdout, buffer, crops", loc="left", fontsize=7.2)
    c.text(0.02, -0.04, f"{len(org):,} valid 32×32 crop origins\n(blue: 6 examples); buffer = 1° "
           "(one\nPOWER cell) so no input leaks\nfrom the holdout", transform=c.transAxes,
           fontsize=5.4, color=C_MUTED, va="top")

    # (d) metric glossary
    d = fig.add_subplot(gs[:, 2]); canvas(d, 100, 100)
    d.text(0, 99, r"$\bf{d}$  Scores", fontsize=7.2, va="top")
    items = [("Grid, vs NSRDB", T_TARGET, ["RMSE, bias (" + UNITS + ")",
                                            "SSIM on the daily field",
                                            "scale bands <30 … 240 km:",
                                            "  placement = corr. of detail",
                                            "  amplitude = sd ratio",
                                            "CRPS, rank histogram (ensembles)"]),
             ("Stations, vs ground", T_AQUA, ["skill = 1 − MSE/MSE(B1),",
                                              "  station mean error removed",
                                              "bias; 90% CI: week-block",
                                              "  bootstrap (500 draws)",
                                              "mimicry = corr(model err,",
                                              "  NSRDB err) at a station"])]
    y = 90
    for head, fc, lines in items:
        box(d, 0, y - 6, 100, 6, head, fc=fc, fs=6)
        for j, ln in enumerate(lines):
            d.text(2, y - 10 - j * 6.2, ln, fontsize=5.6, va="center")
        y -= 50
    save(fig, out, "m1_design")


def main() -> None:
    style()
    out = RUNS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(ImportError):
        import xgboost  # noqa: F401
    from data.build_store import open_store
    from training.common import load_all

    dcfg, _ = load_all()
    frame = MapFrame(open_store(dcfg))
    m1_pipeline(out)
    m3_mean_models(out)
    m6_design(frame, out)
    m5_anchoring(frame, out)
    m4_diffusion(frame, out)
    m2_residual(frame, out)


if __name__ == "__main__":
    main()

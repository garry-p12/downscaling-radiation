// Builds the research-style radiation downscaling report as .docx.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, LevelFormat,
  Footer, Header, PageNumber, TableOfContents, PageBreak, TabStopType,
} = require("docx");

const FIG = "/Users/guruprasadparasnis/downscaling-radiation/runs/figures";
const OUT = process.argv[2];
const TEXT_W = 9360; // US Letter, 1" margins
const FONT = "Cambria";
const UNITS = "kWh m⁻² day⁻¹";

// ---------- inline markup: **bold**, *italic*, _{sub}, ^{sup} ----------
function runs(s, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let i = 0, m;
  while ((m = re.exec(s))) {
    if (m.index > i) out.push(new TextRun({ text: s.slice(i, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), superScript: true, ...base }));
    i = m.index + t.length;
  }
  if (i < s.length) out.push(new TextRun({ text: s.slice(i), ...base }));
  return out;
}

const body = [];
const P = (s, o = {}) => body.push(new Paragraph({
  children: runs(s), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, ...o,
}));
const H1 = (s) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(s)] }));
const H2 = (s) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(s)] }));
const H3 = (s) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(s)] }));
const B = (s) => body.push(new Paragraph({
  children: runs(s), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 276 },
}));
let eqNo = 0;
const EQ = (s) => {
  eqNo += 1;
  body.push(new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 80, after: 120 },
    children: [new TextRun("\t"), ...runs(s, { italics: false }), new TextRun(`\t(${eqNo})`)],
  }));
  return eqNo;
};

let figNo = 0;
const FIGS = {};
function figure(key, file, caption, widthIn = 6.5) {
  figNo += 1;
  FIGS[key] = figNo;
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthIn * 96, hpx = Math.round(wpx * h / w);
  body.push(new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
      altText: { title: `Figure ${figNo}`, description: caption.slice(0, 200), name: file } })],
  }));
  body.push(new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 },
    children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18 }), ...runs(caption, { size: 18 })],
  }));
}

let tabNo = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(caption, header, rows, widths, opts = {}) {
  tabNo += 1;
  const scale = TEXT_W / widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.round(x * scale));
  w[w.length - 1] += TEXT_W - w.reduce((a, b) => a + b, 0);
  body.push(new Paragraph({
    keepNext: true, spacing: { before: 160, after: 80 },
    children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, size: 18 }), ...runs(caption, { size: 18 })],
  }));
  const bold = new Set(opts.bold || []);
  const cell = (txt, j, isHead, isBold, last) => new TableCell({
    width: { size: w[j], type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "E8ECF2" } : undefined,
    borders: { top: isHead ? border : none, bottom: (isHead || last) ? border : none, left: none, right: none },
    children: [new Paragraph({
      alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: 17, bold: isHead || isBold }),
    })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((h, j) => cell(h, j, true, false, false)) })];
  rows.forEach((r, i) => trs.push(new TableRow({
    cantSplit: true, children: r.map((c, j) => cell(c, j, false, bold.has(i), i === rows.length - 1)),
  })));
  body.push(new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: w, rows: trs }));
  if (opts.note) body.push(new Paragraph({ spacing: { before: 60, after: 200 }, children: runs(opts.note, { size: 16, italics: true }) }));
  else body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return tabNo;
}

// ===================================================================== title
const title = [];
title.push(new Paragraph({ spacing: { before: 1800, after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Downscaling Daily Surface Solar Radiation from NASA POWER to 10 km over Texas and Oklahoma", bold: true, size: 36 })] }));
title.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "Learned residual models, cloud optical depth and station-anchored targets, validated against two independent ground networks", italics: true, size: 24 })] }));
title.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480, after: 60 },
  children: [new TextRun({ text: "Guruprasad Parasnis", size: 24 })] }));
title.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
  children: [new TextRun({ text: "Technical report · 30 October 2026", size: 22, color: "555555" })] }));
title.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Abstract", bold: true, size: 24 })] }));
const abstract = [
  "We downscale daily all-sky global horizontal irradiance (GHI) from NASA POWER (1°, CERES SYN1deg) to a 10 km grid over 27–35° N, 104–94° W. Models predict the residual of a clearness index relative to bilinearly interpolated POWER and are trained against the NSRDB GOES product, with an Austin spatial holdout and the 2020–21 test years withheld. Gradient-boosted trees (XGBoost) and a Swin U-Net with MODIS cloud fraction cut holdout × test-year RMSE against NSRDB from 0.478 to 0.397 " + UNITS + ", but at seven USCRN stations they add no day-to-day skill over interpolation: a target-swap experiment with ERA5-Land shows that each model inherits its training target's station errors.",
  "Two changes lift skill at the ground. First, GOES-16 cloud optical depth, converted to a two-stream cloud transmittance, places cloud edges that MODIS and the binary GOES mask cannot, roughly tripling sub-60 km placement. Second, a station-anchored target—NSRDB corrected by a gradient-boosted model of its own station error, fitted in two folds so that no model is scored at a station its target saw—removes most of NSRDB's error. On the anchored target fitted to both networks (v2), Swin B + optical depth reaches a station RMSE of 0.323 " + UNITS + " (NSRDB 0.441) and +45% day-to-day skill over bilinear POWER at 39 Oklahoma Mesonet stations (NSRDB +28%), +31% at USCRN (NSRDB +9%), and improves on interpolation at all 46 stations. XGBoost on the same inputs is statistically tied. The two networks disagree on absolute level by 0.2–0.3 " + UNITS + ", which one correction cannot reconcile. A CorrDiff-style residual diffusion model gives a 10% lower CRPS than a Gaussian spread of matched width but its fine-scale detail is not co-located with NSRDB's.",,
  "The same models were rebuilt on a 5 km grid and on a 1 km subdomain. At 5 km Swin gains another 1.5 points of station skill (RMSE 0.318 " + UNITS + ", +44% at all 46 stations); at 1 km every downscaling model loses ground against 5 km, because neither the target nor the inputs resolve anything below about 4 km. Finally, 90% intervals built from the spread of the anchored models and calibrated across the two test years cover 90% of Mesonet station-days, separate days with twice the typical error, and are narrowest at 5 km; Monte Carlo drop-path sampling of a single network adds no information. A green/amber/red uncertainty index and a low-end scenario are provided for the Austin area."
];
abstract.forEach((s) => title.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, children: runs(s, { size: 21 }) })));
title.push(new Paragraph({ spacing: { before: 120 }, children: [new TextRun({ text: "Keywords: ", bold: true, size: 20 }), new TextRun({ text: "solar radiation; statistical downscaling; NSRDB; NASA POWER; GOES-16 cloud optical depth; Swin Transformer; XGBoost; ground-station validation; diffusion models; grid resolution; uncertainty quantification; conformal prediction", size: 20 })] }));
title.push(new Paragraph({ children: [new PageBreak()] }));
title.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Contents", bold: true, size: 28 })] }));
title.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
title.push(new Paragraph({ children: [new PageBreak()] }));

// ===================================================================== 1 intro
H1("1. Introduction");
P("Gridded daily solar radiation feeds crop, hydrology, energy and heat-stress models, yet the longest consistent global record available at no cost, NASA POWER, is delivered on a 1° grid (roughly 100 km at these latitudes). Within a 1° cell, daily GHI varies with cloud edges, terrain and coastline by more than the cell mean's own error. Satellite products such as the National Solar Radiation Database (NSRDB) resolve this variability at 4 km but cover a shorter period, depend on a single geostationary sensor and are not available for every application.");
P("This report documents a downscaling system that maps POWER daily GHI onto a 10 km EPSG:5070 grid shared with a companion temperature-downscaling project, and onto 5 km and 1 km grids to test which grid spacing users should adopt. It has three aims: (i) restore physically consistent spatial structure, in particular fine-scale texture that is *in the right place* rather than merely of the right amplitude; (ii) generalise to a region and to years never seen in training; and (iii) demonstrate value at the ground, measured against stations that are never model inputs, so that the models do more than reproduce their training target.");
P("The first round of experiments reused the temperature project's recipe—residual learning on top of interpolation, Swin U-Net and gradient-boosted baselines, texture-oriented losses. It improved every grid metric against NSRDB but not station skill. Section 5 traces this to the target itself. The second round therefore changed the inputs (GOES-16 cloud optical depth), the target (station anchoring) and the output type (a residual diffusion ensemble), and evaluated each change at 46 stations from two independent networks.");
P("**Contributions.** (1) A leakage-controlled benchmark with temporal, spatial and joint holdouts and two-network station scoring. (2) Evidence, via a target swap, that downscalers inherit their target's errors at the ground. (3) Cloud optical depth as the first input that takes a model beyond NSRDB at stations. (4) A cross-validated station-anchored target that transfers between networks in day-to-day skill but not in absolute level. (5) A calibrated-but-narrow residual diffusion ensemble. (6) A paired comparison of the same models on 10, 5 and 1 km grids. (7) Station-calibrated 90% intervals, an uncertainty index and worst-case scenarios for the recommended model.");

// ===================================================================== 2 data
H1("2. Study domain and data");
H2("2.1 Domain and grid");
P("The domain spans 27–35° N and 104–94° W (Texas, Oklahoma and adjacent states), projected to a 10 km Albers equal-area (EPSG:5070) grid of 93 × 100 cells. It contains strong west–east moisture and cloud gradients, the Edwards Plateau and Trans-Pecos terrain, a Gulf coastline and the Austin urban area. The Austin box (29.7–31.0° N, 98.6–97.0° W) plus a 1° buffer is a spatial holdout that no training loss, crop or fitted statistic ever touches (Figure 1).");
figure("domain", "fig1_domain.png", "The domain on the 10 km grid, shaded by elevation, with POWER's 1° cells, the Austin holdout (solid), its 1° buffer (dashed) and the seven USCRN stations.", 4.6);

H2("2.2 Data sources");
table("Data sources and processing. All sources are converted to local-solar-day totals on the 10 km grid.",
  ["Source", "Variables used", "Processing"],
  [
    ["NASA POWER daily regional API (v10)", "All-sky and clear-sky surface SW, TOA, diffuse, precipitable water, AOD, cloud amount; 1°", "CERES SYN1deg from 2001-01-01 (SRB before, not used). Bilinear and enclosing-cell mappings to the 10 km grid"],
    ["NSRDB GOES aggregated v4.0.0 (target)", "GHI, 4 km, 30 min", "Read from S3; 100% gap-filled samples dropped; local-solar-day totals (UTC + lon/15); days >10% daylight missing rejected; area-averaged to 10 km"],
    ["MODIS MOD09GA / MYD09GA v061", "state_1km cloud bits, Terra (~10:30) and Aqua (~13:30)", "Via AppEEARS; 1 km cloudy fraction per 10 km cell, per satellite and combined"],
    ["GOES-16 ABI L2 ACMC", "Binary clear-sky mask, 2 km", "Two scans per hour, binned to 10 km; sunlight-weighted (cos z) daily mean plus morning and afternoon halves; 2018 on"],
    ["GOES-16 ABI L2 CODC", "Cloud optical depth τ, 4 km, daytime", "Clear pixels set to τ = 0; per-pixel transmittance (Eq. 4); binned and sunlight-weighted like the mask; 2018 on"],
    ["Static fields", "1 km DEM, land cover, albedo, coastline", "Elevation, sub-grid anomaly, slope, aspect, sky-view factor, roughness, horizon shading by day of year, land-cover fractions, distance to coast"],
    ["USCRN hourly02", "Hourly GHI, 7 in-domain stations, 2003–2021", "BSRN physically-possible and comparison limits; local-solar-day totals; never an input"],
    ["Oklahoma Mesonet", "Daily ATOT, 39 in-domain stations, 2001–2021", "Obtained through the Mesonet data request service; same day convention; never an input"],
    ["ERA5-Land; GOES-16 DSR", "Hourly surface SW at 0.1° and 0.25°", "Reference products scored at stations; ERA5-Land also used in the target-swap experiment"],
  ], [2.2, 2.6, 4.2]);

H2("2.3 Temporal splits");
P("Two windows are used. The **main window** trains on 2001–2018, selects models on 2019 (dev) and tests on 2020–21. The **GOES window**, needed for inputs that start in 2018, trains on January 2018–June 2019, uses July–December 2019 as dev and tests on 2020–21. Test results from different windows are compared only within a window, because the GOES window has 12 times less training data and its own climatology baseline.");

// ===================================================================== 3 methods
H1("3. Methods");
figure("design", "m1_design.png", "Experimental design. (a) Train, dev and test years for the main (2001–2021) and GOES (2018–2021) windows. (b) The three test cells; *both* (unseen place and unseen years) is the headline number. (c) The Austin holdout (solid) and its one-POWER-cell buffer (shaded): no 32 × 32 training crop touches either, so no input reaches the model from the holdout. (d) What each score measures.");
figure("pipeline", "m2_pipeline.png", "End-to-end pipeline. Every source is regridded to one 10 km EPSG:5070 grid as local-solar-day totals; models predict a residual clearness index on top of bilinear POWER; products are scored against NSRDB on the grid and, independently, against ground stations that never enter a model.");

H2("3.1 Problem formulation");
P("Let *N* be a daily normaliser on the fine grid. The default is the analytic daily top-of-atmosphere irradiance (TOA), which makes *k* the clearness index *k*_{t}:");
EQ("k = GHI / N");
P("Models predict the residual between the target's index and bilinearly interpolated POWER, B(·):");
EQ("r = k_{target} − B(k_{POWER})");
P("and GHI is reconstructed as");
EQ("GHI^ = clip( B(k_{POWER}) + r^ ) · N");
P("with *k*_{t} clipped to [0, 0.85] and to 1.10 × POWER's clear-sky *k*_{t}. In the clear-sky variant (method B), *N* is a fine clear-sky GHI, TOA × B(POWER clear-sky / TOA), so *k* becomes the clear-sky index *k*^{*} and is clipped to [0, 1.10]. Predicting a bounded, nearly season-free index means the network does not have to relearn solar geometry, and a zero prediction reproduces the interpolation baseline exactly (Figure " + (figNo + 1) + ").");
figure("residual", "m3_residual.png", "The residual formulation on 23 April 2021. POWER (a) is divided by the top-of-atmosphere normaliser (e) and interpolated bilinearly (b); the target residual (d) is NSRDB's index (c) minus (b). The model predicts (f), and GHI is rebuilt as (b + f) · e (g); (h) is the reconstruction error against NSRDB.");

H2("3.2 Baselines");
B("**B0:** bilinear POWER GHI.");
B("**B1:** bilinear POWER *k*_{t} × fine TOA. This is the reference for all skill scores.");
B("**B2:** B1 split into beam and diffuse components, with beam × daily unblocked-horizon fraction and diffuse × sky-view factor.");
B("**B3:** B1 × a training-only climatological ratio of fine NSRDB *k*_{t} to POWER *k*_{t}, by day of year.");

H2("3.3 Feature tracks");
table("Input feature tracks. Each track adds to the previous one; modifiers can be combined with any track.",
  ["Track", "Inputs"],
  [
    ["A", "POWER *k*_{t} (bilinear, enclosing cell, anomaly against its day-of-year climatology, 3- and 7-day means, clear-sky ratio); terrain; land cover; albedo; coast distance; training-only NSRDB *k*_{t} climatology; day of year; latitude and longitude"],
    ["B", "A + MODIS Terra and Aqua cloud fraction + POWER precipitable water, AOD and cloud amount"],
    ["C", "B + GOES-16 cloud-mask fraction (daily, morning, afternoon), 48 scans per day. Flagged: NSRDB is itself built from GOES"],
    ["+cod", "GOES-16 cloud transmittance from optical depth (daily, morning, afternoon)"],
    ["+T", "*k*_{t} and cloud fields of the previous and next day (method D)"],
  ], [1, 8]);
P("Cloud optical depth τ enters through a two-stream transmittance that maps it to a bounded, approximately linear proxy for cloud attenuation:");
EQ("T = 1 / [ 1 + 0.75 (1 − g) τ ],   g = 0.85");
P("computed per 4 km pixel before averaging, so that a partly cloudy 10 km cell is not represented by the transmittance of its mean optical depth.");

H2("3.4 Mean models");
P("**XGBoost (pixel-day table).** Each sample is one pixel on one day. Three million training rows are drawn stratified by POWER sky class (bilinear *k*_{t}). Trees have depth 8 and learning rate 0.05, with up to 2,000 rounds and early stopping after 50 rounds without dev improvement. Loss weighting (none, TOA or energy) is selected on dev RMSE.");
P("**Swin U-Net (field).** The encoder follows the temperature project's backbone but is shallower, so that a 32 × 32 crop fits outside the holdout buffer: patch size 2, embedding 96, stage depths [2, 2, 6], heads [3, 6, 12], window 4 and stochastic depth 0.1. The decoder mirrors the encoder with skip connections; the head is conv 3×3 → GELU → conv 3×3 → PixelShuffle, with the final convolution zero-initialised so that training starts exactly at B1. Training uses 32 × 32 crops, batch 16, 40 epochs and AdamW (learning rate 2 × 10^{−4}, weight decay 0.05). The loss is");
EQ("L = MSE(r^, r) + 0.2 · [ 1 − SSIM(k^_{t}, k_{t}) ]");
P("and the epoch with the lowest dev RMSE is kept. Inference pads the 93 × 100 domain to 96 × 128.");
figure("models", "m4_mean_models.png", "Mean models, drawn to the configured sizes. (a) XGBoost sees one pixel-day at a time and has no spatial context beyond what its features carry. (b) The Swin U-Net (window 4, patch 2, three stages of 96/192/384 channels); its last layer is zero-initialised, so training starts from B1. (c) Each stage alternates regular and shifted 4 × 4 attention windows.");

H2("3.5 Station anchoring");
P("The station-anchored target replaces NSRDB with NSRDB corrected by a learned model of its own station error. Let *d* be NSRDB's normalised error at a station:");
EQ("d = (obs − NSRDB) / N");
P("An XGBoost error model (depth 4, 400 trees) predicts *d* from the Track B features at the station pixel plus NSRDB's own index, using training-year station-days only. It is applied to every pixel and day to build the anchored target");
EQ("k_{anchored} · N = NSRDB + d^ · N");
P("Stations are split into two folds, and each fold's target is built without that fold's stations; models trained on fold *a*'s target are scored only at fold *a*'s stations and vice versa. The **v1** target uses the seven USCRN stations (fold a: Austin 33 NW, Monahans, Palestine, Port Aransas; fold b: Bronte, Muleshoe, Panther Junction). The **v2** target uses all 46 stations from both networks in two latitude-interleaved folds, with Austin in fold a. Three New Mexico USCRN stations lie inside the grid but outside the folds and are not scored.");
P("As an alternative, method A1 merges stations at run time: each day, ordinary kriging of the other stations' residuals is evaluated at a withheld station, with an exponential correlation whose length is fitted on training years, and scored leave-one-station-out.");
figure("anchoring", "m5_anchoring.png", "Station anchoring. (a) The 46 stations in two latitude-interleaved folds. (b) The fold-a correction (anchored − NSRDB) on 23 April 2021 and (c) its mean over the test years: the error model lowers NSRDB almost everywhere, most over cloudy Oklahoma. (d) How a fold's target is built and scored.");

H2("3.6 Residual diffusion");
P("To represent the variability that a mean model cannot place, a CorrDiff-style model learns the distribution of the residual left by the frozen Swin B + MODIS mean, *e* = *k*_{NSRDB} − *k*^_{Swin}. An EDM-preconditioned U-Net (base width 64, channel multipliers (1, 2, 4), two residual blocks per level, FiLM conditioning on Fourier features of ln σ / 4) denoises the scaled residual conditioned on the Swin inputs and the Swin prediction. Noise levels are drawn as ln σ ~ N(−1.2, 1.2²). Training runs 60,000 steps with batch 64 on 32 × 32 crops and an exponential moving average of the weights (decay 0.999); the checkpoint is chosen by dev CRPS. Sampling integrates the probability-flow ODE with 18 Heun steps from independent noise to produce 16-member ensembles.");
figure("diffusion_arch", "m6_diffusion.png", "Residual diffusion. (a) Training: the U-Net learns to denoise the scaled residual between NSRDB and the frozen Swin mean, conditioned on the Swin inputs and its prediction; sampling runs 18 Heun steps from noise. (b) 23 April 2021: the residual the mean misses (left), three independent draws and the mean of eight. Draws carry texture of the right amount but in different places; their mean flattens toward zero.");

H2("3.7 Evaluation");
P("**Against NSRDB, by test cell.** *Temporal* = test years × training region; *spatial* = training years × holdout; *both* = test years × holdout (headline). Metrics are RMSE, bias, SSIM, residual correlation with B1, and per-band placement and amplitude. Bands are difference-of-Gaussian filters at < 30, 30–60, 60–120 and 120–240 km; *placement* is the correlation between a model's band-passed field and NSRDB's, and *amplitude* is the ratio of their band standard deviations. Spectral coherence is reported as a function of wavelength.");
P("**At stations.** For a product *m* at a station, let MSE_{c}(*m*) be the mean squared error after removing that product's mean error at that station. Day-to-day skill is");
EQ("S = 1 − MSE_{c}(m) / MSE_{c}(B1)");
P("pooled over stations, with 90% intervals from a week-block bootstrap (500 draws) that preserves synoptic autocorrelation. Removing each product's station mean separates day-to-day skill from calibration, which is reported separately as bias. *Mimicry* is the correlation between a model's station errors and a reference product's station errors.");
P("**Probabilistic.** Ensembles are scored with the fair continuous ranked probability score (CRPS), the spread–skill ratio and rank histograms, both against NSRDB on the grid and at stations.");
P("**Intervals.** For the deterministic models, 90% intervals are centre ± *q*·*s*, where *s* is a spread estimate and *q* the 90th percentile of |error| / *s* at the stations in one test year (split conformal calibration); each interval is scored in the other year. They are scored by coverage, mean width, the interval score (width plus 20 times the amount by which the observation falls outside) and the rank correlation of width with absolute error (Section 4.14).");
P("**Grid resolution.** Models on different grids are compared on the station-days all grids share, each with skill relative to B1 on its own grid; the paired gain resamples the same weeks for both grids.");

H2("3.8 Leakage controls");
B("Every statistic that touches the target (NSRDB *k*_{t} climatology, the B3 ratio map, input normalisers) is fitted on training years × training pixels only and filled into the holdout by Gaussian normalised convolution (σ = 40 km).");
B("Swin training crops are drawn only where the whole 32 × 32 crop avoids the holdout and its buffer (1,173 valid crop origins).");
B("Stations are never inputs. Anchored targets are built without the stations they are scored on.");
B("Model selection uses dev years only; test years are evaluated once per configuration.");

H2("3.9 Computation");
P("Downloads run on the TACC Vista login node. Training runs on Vista GH200 nodes as resumable jobs of 35 minutes or less, which fit backfill gaps; a driver resubmits each task until its completion marker appears. XGBoost and evaluation also run on an Apple M3 laptop. When the main GPU queue was congested, the 5 and 1 km experiments ran as sequential two-hour jobs on the development queue, each resuming from completion markers; a 1 km evaluation, which cannot resume, is skipped when too little time remains in a job.");

// ===================================================================== 4 results
H1("4. Results");
H2("4.1 Accuracy against NSRDB");
P("Every learned model beats bilinear POWER (B1) and the climatology baseline (B3) in all three test cells (Table " + (tabNo + 1) + "). MODIS cloud fraction (Track B) gives the largest gain in the main window; the GOES cloud mask (Track C) does so in its own window. Reframing GHI as a clearness index alone (B1 vs B0) is a tie on daily GHI.");
table("RMSE against NSRDB (" + UNITS + ") by test cell, and skill relative to B3 in the *both* cell.",
  ["Model", "Window", "Temporal", "Spatial", "Both (headline)", "Skill vs B3"],
  [
    ["B1: bilinear *k*_{t} × TOA", "–", "0.426", "0.394", "0.478", "−0.07"],
    ["B3: climatology ratio", "main", "0.402", "0.380", "0.462", "0"],
    ["Swin A", "main", "0.367", "0.310", "0.405", "+0.23"],
    ["Swin B + MODIS", "main", "0.343", "0.307", "0.397", "+0.26"],
    ["XGBoost B + MODIS", "main", "0.362", "0.292", "0.397", "+0.26"],
    ["Swin C (GOES mask)", "GOES", "0.318", "0.322", "0.360", "+0.50"],
    ["XGBoost C (GOES mask)", "GOES", "0.327", "0.320", "0.358", "+0.51"],
  ], [3, 1, 1.1, 1.1, 1.4, 1.2], { bold: [3], note: "GOES-window rows have a shorter training period and their own B3; compare them only with each other." });
figure("accuracy", "fig5_accuracy_structure.png", "(a) RMSE against NSRDB in each test cell, main window. (b) Holdout × test-year RMSE against SSIM for every model; hollow markers are the GOES window. Up and left is better: cloud inputs move models there, while the AMSE loss moves them down and right.");

H2("4.2 Case days and error maps");
P("Two test days were picked by a fixed rule rather than by eye: the largest spatial spread of NSRDB GHI in each half-year, among days with under 5% missing NSRDB and under 20% missing MODIS. On 20 March 2020 (Figure " + (figNo + 1) + ") the MODIS cloud edge places a frontal boundary that bilinear POWER smears; both learned models cut domain RMSE from 0.50 (B1) to 0.35. On 23 April 2021 (Figure " + (figNo + 2) + ") they remove POWER's blocky 1° error pattern and do best in the holdout (Swin 0.38 vs B1 0.45), but are slightly worse than B1 over the whole domain (0.35–0.36 vs 0.32). Single days vary, so Figure " + (figNo + 3) + " gives the two-year picture.");
figure("case1", "fig2_case_cool.png", "20 March 2020. Top: daily GHI from NSRDB (target), POWER at its native 1°, B1, XGBoost B + MODIS and Swin B + MODIS. Bottom: MODIS cloud fraction (an input) and each estimate minus NSRDB, with RMSE over the domain and the holdout box.");
figure("case2", "fig3_case_warm.png", "23 April 2021; panels as in Figure " + figNo + ".");
figure("skillmaps", "fig4_skill_maps.png", "Per-pixel daily RMSE (top) and correlation (bottom) against NSRDB over 2020–21 for B1 and Swin B + MODIS, and their difference (blue = Swin better).");
P("Over all test days, Swin B + MODIS beats B1 at 99.8% of pixels, holdout included. The exceptions are in central Austin, inside the holdout, where Swin draws a small dark spot (Figures " + FIGS.case1 + "e, " + FIGS.case2 + "j and " + FIGS.skillmaps + "c). Those pixels are 65–93% developed land, above the domain's 99th percentile, so the model is extrapolating its land-cover response to a place it never trained on. Capping or dropping the developed-land feature is a cheap fix.");

H2("4.3 Structure and texture");
P("Texture in the right place comes from cloud inputs, not from losses. MODIS raises sub-30 km placement from 0.006 to 0.047 in the main window, and the GOES mask from 0.029 to 0.102 in the GOES window, while loss changes only add amplitude (Table " + (tabNo + 1) + ", Figure " + (figNo + 1) + ").");
table("Structure metrics in the holdout × test-year cell. Placement is the band-passed correlation with NSRDB.",
  ["Model", "SSIM", "Resid. corr. vs B1", "Placement < 30 km", "30–60 km", "60–120 km"],
  [
    ["B1", "0.375", "–", "0.007", "0.021", "0.124"],
    ["Swin A, main", "0.413", "0.324", "0.003", "0.019", "0.138"],
    ["Swin B + MODIS, main", "0.491", "0.449", "0.047", "0.067", "0.197"],
    ["Swin C (GOES mask), GOES window", "0.567", "0.585", "0.102", "0.142", "0.282"],
  ], [3, 1, 1.3, 1.3, 1, 1], { bold: [3] });
figure("bands", "fig6_scale_bands.png", "Holdout × test years. (a) Placement: correlation with NSRDB in each scale band. (b) Amplitude relative to NSRDB (1 = as much texture as the target). (c) Spectral coherence with NSRDB by wavelength. Cloud inputs raise placement at every scale; the AMSE loss raises amplitude below 60 km with no gain in placement.");
P("Loss-function experiments, in summary:");
B("**SSIM term (weight 0.2): kept as default.** In-region SSIM rises from 0.695 to 0.715 with RMSE unchanged; it costs 6–11% RMSE on the unseen holdout.");
B("**AMSE (spectral amplitude loss): rejected.** It adds texture without placement, so SSIM falls to 0.32 and residual correlation halves.");
B("**Post-hoc band-amplitude gains: rejected.** Every gain reached its clip and SSIM fell from 0.41 to 0.29.");
B("**Input ceilings.** MODIS (two overpasses per day) correlates 0.03–0.07 with NSRDB detail below 60 km and 0.11–0.31 above; the GOES mask (48 scans per day) 0.04–0.10 and 0.13–0.39. Even NSRDB's own satellite explains little of NSRDB's sub-60 km day-to-day texture.");

H2("4.4 Ground-station check against USCRN");
P("At the ground the first-round models are capped by their target. NSRDB beats POWER day to day by 9%, but no model trained on it adds significant skill over B1 (Figure " + (figNo + 1) + "). Every product, POWER included, reads high against USCRN by 0.35–0.51 " + UNITS + ", so absolute level cannot be settled from these data alone; the skill score therefore removes each station's mean error. Models follow NSRDB's errors more closely as they improve against NSRDB (Table " + (tabNo + 1) + "). Dropping the NSRDB-climatology feature (kt_clim_fine) changed station skill by about one point in both windows, so it is not the mechanism. At Austin, inside the holdout, NSRDB itself is 16% worse than B1 [−32, −3].");
figure("stations", "fig7_station_skill.png", "Day-to-day skill against B1 at USCRN stations, 2020–21, with 90% week-block bootstrap intervals (500 draws). (a) All seven stations, 4,723 station-days. (b) Austin 33 NW, inside the holdout. The grey area is worse than bilinear POWER.");
table("Bias and NSRDB mimicry at the seven USCRN stations, 2020–21.",
  ["Product or model", "Bias vs USCRN (" + UNITS + ")", "NSRDB mimicry"],
  [
    ["B1 (bilinear POWER)", "+0.35", "0.57"],
    ["NSRDB", "+0.48", "1.00"],
    ["Swin B + MODIS, main window", "+0.51", "0.70"],
    ["Swin C (GOES mask), GOES window", "+0.47", "0.75"],
  ], [4, 2.5, 2]);

H2("4.5 Target swap with ERA5-Land");
P("To test whether the target caps ground skill, the same Swin B + MODIS was trained on ERA5-Land instead of NSRDB, with identical inputs, splits, architecture and loss. It loses 46% day-to-day skill at the stations, and its errors swing from NSRDB's to ERA5-Land's (Table " + (tabNo + 1) + ", Figure " + (figNo + 1) + "). Against its own target the ERA5-Land model still improves on B1 (+31%; holdout RMSE 0.563 → 0.504), so a skill score measured against the training target cannot tell these two models apart. XGBoost A and B trained on ERA5-Land show the same pattern (−49% and −47% at the stations). Even the bias follows the target: +0.38 against ERA5-Land's +0.35, versus +0.51 when trained on NSRDB.");
table("Target swap at seven USCRN stations, 2020–21.",
  ["Product or model", "Skill vs B1 (90% CI)", "NSRDB mimicry", "ERA5-Land mimicry", "Bias"],
  [
    ["NSRDB, the product", "+9% (+1, +16)", "1.00", "0.27", "+0.48"],
    ["ERA5-Land, the product", "−128% (−146, −110)", "0.27", "1.00", "+0.35"],
    ["B1, bilinear POWER", "0", "0.57", "0.60", "+0.36"],
    ["Swin B + MODIS trained on NSRDB", "+1% (−4, +6)", "0.70", "0.41", "+0.51"],
    ["Swin B + MODIS trained on ERA5-Land", "−46% (−53, −39)", "0.42", "0.75", "+0.38"],
  ], [3.4, 2, 1.3, 1.5, 1]);
figure("swap", "fig8_target_swap.png", "(a) Each model's station-error correlation with NSRDB's errors (x) and with ERA5-Land's (y); arrows run from B1 to the same architecture trained on each target. (b, c) Daily Swin B + MODIS GHI against USCRN when trained on NSRDB and on ERA5-Land, seven stations, 2020–21; dashed line is 1:1.");

H2("4.6 Clear-sky index and neighbouring days");
P("Neither change moves ground skill (Table " + (tabNo + 1) + "). The clear-sky index gives Swin a small grid gain; neighbouring days give nothing.");
table("Normalisation and temporal-context variants, main window, holdout × test years.",
  ["Model", "RMSE vs NSRDB", "SSIM", "Station skill vs B1"],
  [
    ["Swin B + MODIS, *k*_{t} (production)", "0.397", "0.491", "+1%"],
    ["Swin B + MODIS, clear-sky index", "0.391", "0.507", "+1%"],
    ["Swin B + MODIS, + neighbouring days", "0.398", "0.497", "+2%"],
    ["XGBoost B + MODIS, *k*_{t}", "0.397", "0.471", "−2%"],
    ["XGBoost B + MODIS, clear-sky index", "0.398", "0.475", "−2%"],
    ["XGBoost B + MODIS, + neighbouring days", "0.397", "0.470", "−3%"],
  ], [4, 1.6, 1.2, 1.8], { bold: [1] });

H2("4.7 GOES cloud optical depth");
P("Optical depth is the first input that takes a model past NSRDB at the ground, including at Austin inside the holdout (Table " + (tabNo + 1) + "). Its station errors track NSRDB's more (mimicry 0.80 against 0.69 for Track B), yet they are smaller than NSRDB's own, so the model is not simply copying the target. NOAA's operational GOES radiation product (DSR) scores −82% at the same stations, so the gain comes from optical depth combined with POWER, not from GOES radiation as such.");
table("Effect of cloud optical depth, GOES window, XGBoost. Station skill at seven USCRN stations.",
  ["Model", "RMSE vs NSRDB", "SSIM", "Placement < 30 km", "Station skill (90% CI)", "Austin 33 NW"],
  [
    ["B (MODIS)", "0.387", "0.46", "0.05", "−8% (−12, −4)", "−13%"],
    ["C (GOES mask)", "0.358", "0.55", "0.11", "−10% (−15, −5)", "−9%"],
    ["B + optical depth", "0.318", "0.65", "0.20", "+14% (+8, +21)", "+10%"],
    ["C + optical depth", "0.305", "0.66", "0.21", "+12% (+6, +18)", "+7%"],
    ["NSRDB itself", "–", "–", "–", "+9% (+1, +16)", "−16%"],
  ], [2.4, 1.3, 0.9, 1.4, 1.8, 1.2], { bold: [2] });
P("Swin gains as much from optical depth as XGBoost on the test years (holdout × test RMSE 0.396 → 0.320 for B, 0.311 for C + optical depth; SSIM 0.67–0.68), but not more, and its station skill is similar (USCRN +11% for B, +14% for C). The gain is largest in the spatial holdout, where optical depth supplies cloud detail that interpolation of POWER cannot (Figures " + (figNo + 1) + "–" + (figNo + 2) + ").");
figure("codcase", "fig9_cod_case.png", "23 April 2021 (test year). Top: NSRDB and three models. Bottom: GOES optical-depth transmittance (white = thick cloud) and each model's error against NSRDB. Optical depth places the cloud-band edge and cuts holdout RMSE from 0.45 to 0.20.");
figure("codscores", "fig10_cod_scores.png", "(a) RMSE against NSRDB before (open circle) and after (arrow head) adding optical depth, per test cell. (b) Placement (correlation with NSRDB's detail) by scale band in the holdout; optical depth roughly triples placement below 60 km.");

H2("4.8 Anchoring to ground stations");
P("Learning NSRDB's station error removes most of it. Corrected with an error model fitted on the other six stations, NSRDB scores +31% day to day at each withheld USCRN station (RMSE 0.65 → 0.39, bias +0.48 → +0.07) and improves at all seven. Same-day kriging (A1) mostly removes bias: with seven stations about 300 km apart (fitted correlation length 316 km), day-to-day skill rises only from +9% to +16% for NSRDB and from +1% to +4% for Swin. Training on the anchored target carries the gain into the downscaler, which never sees NSRDB at run time (Table " + (tabNo + 1) + ").");
table("v1 anchored target (USCRN folds): skill at withheld USCRN stations, 2020–21.",
  ["Model", "Skill vs B1 (90% CI)", "RMSE", "Bias"],
  [
    ["NSRDB", "+9% (+1, +16)", "0.65", "+0.48"],
    ["Swin B + MODIS, NSRDB target", "+1%", "0.68", "+0.51"],
    ["Swin B + MODIS, anchored", "+16% (+13, +19)", "0.42", "+0.05"],
    ["XGBoost B + MODIS, anchored", "+7% (+3, +10)", "0.45", "+0.03"],
    ["XGBoost B + optical depth, NSRDB target", "+14% (+8, +21)", "0.65", "+0.49"],
    ["XGBoost B + optical depth, anchored", "+30% (+26, +35)", "0.39", "+0.03"],
    ["Swin B + optical depth, anchored", "+34% (+29, +39)", "0.37", "+0.02"],
  ], [4.2, 2.2, 1, 1], { bold: [6] });
P("With optical depth and the anchored target, every USCRN station beats B1: Austin +37%, Bronte +45%, Muleshoe +41%, Monahans +38%, Port Aransas +27%, Panther Junction +13% and Palestine +11% (XGBoost).");

H2("4.9 Two networks: Oklahoma Mesonet");
P("The v1 target was learned from seven USCRN stations, so it could have learned USCRN's calibration rather than NSRDB's error. Daily totals from 39 in-domain Oklahoma Mesonet stations (2001–2021; a different network, pyranometer type and maintenance regime) test this.");
B("**Day-to-day skill transfers.** Fitted on USCRN only, the error model lifts NSRDB at all 39 Mesonet stations from +27.5% to +44%. Fitted on Mesonet only, it lifts USCRN from +9% to +30%.");
B("**Absolute level does not.** NSRDB reads +0.19 " + UNITS + " high at Mesonet and +0.48 at USCRN. A USCRN-anchored target therefore lands about 0.2 low at Mesonet, and a Mesonet-anchored one stays about 0.3 high at USCRN. The networks' calibrations differ by more than the error model can reconcile; neither is wrong by construction.");
B("**v2 target.** The error model refitted on all 46 stations, in two latitude-interleaved folds.");
table("Station scores by network, test years, GOES window. Anchored models are scored at each station by the fold that never saw it.",
  ["Model", "Mesonet skill (39)", "Mesonet bias", "USCRN skill (7)", "USCRN bias", "RMSE, all 46"],
  [
    ["NSRDB", "+27.5% (+21, +34)", "+0.19", "+9% (+2, +16)", "+0.48", "0.441"],
    ["XGBoost B + opt. depth, NSRDB target", "+31%", "+0.20", "+14%", "+0.49", "0.437"],
    ["Swin C + opt. depth, NSRDB target", "+30%", "+0.16", "+14%", "+0.47", "0.420"],
    ["XGBoost B + opt. depth, anchored v1", "+42% (+37, +46)", "−0.21", "+30% (+26, +35)", "+0.03", "0.371"],
    ["Swin B + opt. depth, anchored v1", "+42% (+36, +46)", "−0.22", "+34% (+29, +39)", "+0.02", "0.373"],
    ["XGBoost B + opt. depth, anchored v2", "+44% (+38, +49)", "0.00", "+30% (+25, +35)", "+0.20", "0.327"],
    ["Swin B + opt. depth, anchored v2", "+45% (+40, +50)", "0.00", "+31% (+26, +36)", "+0.20", "0.323"],
    ["Swin B + MODIS, anchored v2 (main window)", "+27% (+23, +30)", "0.00", "+19% (+15, +22)", "+0.22", "0.367"],
  ], [3.3, 1.7, 1, 1.6, 1, 1.1], { bold: [6], note: "The last row is the main-window model for years without GOES (2001–2017), scored on the same 2020–21 station-days." });
figure("networks", "fig11_networks.png", "Day-to-day skill against B1 at Oklahoma Mesonet (a) and USCRN (b) stations, with 90% week-block bootstrap intervals, and mean bias per network (c). Anchored models are scored at each station by the fold whose correction never saw it (v1: corrections from seven USCRN stations; v2: from all 46 stations of both networks).");
P("Swin B + optical depth on the v2 target edges XGBoost on every pooled station score (RMSE 0.323 vs 0.327, Mesonet +45% vs +44%), but the intervals overlap almost fully: the two are tied at stations. Both improve on B1 at all 46 stations individually; the median station skill is +44% (Swin) and +43% (XGBoost) at Mesonet and +27% and +28% at USCRN, against +27% and +11% for NSRDB (Table " + (tabNo + 1) + "). Swin's advantage is spatial: against its own anchored target in the *both* cell it reaches SSIM 0.68 and residual correlation 0.74–0.75. The USCRN bias of +0.20 is shared by both v2 models; it comes from the v2 correction being fitted mostly to the 39 Mesonet stations, which pulls it toward Oklahoma's error pattern.");
const ps = fs.readFileSync(path.join(__dirname, "perstation.csv"), "utf8").trim().split("\n").slice(1).map((l) => l.split(","));
const usc = ps.filter((r) => r[2] === "uscrn").map((r) => [r[1].replace("TX_", "").replace(/_/g, " "), r[0], r[3], `${(+r[4]).toFixed(1)}`, `${(+r[5]).toFixed(1)}`, `${(+r[6]).toFixed(1)}`, (+r[7] >= 0 ? "+" : "") + (+r[7]).toFixed(2)]);
const meso = ps.filter((r) => r[2] === "ok_mesonet").map((r) => [r[1], r[0], r[3], `${(+r[4]).toFixed(1)}`, `${(+r[5]).toFixed(1)}`, `${(+r[6]).toFixed(1)}`, (+r[7] >= 0 ? "+" : "") + (+r[7]).toFixed(2)]);
table("Per-station day-to-day skill vs B1 (%) at the seven USCRN stations, v2 target, 2020–21. Bias is Swin's, in " + UNITS + ".",
  ["Station", "Fold", "Days", "NSRDB", "XGBoost v2", "Swin v2", "Swin bias"], usc, [3, 0.7, 0.8, 1, 1.2, 1.1, 1.1]);

H2("4.10 Spatial-leakage test of the anchoring");
P("The v2 folds interleave stations by latitude, so every withheld Mesonet station has a neighbour 11–52 km away in the other fold, and the error model could learn local errors from that neighbour rather than NSRDB's general error. To bound this, the error model was refitted leave-one-station-out with an exclusion buffer: for each held-out station, every station within *R* km was also removed from the fit (main window, training 2001–2018, scored on 2020–21). Because the downscalers follow their target closely (Section 4.5), the corrected-NSRDB score is an upper bound on what a model trained on that target can gain from neighbours.");
table("Corrected NSRDB at held-out stations as the exclusion buffer grows. Day-to-day skill vs B1 with 90% week-block bootstrap intervals; bias in " + UNITS + ".",
  ["Fit excludes stations within", "All 46: skill", "Mesonet skill", "Mesonet bias", "USCRN skill", "USCRN bias"],
  [
    ["NSRDB, uncorrected", "+24% (+19, +29)", "+27.5%", "+0.19", "+9%", "+0.48"],
    ["0 km (plain leave-one-out)", "+44% (+41, +48)", "+47%", "+0.02", "+33%", "+0.13"],
    ["50 km", "+44% (+41, +48)", "+47%", "+0.01", "+33%", "+0.13"],
    ["100 km", "+44% (+41, +48)", "+47%", "−0.01", "+33%", "+0.13"],
    ["200 km", "+44% (+41, +48)", "+47%", "−0.08", "+33%", "+0.13"],
    ["400 km", "+37% (+34, +41)", "+39%", "−0.20", "+31%", "+0.20"],
    ["v2 folds (latitude-interleaved)", "+44% (+41, +48)", "+47%", "+0.01", "+32%", "+0.20"],
  ], [3, 1.9, 1.3, 1.2, 1.2, 1.1], { bold: [4] });
P("Day-to-day skill does not depend on neighbours: it is unchanged from 0 to 200 km, and the v2 folds score the same as a 200 km buffer, so the interleaved folds do not flatter the anchored models. At 400 km, which removes most of the Mesonet from the fit for any Oklahoma station, skill falls only to +37%, still well above NSRDB. What neighbours do carry is the absolute level: as the buffer grows, Mesonet bias drifts from +0.02 to −0.20 because the fit is left with mostly USCRN stations, consistent with the network calibration difference in Section 4.9. USCRN stations are 240–290 km from their nearest neighbour, so their scores change only at 400 km.");

H2("4.11 Residual diffusion");
P("Scored on the test years against NSRDB, the 16-member diffusion ensemble improves CRPS by about 10% over a Gaussian spread whose width equals the Swin mean's dev-year RMSE (Table " + (tabNo + 1) + ", Figure " + (figNo + 1) + "), so the spread it places is informative.");
table("Probabilistic scores against NSRDB, test years (" + UNITS + ").",
  ["Forecast", "CRPS, temporal", "CRPS, both", "Spread / skill"],
  [
    ["Swin mean alone", "0.214", "0.251", "–"],
    ["Swin mean + Gaussian (dev RMSE as s.d.)", "0.171", "0.197", "–"],
    ["Diffusion, 16 members", "0.152", "0.179", "0.76 / 0.67"],
  ], [4.4, 1.6, 1.6, 1.6], { bold: [2] });
B("**Too narrow.** Spread/skill is 0.67–0.76 and the rank histogram is U-shaped and heavier at the top: the ensemble is over-confident and slightly low.");
B("**Amplitude yes, placement no.** A single member has realistic fine-scale amplitude (0.43 of NSRDB's below 30 km, against 0.19 for the mean), but its fine detail correlates with NSRDB's at only 0.05. Member SSIM is 0.34; the ensemble mean behaves like the Swin mean (SSIM 0.48 vs 0.49).");
B("**At stations.** Ensemble CRPS is 0.267 against 0.252 for the Gaussian; the shared NSRDB bias dominates. With each forecast's own station bias removed they tie (0.201 vs 0.202).");
figure("diffusion", "fig12_diffusion.png", "Same day as Figure " + FIGS.case2 + ". (a–e) NSRDB, the Swin mean, two members and the 8-member spread. (f) CRPS of the three forecasts; (g) rank histogram of NSRDB among 16 members; (h) detail amplitude relative to NSRDB by scale.");

H2("4.12 Grid resolution: 5 km");
P("Every GOES-window B + optical depth model was rebuilt on a 5 km grid (185 × 200 cells, each 10 km cell split exactly 2 × 2) and scored on the same 30,296 station-days as at 10 km. Nothing else changed:");
B("**Inputs.** Statics were built natively at 5 km from the Copernicus 90 m DEM, NLCD and Natural Earth coastlines; the 5 km statics, rebuilt at 10 km the same way, match the 10 km store (r = 1.000). NSRDB (~4 km points, about 1.5 per cell) and the MODIS and GOES fields were rebinned to the new cells.");
B("**Targets.** The anchored targets were refitted at 5 km with the same folds and the same 2001–2018 station-days.");
B("**Context.** Swin crops grew from 32 to 64 cells, so each training sample still covers 320 km.");
table("Day-to-day skill vs B1 (each on its own grid) at all 46 stations, 2020–21, and the paired 5 km − 10 km gain with its 90% week-block bootstrap interval (the same weeks resampled for both grids).",
  ["Model", "Skill 10 km", "Skill 5 km", "Gain (points)", "RMSE 10 km", "RMSE 5 km"],
  [
    ["NSRDB", "+23.7%", "+24.5%", "+0.8 (0.1, 1.6)", "0.442", "0.441"],
    ["XGBoost, NSRDB target", "+27.4%", "+27.2%", "−0.2 (−0.8, 0.4)", "0.437", "0.439"],
    ["Swin, NSRDB target", "+24.9%", "+28.1%", "+3.1 (2.3, 4.1)", "0.430", "0.417"],
    ["XGBoost, anchored v1", "+39.4%", "+40.6%", "+1.2 (0.7, 1.7)", "0.371", "0.361"],
    ["Swin, anchored v1", "+39.9%", "+43.1%", "+3.2 (2.4, 3.9)", "0.374", "0.359"],
    ["XGBoost, anchored v2", "+40.9%", "+41.3%", "+0.5 (0.0, 0.8)", "0.327", "0.327"],
    ["Swin, anchored v2", "+42.3%", "+43.8%", "+1.5 (0.8, 2.2)", "0.323", "0.318"],
  ], [3, 1.2, 1.2, 1.6, 1.15, 1.15], { bold: [6] });
figure("resolution", "fig13_resolution.png", "(a) Day-to-day skill against B1 on each grid at all 46 stations, 2020–21, with 90% week-block bootstrap intervals. (b) Paired gain of the 5 km grid over the 10 km grid, both resampled with the same weeks. Blue: Swin; orange: XGBoost; grey: NSRDB.");
B("**Swin gains, XGBoost barely.** Every Swin variant improves at 5 km, by 1.5–3.2 points, and every paired interval excludes zero. XGBoost gains 0–1.2 points: it sees one pixel at a time, so smaller pixels give it little new information, while Swin's attention windows now resolve cloud edges twice as finely.");
B("**The ranking does not change.** Swin on the v2 target is still the best model overall (RMSE 0.318 " + UNITS + ", +44%). At USCRN, v1 remains better calibrated (Swin v1: +37%, bias +0.02; Swin v2: +33%, bias +0.19).");
figure("austin10v5", "a4_austin_10v5.png", "Austin area on 14 March 2021 at 10 km (top) and 5 km (bottom), one colour scale, and the 2020–21 daily RMSE at USCRN Austin 33 NW (inside the holdout) on each grid; skill vs B1 on the same grid in brackets. White specks in the 5 km NSRDB are cells that contain no NSRDB point.");
P("Inside the Austin holdout, which no model trained on, the station pixel improves for every Swin model at 5 km: anchored v2 from RMSE 0.46 to 0.42 (+25% → +33%), the NSRDB-target model from 0.64 to 0.61, and anchored v1, already best, from 0.32 to 0.31 (+47%). The 5 km maps place the cloud edge in the same position as the 10 km maps, but with finer streaks in the clear area west of Austin. A single station cannot show whether those streaks are real.");
B("**What the test can and cannot show.** Stations are points, so a finer grid brings the estimate closer to what a station sees. Part of the gain may be this representativeness effect rather than new physical detail. NSRDB itself improves by 0.8 points just from being rebinned to smaller cells. Even so, Swin's 1.5–3.2 points exceed that, and the 10 km and 5 km models share their inputs except for cell size.");

H2("4.13 Grid resolution: 1 km subdomain");
P("To see whether the trend continues, the same models were rebuilt on a 1 km grid over a subdomain (30–35°N, 100.5–94.5°W; 567 × 579 cells) for the GOES window, and scored with the 10 and 5 km models on the station-days all three grids share: the 39 Mesonet stations and USCRN Austin, Bronte and Palestine (42 stations, 27,628 station-days). The setup differs from 5 km in four ways:");
B("**Inputs.** At 1 km a cell holds about 0.06 NSRDB points, so NSRDB, GOES and MODIS were interpolated linearly (Delaunay triangles) instead of binned. Terrain statistics were built from a 250 m DEM.");
B("**Targets.** The anchored targets were not refitted; the 5 km correction (anchored minus NSRDB) was interpolated bilinearly onto the 1 km NSRDB. The target therefore carries no information below about 4 km.");
B("**Context.** Swin crops are 128 cells (128 km) instead of 320 km, to keep the per-sample cost manageable.");
B("**Loss.** The SSIM term was dropped. Over its 11-cell window (11 km) the 1 km target is almost featureless, while the constants scale with the 128 km crop's range; 1 − SSIM stayed near 0.85, swamped the MSE (about 0.001), and the model drew sub-30 km texture at 5.6 times the target's amplitude (dev RMSE 0.53 vs 0.21 without it).");
table("Day-to-day skill vs B1 (each on its own grid) at the 42 stations of the 1 km subdomain, 2020–21, the paired 1 km − 5 km gain with its 90% week-block bootstrap interval, and RMSE (" + UNITS + ").",
  ["Model", "Skill 10 km", "Skill 5 km", "Skill 1 km", "Gain 1 vs 5 km", "RMSE 5 km", "RMSE 1 km"],
  [
    ["NSRDB", "+24.3%", "+25.3%", "+26.7%", "+1.4 (1.1, 1.8)", "0.426", "0.422"],
    ["XGBoost, NSRDB target", "+28.9%", "+29.0%", "+27.5%", "−1.5 (−2.4, −0.6)", "0.420", "0.422"],
    ["Swin, NSRDB target", "+26.2%", "+29.5%", "+28.9%", "−0.6 (−2.1, 0.8)", "0.399", "0.411"],
    ["XGBoost, anchored v1", "+40.7%", "+42.2%", "+41.3%", "−1.0 (−1.6, −0.4)", "0.357", "0.360"],
    ["Swin, anchored v1", "+40.7%", "+44.1%", "+40.5%", "−3.6 (−4.4, −2.9)", "0.357", "0.363"],
    ["XGBoost, anchored v2", "+42.1%", "+42.7%", "+41.7%", "−1.0 (−1.8, −0.4)", "0.318", "0.320"],
    ["Swin, anchored v2", "+43.6%", "+45.2%", "+41.8%", "−3.3 (−4.2, −2.6)", "0.308", "0.319"],
  ], [2.6, 0.95, 0.95, 0.95, 1.55, 0.9, 0.9], { bold: [6] });
figure("resolution1km", "fig13_resolution_5_10_1.png", "(a) Day-to-day skill against B1 on the 10, 5 and 1 km grids at the 42 stations of the 1 km subdomain, 2020–21, with 90% week-block bootstrap intervals. (b) Paired gain over the 5 km grid: filled, 1 km; open, 10 km. Blue: Swin; orange: XGBoost; grey: NSRDB.");
B("**5 km is the best grid.** Every downscaling model is at least as good at 5 km as at 1 km. The anchored Swin models lose 3.3–3.6 points at 1 km, about what they gained going from 10 to 5 km; XGBoost loses about 1 point. Only NSRDB gains (+1.4), and only because it is interpolated rather than binned, which is a representativeness effect, not new information.");
B("**Why 1 km does not help.** Below about 4 km neither the target nor the satellite inputs contain new information, so a finer grid can only add noise. Three further changes work against Swin at 1 km and cannot be separated here: crops cover 128 km instead of 320 km, the SSIM term was removed, and the anchored correction was transferred from 5 km instead of refitted.");
B("**USCRN.** At the three USCRN stations in the subdomain the pattern is the same: Swin v1 scores +35.6% at 1 km against +39.0% at 5 km, and its bias rises from +0.08 to +0.12 " + UNITS + ".");

H2("4.14 Uncertainty: intervals, an uncertainty index and worst-case scenarios");
P("The models above give one value per cell and day. For planning, users also need how far that value can be off. Intervals were built around the recommended model (Swin, anchored v2, 5 km) from four estimates of spread, and scored at the stations, each station by the models that never saw it:");
B("**Fixed width per month.** The same interval for every cell and day of a calendar month: the cheap baseline any method must beat.");
B("**Monte Carlo drop-path.** The Swin's stochastic-depth layers are kept active at inference, so each of 20 forward passes drops a different set of residual blocks. This samples the model's own uncertainty without retraining.");
B("**Spread of the three Swin targets.** Swin trained on NSRDB, anchored v1 and anchored v2: how much the answer depends on which reference the model was trained to match.");
B("**Spread of six models.** The same three targets with XGBoost added.");
P("Raw spreads are too narrow (the six-model spread is 0.60 of the error, Monte Carlo 0.07), so each spread *s* was turned into an interval centre ± *q*·*s* by split conformal calibration: *q* is the 90th percentile of |error| / *s* in one test year, the interval is scored in the other year, and the years then swap. Every method therefore covers close to 90% by construction, and the comparison is about width: the narrower interval at the same coverage is more useful.");
table("90% intervals around Swin anchored v2 at 5 km, 46 stations, 2020–21 (" + UNITS + "). Interval score = width plus 20 × the amount by which the observation falls outside; lower is better. Rank correlation: Spearman correlation of interval width with the absolute error.",
  ["Spread estimate", "Coverage", "Mean width", "Interval score", "Rank correlation", "Coverage, USCRN"],
  [
    ["Fixed width per month", "88.9%", "0.97", "1.44", "0.22", "78.1%"],
    ["Monte Carlo drop-path", "89.5%", "1.14", "1.55", "0.23", "77.4%"],
    ["Spread of 3 Swin targets", "89.5%", "1.05", "1.46", "0.30", "83.4%"],
    ["Spread of 6 models", "89.7%", "1.00", "1.41", "0.32", "84.0%"],
  ], [2.6, 1, 1, 1.2, 1.3, 1.3], { bold: [3] });
figure("uq_stations", "fig14_uncertainty.png", "(a) Interval score of the calibrated 90% interval for each spread estimate at 5 km (lower is better), with its mean width and the share of station-days it covers. (b) Mean absolute error of Swin v2 in each class of the uncertainty index (tercile of interval width) for the three spread estimates (edge colour). (c) Mean interval width on the 10, 5 and 1 km grids at the 42 stations they share.");
B("**Monte Carlo drop-path adds nothing.** Its raw spread is only 0.07 of the error, and once calibrated its interval is wider than the monthly band at the same coverage (1.14 vs 0.97 " + UNITS + "). The model is confident in a way that does not track its error: what it does not know is mostly what the target does not contain, not which of its own blocks fire.");
B("**Model spread carries information, modestly.** The six-model spread ranks errors (ρ = 0.32 against 0.22 for the monthly band) and gives the lowest interval score, 2% below the monthly band. Its gain is small because most of the error at a station is not something the models disagree about.");
B("**The index separates good days from bad.** In the green tercile the mean error is 0.15 " + UNITS + ", in the red 0.31, twice as large. Coverage is not uniform: 84% in green, 94% in red, so green intervals are slightly too narrow and red ones slightly too wide.");
B("**USCRN is under-covered.** At USCRN only 77–84% of station-days fall inside, because the v2 target is biased high there (Section 4.9). The network disagreement is a level shift that no spread estimate sees; it has to be stated separately as a ±0.1–0.15 " + UNITS + " uncertainty in the absolute level.");
B("**Resolution.** At the same 90% coverage the six-model interval is 0.97 " + UNITS + " wide at 5 km, 1.00 at 10 km and 1.02 at 1 km. The uncertainty is smallest at 5 km, the same answer as the skill comparison.");
figure("uq_austin", "fig15_uq_austin.png", "Uncertainty products for 14 March 2021 at 5 km: (a) central estimate; (b) half-width of the calibrated 90% interval; (c) uncertainty index; (d) low-end scenario, the central estimate minus the half-width.");
P("Figure " + FIGS.uq_austin + " shows the products a user would receive for one day. On 14 March 2021 the central estimate for East Austin is 2.87 " + UNITS + " with a 90% interval of 2.28–3.45 (amber); the low-end scenario is the level that daily GHI falls below on about one day in twenty, the quantity needed for a worst-case solar-supply or building-load calculation. Uncertainty is largest along the cloud edge running north–south through the holdout, where the three targets disagree most about where the edge lies.");
table("Distribution and extremes at 5 km, 46 stations, 2020–21. KL: divergence of each model's daily-GHI distribution from the stations' (lower is closer). Dark and bright days: each station's lowest and highest 10% of days in each calendar month; POD = share of those days the model also places there; bias on those days in " + UNITS + ".",
  ["Model", "KL", "Dark POD", "Dark bias", "Bright POD", "Bright bias"],
  [
    ["B1 bilinear", "0.054", "57%", "+0.52", "70%", "−0.02"],
    ["NSRDB", "0.023", "76%", "+0.25", "93%", "+0.15"],
    ["Swin, NSRDB target", "0.022", "76%", "+0.25", "89%", "+0.11"],
    ["Swin, anchored v1", "0.019", "93%", "−0.11", "37%", "−0.23"],
    ["Swin, anchored v2", "0.004", "85%", "+0.06", "73%", "−0.03"],
    ["XGBoost, anchored v2", "0.003", "84%", "+0.08", "75%", "−0.03"],
  ], [2.6, 0.9, 1.1, 1.1, 1.1, 1.1], { bold: [4] });
B("**Distribution.** The anchored v2 models reproduce the stations' distribution of daily GHI almost exactly (KL 0.003–0.004, against 0.023 for NSRDB and 0.054 for bilinear POWER).");
B("**Extremes.** Bilinear POWER misses dark days (57% detected, +0.52 " + UNITS + " too bright on them). The v2 models detect 85% of dark days with a bias of only +0.06. NSRDB detects 93% of bright days, but only because it is too bright overall (+0.15); v1 detects 93% of dark days for the opposite reason. v2 is the only model that is close on both tails.");

// ===================================================================== 5 discussion
H1("5. Discussion");
H2("5.1 The target is the ceiling");
P("Better inputs raise skill against NSRDB substantially (holdout RMSE 0.478 → 0.397 with MODIS) but only return station skill to B1's level. The target swap explains why: a model moves toward its target's errors, so improvement measured against the target says little about improvement at the ground. Any new target should be scored at ground stations before a model is trained on it, and any downscaling claim should include an independent station check.");
H2("5.2 What optical depth adds");
P("Cloud presence (MODIS, GOES mask) tells the model where clouds are; optical depth tells it how much light they remove. The latter is the quantity daily GHI depends on, and it is the only input that took station skill above NSRDB's. That NOAA's own DSR product, built from the same satellite, scores far below B1 indicates the value lies in combining a cloud-attenuation proxy with POWER's well-calibrated daily totals, not in replacing POWER.");
H2("5.3 Network calibration");
P("The anchored target improves day-to-day skill at stations of a network it never saw, which argues that it learns NSRDB's error rather than one network's quirks. Absolute level is different: USCRN and Mesonet disagree by 0.2–0.3 " + UNITS + " relative to NSRDB, and a single correction can match one or the other or split the difference (v2). Deciding which is right needs a third reference—SURFRAD or BSRN stations outside the domain, or co-located instruments.");
H2("5.4 Texture");
P("Losses that reward texture amplitude (AMSE, post-hoc gains) produce texture in the wrong place and lower SSIM. Diffusion produces texture of the right amount but likewise not co-located with NSRDB's. Placement comes only from inputs that observe clouds at the relevant scale, and even GOES at 48 scans per day explains a minority of NSRDB's sub-60 km variance. Realistic texture for applications that need it is best drawn from the diffusion ensemble and labelled as stochastic.");
H2("5.5 Limitations");
B("The GOES-window comparisons rest on 18 months of training. In the main window, Track A loses 4% at stations; in the GOES window it loses 13%, so a longer GOES record (2018–2025) may change the ranking of GOES inputs.");
B("Seven USCRN stations constrain USCRN results; intervals are correspondingly wide.");
B("The leakage test (Section 4.10) refits the error model, not the downscalers; it bounds neighbour leakage through the target but does not retrain a model for each buffer.");
B("Track C uses the same satellite as NSRDB and is flagged accordingly.");
B("The Austin urban artefact (developed-land extrapolation) is not yet fixed.");
B("The diffusion model is trained on the NSRDB residual, not on the anchored target.");
B("Model uncertainty was sampled only by Monte Carlo drop-path and by the spread across already-trained models; a deep ensemble (the same network retrained with several seeds) was not run. The interval coverage of 90% holds for the station mix scored; at USCRN alone it is 77–84%.");
B("The 1 km comparison covers 42 stations, uses anchored targets transferred from 5 km, shorter Swin crops (128 km) and no SSIM term; it tests the 1 km pipeline as built, not the best possible 1 km model.");

// ===================================================================== 6 conclusions
H1("6. Conclusions and recommendations");
P("Train on the station-anchored v2 target fitted to both networks. Where GOES exists (2018 on), use Swin B + cloud optical depth: tied with XGBoost Track B + optical depth at stations (RMSE 0.323 vs 0.327 " + UNITS + "; +45% vs +44% over B1 at Mesonet, +31% vs +30% at USCRN) and better in spatial structure, with XGBoost as the inexpensive fallback. For 2001–2017, Swin B + MODIS on the v2 target is markedly weaker (RMSE 0.367; +27% at Mesonet and +19% at USCRN): it removes NSRDB's bias but matches NSRDB's day-to-day skill at Mesonet rather than exceeding it, because MODIS cannot place clouds as optical depth can. Where NSRDB itself is available, the corrected NSRDB (+44% under the 200 km buffer test) is the stronger pre-2018 product; the POWER downscaler is for settings without NSRDB. For Texas specifically, the USCRN-anchored v1 target is better calibrated (bias ≈ 0) than v2 (+0.2). Keep Track C flagged, and ship the diffusion ensemble only as an uncertainty layer after spread inflation or retraining on the anchored target.");
P("On the 5 km grid the same recipe gains another 1.5 points with Swin (RMSE 0.318, +44% over B1 at all 46 stations) and nothing is lost, so 5 km is the recommended grid where GOES exists. A 1 km grid does not help: on the 1 km subdomain the anchored Swin models lose 3.3–3.6 points against 5 km, because neither the target nor the inputs resolve anything below about 4 km.");
P("Ship each day's field with a calibrated 90% interval and the green/amber/red index built from the spread of the anchored models (Section 4.14). The interval covers 90% of Mesonet station-days and doubles from green to red in error; the absolute-level disagreement between networks (±0.1–0.15 " + UNITS + ") should be stated beside it, because no spread estimate captures it.");
P("Next steps, in priority order:");
B("Improve the pre-2018 model: MODIS alone leaves day-to-day skill at NSRDB's level; candidates are AVHRR/PATMOS-x optical depth or GOES-13 cloud products for 2001–2017.");
B("Resolve the inter-network level with an external reference (SURFRAD/BSRN) and decide how much bias the anchored target should keep.");
B("Extend the GOES record to 2018–2025 and retrain the optical-depth models.");
B("Cap or drop the developed-land feature to remove the Austin artefact; drop kt_clim_fine, which changed neither accuracy nor station skill.");
B("Retrain the diffusion model on the anchored target and calibrate its spread.");

// ===================================================================== reproducibility
H1("Data and code availability");
P("All steps are modules of the downscaling-radiation repository. Configurations are YAML files; the environment variables DOWNSCALE_CONFIG_data and DOWNSCALE_CONFIG_train switch experiments, and runs are written to runs/<group>/. The main commands are:");
const code = [
  "python -m data.build_store                      # POWER, NSRDB, statics -> zarr store",
  "python -m data.goes_cloud --product cod days --start 2018-01-01 --end 2021-12-31",
  "python -m training.train_xgb --track B+cod      # tracks: A, B, C, +cod, +T",
  "python -m training.train_swin --track B --tag modis",
  "python -m evaluation.station_anchor errmodel    # A2, leave-one-station-out",
  "python -m evaluation.station_anchor build-target --exclude <fold stations> --fold a",
  "python -m training.train_diffusion --mean-run swin_B_modis --tag B",
  "python -m evaluation.evaluate --models B1 xgb_B_cod --out eval_cod",
  "python -m evaluation.probabilistic --runs diff_B",
  "python -m evaluation.station_anchor networks    # two-network scores",
  "python -m evaluation.station_anchor buffered    # spatial-leakage test",
  "python -m evaluation.figures; python -m evaluation.schematics",
  "bash slurm/build_1km.sh                         # 1 km subdomain store (resumable)",
  "RES=5km sbatch slurm/train_seq.sh               # XGBoost, Swin, eval per target",
  "python -m evaluation.resolution --res 5 10 1    # 10 / 5 / 1 km comparison",
  "python -m evaluation.uncertainty sample --members 20   # per run group, GPU",
  "python -m evaluation.uncertainty analyse --res 5 10 1  # intervals, index, maps",
];
code.forEach((l) => body.push(new Paragraph({ spacing: { after: 0 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F3F3" }, children: [new TextRun({ text: l, font: "Consolas", size: 16 })] })));
body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));

// ===================================================================== references
H1("References");
const refs = [
  "Brock, F. V., et al. (1995). The Oklahoma Mesonet: a technical overview. *Journal of Atmospheric and Oceanic Technology*, 12, 5–19.",
  "Cao, H., et al. (2022). Swin-Unet: Unet-like pure Transformer for medical image segmentation. *ECCV Workshops*.",
  "Chen, T., & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of KDD*, 785–794.",
  "Diamond, H. J., et al. (2013). U.S. Climate Reference Network after one decade of operations. *Bulletin of the American Meteorological Society*, 94, 485–498.",
  "Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Quarterly Journal of the Royal Meteorological Society*, 140, 1917–1923.",
  "Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *NeurIPS*.",
  "Liu, Z., et al. (2021). Swin Transformer: hierarchical vision Transformer using shifted windows. *ICCV*.",
  "Mardani, M., et al. (2025). Residual corrective diffusion modeling for km-scale atmospheric downscaling. *Communications Earth & Environment*, 6, 124.",
  "Muñoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383.",
  "Sengupta, M., et al. (2018). The National Solar Radiation Data Base (NSRDB). *Renewable and Sustainable Energy Reviews*, 89, 51–60.",
  "Stackhouse, P. W., et al. NASA Prediction Of Worldwide Energy Resources (POWER), Data Access Viewer and API, v10. https://power.larc.nasa.gov.",
  "Walther, A., & Heidinger, A. K. (2012). Implementation of the Daytime Cloud Optical and Microphysical Properties algorithm (DCOMP) in PATMOS-x. *Journal of Applied Meteorology and Climatology*, 51, 1371–1390.",
  "Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13, 600–612.",
  "Wielicki, B. A., et al. (1996). Clouds and the Earth's Radiant Energy System (CERES): an Earth observing system experiment. *Bulletin of the American Meteorological Society*, 77, 853–868.",
];
refs.forEach((r) => body.push(new Paragraph({ children: runs(r, { size: 19 }), indent: { left: 360, hanging: 360 }, spacing: { after: 80 } })));

// ===================================================================== appendix
body.push(new Paragraph({ children: [new PageBreak()] }));
H1("Appendix A. Hyperparameters");
table("Model and training hyperparameters.",
  ["Component", "Setting"],
  [
    ["XGBoost mean model", "depth 8; η 0.05; ≤ 2,000 rounds; early stop 50; 3 M rows stratified on bilinear *k*_{t}; loss weighting chosen on dev"],
    ["Swin U-Net", "patch 2; window 4; embed 96; depths [2, 2, 6]; heads [3, 6, 12]; drop-path 0.1; zero-initialised output conv; PixelShuffle head"],
    ["Swin training", "crop 32 × 32; batch 16; 40 epochs; AdamW lr 2 × 10^{−4}, weight decay 0.05; loss MSE + 0.2 (1 − SSIM); best dev RMSE"],
    ["Error model (anchoring)", "XGBoost depth 4, 400 trees; Track B features + NSRDB index; training-year station-days"],
    ["Diffusion U-Net", "base 64; multipliers (1, 2, 4); 2 ResBlocks/level; FiLM on Fourier(ln σ / 4); EDM preconditioning"],
    ["Diffusion training", "ln σ ~ N(−1.2, 1.2²); 60k steps; batch 64; crop 32; EMA 0.999; checkpoint by dev CRPS"],
    ["Diffusion sampling", "18 Heun steps; 16 members"],
    ["Station bootstrap", "week blocks; 500 draws; 90% percentile intervals"],
    ["Swin, 5 km grid", "crop 64 × 64 (320 km, as at 10 km); otherwise as above"],
    ["Swin, 1 km grid", "crop 128 × 128 (128 km); batch 8; no SSIM term (Section 4.13); otherwise as above"],
    ["Monte Carlo drop-path", "drop-path layers active at inference; 20 passes per day; test years"],
    ["Intervals", "90% level; split conformal across the two test years; index = tercile of interval width"],
  ], [2.4, 6.6]);
H1("Appendix B. Per-station scores, Oklahoma Mesonet");
table("Per-station day-to-day skill vs B1 (%) at the 39 Mesonet stations, v2 target, 2020–21. Bias is Swin's, in " + UNITS + ".",
  ["Station", "Fold", "Days", "NSRDB", "XGBoost v2", "Swin v2", "Swin bias"], meso, [1.6, 0.8, 0.9, 1, 1.2, 1.1, 1.1]);

// ===================================================================== assemble
const doc = new Document({
  creator: "Guruprasad Parasnis",
  title: "Downscaling Daily Surface Solar Radiation from NASA POWER to 10 km",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2E4A7D" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: title },
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }, pageNumbers: { start: 1 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Radiation downscaling: POWER → 10 km", size: 16, color: "777777", italics: true })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
      children: body },
  ],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, figNo, "figures", tabNo, "tables"); });

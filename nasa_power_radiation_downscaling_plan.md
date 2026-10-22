# NASA POWER Solar Radiation Downscaling: Implementation Plan

## 1. Goal

Downscale daily surface shortwave radiation from NASA POWER (coarse input) to a fine grid over south-central USA, reusing the grid, split, and model infrastructure from the NASA POWER to ERA5-Land temperature downscaling work.

- **Primary variable:** `ALLSKY_SFC_SW_DWN` (GHI, all-sky surface shortwave downward, daily total)
- **Secondary (phase 2+):** `ALLSKY_SFC_SW_DIFF` (diffuse), `ALLSKY_SFC_SW_DNI` (direct normal), `ALLSKY_SFC_LW_DWN` (longwave)
- **Target resolution:** 0.1° on the same master grid as the temperature work (EPSG:5070 for modeling), optional 4 km stage later
- **Success criterion:** beat bilinear interpolation of POWER on RMSE and MBE against both the gridded target and independent ground stations, stratified by sky condition

## 2. Key Design Decisions (read first)

### 2.1 The target is the hard part (same lesson as the temperature paper)

Radiation is dominated by clouds, not terrain, across most of Texas/Oklahoma. That makes the choice of fine-resolution target the single biggest decision.

| Candidate target | Res. | Verdict |
|---|---|---|
| **NSRDB (NREL, PSM v3/v4)** | ~4 km, 30 min | **Use as primary gridded target.** Satellite-derived (GOES), real fine-scale cloud signal. |
| ERA5-Land `ssrd` | 0.1° | **Do not use as target.** ERA5-Land radiation is ERA5 atmospheric forcing interpolated, so it carries ~31 km information on a 9 km grid. Fine for a sanity baseline only. |
| ERA5 `ssrd` | 0.25° | Coarse comparison baseline only. |
| Ground stations (SURFRAD, ARM SGP, Oklahoma Mesonet, TexMesonet / West Texas Mesonet pyranometers) | point | **Use as independent truth** for final evaluation. Never train on them. |

**Caveat to state in the paper:** NSRDB is a model product, not ground truth. It shares some inputs (MERRA-2 aerosol and water vapor) with the reanalysis chain behind POWER, so input and target errors may be correlated. Station validation is what breaks that circularity.

### 2.2 Predict clearness index, not raw radiation

Model the clearness index `kt = GHI / TOA` (POWER provides `ALLSKY_KT` and `TOA_SW_DWN`).

- Bounded in [0, 1], removes the seasonal and latitudinal cycle, so the model learns clouds and aerosols rather than astronomy.
- Reconstruct: `GHI_fine = kt_fine_hat × TOA_fine`, where `TOA_fine` is computed analytically per pixel per day (exact, no learning needed).
- Clip predictions to [0, ~0.85] and also check against `CLRSKY_SFC_SW_DWN` as a physical ceiling.

### 2.3 Restrict the time period to avoid a source discontinuity

POWER solar parameters switched from the older SRB-based record to CERES-based data in the early 2000s. Use **2001 onward** only, and confirm the exact switch date from POWER documentation before building the dataset. Overlap with NSRDB GOES-era coverage defines the final window.

### 2.4 Be honest about which covariates carry real information

| Covariate group | Examples | Information content |
|---|---|---|
| Static terrain | DEM, slope, aspect, sky-view factor, horizon angle | Small effect on horizontal GHI in Texas except Hill Country and Trans-Pecos. Matters more for DNI and tilted surfaces. |
| Static surface | NLCD land cover, MODIS albedo (MCD43), distance to coast, NDVI climatology | Captures persistent patterns (coastal stratus, urban aerosol). |
| Dynamic clouds | MODIS cloud fraction (MOD/MYD06, MOD35), GOES-16 ABI clear-sky mask (2017+) | **Most of the real fine-scale signal.** |
| Dynamic atmosphere | MAIAC AOD (1 km), MERRA-2 precipitable water | Moderate, mostly in clear-sky days. |

**Leakage-style risk:** NSRDB is itself built from GOES cloud retrievals. Feeding GOES cloud products as predictors lets the model partly reproduce NSRDB's own algorithm, which inflates scores without real downscaling skill. Run two tracks and report both:

- **Track A (static-only):** POWER + static covariates. Honest lower bound on what pure downscaling achieves.
- **Track B (dynamic):** Track A + MODIS cloud fraction + AOD. Avoid GOES products as inputs when NSRDB is the target, or flag the result clearly if used.

## 3. Data Pipeline

### 3.1 Sources and access

| Dataset | Variables | Access |
|---|---|---|
| NASA POWER | `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `ALLSKY_KT`, `TOA_SW_DWN`, `ALLSKY_SFC_SW_DIFF`, `ALLSKY_SRF_ALB`, `CLOUD_AMT` | POWER regional/daily API or S3 (Zarr). Use the same domain as the temperature work. |
| NSRDB | GHI, DNI, DHI (30 min) | NREL API / HSDS (h5pyd) or AWS open data. Aggregate to daily totals. |
| DEM | elevation, slope, aspect, SVF, horizon | 3DEP or SRTM; derive with `richdem` or `whitebox`. |
| MODIS | cloud fraction, albedo, AOD (MAIAC) | Earthdata / AppEEARS. |
| Stations | measured GHI | SURFRAD, ARM SGP, Oklahoma Mesonet, TexMesonet. |

### 3.2 Unit harmonization

- POWER daily GHI: kWh/m²/day (check the API `community` parameter, AG returns MJ/m²/day).
- NSRDB: W/m² instantaneous at 30 min. Daily total: `sum(GHI_i) × 0.5 h / 1000` → kWh/m²/day. Use **local solar day**, not UTC day, and handle DST consistently.
- Log everything in physical units (kWh/m²/day) in W&B, same as the °C convention in the temperature work.

### 3.3 Grids

- Coarse: POWER native solar grid (verify in docs; solar parameters are coarser than POWER meteorology).
- Fine: reuse the 0.1° master grid from the temperature pipeline, reprojected to EPSG:5070 for modeling.
- Regrid NSRDB to fine grid by **area-weighted mean** (conservative), not bilinear.
- Precompute the coarse-to-fine index mapping once and cache it.

### 3.4 Quality control

- Drop NSRDB fill values and flagged intervals.
- Station QC: BSRN-style physical limits, closure test (GHI vs DNI·cos(z) + DHI) where all three are measured, drop days with >10% missing daylight samples.
- Check for POWER/NSRDB version changes inside the study period.

## 4. Features

Per fine pixel per day:

1. `kt_power_bilinear`: POWER `kt` bilinearly upsampled
2. `kt_power_coarse_cell`: raw value of the enclosing coarse cell
3. `kt_clr_ratio`: POWER all-sky / clear-sky ratio
4. `kt_anom`: POWER `kt` minus its training-period day-of-year climatology (climatology from training days only)
5. Rolling means of `kt` (3 and 7 day)
6. Static: elevation, `dem_anom` (vs bilinear mean of DEM), slope, aspect (sin/cos), sky-view factor, land-cover fractions, albedo, distance to coast
7. Fine-scale `kt` climatology from training-period NSRDB (day-of-year smoothed). **Training years only.**
8. Day-of-year (sin/cos), latitude, longitude
9. Track B only: MODIS cloud fraction (Terra + Aqua mean), MAIAC AOD, precipitable water

## 5. Models (in order)

| Stage | Model | Purpose |
|---|---|---|
| B0 | Bilinear POWER GHI | Floor baseline |
| B1 | Bilinear `kt` × analytic TOA | Tests whether the kt reframing alone helps |
| B2 | B1 + topographic correction (horizon shading, SVF on diffuse) | Physics-only baseline |
| B3 | B1 + fine-scale NSRDB climatology ratio (static bias map) | Strong "climatology" baseline. Must beat this, not just bilinear. |
| M1 | XGBoost on residual `kt_target - kt_bilinear` | Main tabular model |
| M2 | Swin U-Net (reuse temperature architecture), input = stacked coarse + static channels, output = residual `kt` | Spatial model |
| M3 (optional) | M2 with quantile head (0.1 / 0.5 / 0.9) | Uncertainty, partly cloudy days |

**Loss:** MSE in `kt` space, optionally weighted by `TOA` so errors are proportional to energy impact. Compare against MSE directly in kWh/m²/day space on the dev set and keep whichever wins.

**Optional constraint:** coarse-cell consistency (mean of fine predictions inside a coarse cell equals POWER value after bias correction). Test as a soft penalty only, because POWER and NSRDB have systematic bias differences, so a hard constraint would force the model to inherit POWER bias.

## 6. Splits and Leakage Controls

Keep the same split philosophy as the temperature paper:

- **Temporal:** train / dev / test by year (for example, earliest years train, one year dev, last two years test).
- **Spatial:** Austin region held out entirely as a spatial test set.
- Report three test cells: temporal-only, spatial-only, both.
- All climatologies, normalization stats, and feature scalers fit on **training years and training pixels only**.
- Swin patches must not straddle the spatial holdout boundary; mask a buffer of at least one coarse cell around it.

## 7. Evaluation

### 7.1 Metrics (kWh/m²/day)

- RMSE, MBE, nRMSE (% of mean), Pearson r, KGE
- Skill score vs B0 and vs B3

### 7.2 Stratification

- **Sky condition by target kt:** overcast (< 0.3), partly cloudy (0.3 to 0.65), clear (> 0.65). Partly cloudy is where downscaling should matter most and where models usually fail.
- Season (DJF / MAM / JJA / SON)
- Terrain class (flat vs Hill Country / Trans-Pecos)
- Coastal vs inland

### 7.3 Independent station validation

- Compare all models at station pixels against measured GHI.
- If the gap between "model vs NSRDB" and "model vs station" is large, report it. That gap is the target-quality story.

### 7.4 Diagnostics

- Spatial skill maps (per-pixel RMSE and correlation)
- Power spectra of predicted vs target fields (checks whether fine-scale variance is real or smoothed)
- Feature importance (XGBoost gain + SHAP) per track

## 8. Complexity

Let `N_c` = coarse cells, `r` = refinement factor per axis, `T` = days, `F` = features.

| Component | Time | Space |
|---|---|---|
| NSRDB daily aggregation | O(N_f × 48 × T) | O(N_f × T) after aggregation |
| Fine samples | N_f = N_c × r² | O(N_f × T × F) feature table |
| Analytic TOA | O(N_f × T), trivial | O(N_f × T) or compute on the fly |
| XGBoost training | O(K × N_f T × F × log(N_f T)) for K trees (hist method: O(K × N_f T × F)) | O(N_f T × F) |
| Swin U-Net (per patch, window size w) | O(H W × w² × C) per layer | O(H W × C) activations |
| Inference | O(N_f × T) | streaming by day |

Rough scale: a 1° to 0.1° step gives r² = 100 fine cells per coarse cell. At ~1,000 coarse-equivalent cells and ~20 years, that is about 7 × 10⁸ pixel-days. Subsample days for XGBoost (stratified by sky condition) and use chunked Zarr for the Swin data loader.

## 9. Phases (Claude Code handoff)

| Phase | Deliverable | Exit check |
|---|---|---|
| 0. Setup | Repo scaffolding reusing temperature pipeline modules (grid, splits, W&B logging) | Tests pass on shared modules |
| 1. Ingest | POWER + NSRDB + statics on master grid as Zarr | Unit test: daily NSRDB total matches manual calc for 3 random pixels |
| 2. Stations | Cleaned station GHI table with pixel mapping | QC pass rate reported per station |
| 3. Baselines | B0 to B3 metrics on all test cells | B1 not worse than B0; B3 reported |
| 4. XGBoost | Track A and Track B results | Beats B3 on dev, not just B0 |
| 5. Swin U-Net | Same tracks, same splits | Spectra show fine-scale variance closer to target than XGBoost |
| 6. Evaluation | Stratified tables, skill maps, station comparison | All three test cells reported |
| 7. Extensions | DNI / DHI / LW, quantile head, 4 km stage | Optional |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Static covariates add little in flat terrain | Report Track A honestly as a lower bound; the finding itself is publishable |
| GOES-derived predictors reproduce NSRDB | Exclude from Track B or flag clearly |
| POWER/NSRDB systematic bias dominates the residual | Separate bias correction (B3) from spatial downscaling in the results |
| Local solar day vs UTC day mismatch | Single tested utility for daily aggregation, used everywhere |
| POWER source discontinuity | Restrict to CERES era; verify switch date |
| Few stations in the Austin holdout | Use nearest TexMesonet stations; report station count per region |

## 11. Open Questions

1. Final target resolution: stay at 0.1° for consistency with the temperature paper, or push to native NSRDB ~4 km?
2. Is this a standalone paper or a second variable for the "bottleneck is the target" paper? Radiation (target dominated by satellite cloud retrievals) is a strong second case for that framing.
3. Daily only, or hourly later? Hourly changes the problem (solar geometry, cloud timing) and roughly 24× the data.

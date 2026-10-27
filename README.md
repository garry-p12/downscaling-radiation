# Radiation Downscaling: NASA POWER 1° → 10 km

Downscales daily all-sky surface shortwave (GHI, `ALLSKY_SFC_SW_DWN`) from
**NASA POWER** (CERES SYN1deg, native 1° × 1°) to the **10 km EPSG:5070 grid
of the temperature project** over South-Central US, with **NSRDB** as the
gridded target and ground pyranometers as independent truth. The full plan is
in [nasa_power_radiation_downscaling_plan.md](nasa_power_radiation_downscaling_plan.md).

| | |
|---|---|
| Domain | 27–35° N, 104–94° W → 93 × 100 cells (identical to `downscaling-temperature`) |
| Coarse input | POWER daily, **1° × 1°**, local-solar-time days (`time-standard=LST`) |
| Target | NSRDB GOES aggregated v4.0.0, 4 km / 30 min → local-solar-day totals → 10 km (area-weighted binning) |
| Modelled quantity | clearness index residual `kt_nsrdb − kt_bilinear`; GHI = kt × analytic TOA |
| Period | 2001–2021: train 2001–18, dev 2019, test 2020–21 |
| Holdout | Austin box + 1° buffer (one coarse cell), never trained on |
| Units | kWh/m²/day everywhere |

## Facts checked while building this (2026-10)

- **POWER solar grid is 1° × 1°**, not the 0.5° × 0.625° meteorology grid: a
  live POWER v10.0.0 regional response returns centres at x.5°. So this is a
  ~100 km → 10 km problem (r ≈ 10 per axis), twice the factor of the
  temperature task.
- **POWER source switch: GEWEX SRB → CERES SYN1deg Ed4.2 on 2001-01-01**
  (POWER Data Sources page). The last ~3–4 months of the record are CERES
  FLASHFlux, which is a second source change. The split starts in 2001 and ends
  well before the FLASHFlux tail.
- POWER units under `community=RE`: irradiation in `kW-hr/m^2/day`, `ALLSKY_KT`
  dimensionless, `CLOUD_AMT` in %, `PW` in cm. Units are read from each file and
  converted (`common/units.py`). `AG` would return MJ.
- Analytic TOA (`common/solar.py`) matches POWER `TOA_SW_DWN` within 1%
  (29.5° N, 2020-06-01: 11.32), and matches FAO-56 Example 8 exactly.
- **NSRDB** `s3://nrel-pds-nsrdb/GOES/aggregated/v4.0.0/nsrdb_{1998..2025}.h5`
  can be read anonymously over h5py + s3fs. It has `ghi` (uint16 W/m²),
  `fill_flag`, 30-min UTC `time_index`, and per-site `meta`. Each file is
  ~1.6 TB; only the domain's columns are read, about 4 GB per year.
- NSRDB aggregated `fill_flag` is **the percent of gap-filled native samples
  (0–100)**, not the 0–8 code its attribute lists. Dropping every nonzero flag
  left 57% of days NaN, so only fully filled samples are dropped
  (`fill_flag_drop_at: 100`).
- A real end-to-end check on a 0.5° box near Austin, 2020 (169 NSRDB sites,
  ~6 min over S3) gave 13.9% NaN site-days. Box-mean NSRDB vs POWER daily GHI:
  r = 0.988, RMSE 0.38 kWh/m²/day, and NSRDB is about 3% higher. Correlation
  peaks at lag 0 (0.69–0.70 at ±1 day), which confirms that the solar-day
  definitions line up.
- NSRDB changes satellite generation (GOES-13/15 → GOES-16/17) in **2018**.
  `data/check_discontinuity.py` reports the yearly NSRDB − POWER offset so a
  step there shows up before modelling.
- **No SURFRAD site and no ARM SGP facility lies inside the domain** (Goodwin
  Creek is at 89.9° W; the SGP Central Facility is at 36.6° N). The in-domain
  truth is the Oklahoma Mesonet south of 35° N plus the Texas networks via
  generic CSV. Both parsers are still included.
- The temperature store's input tensor holds physical values (not z-scores),
  so its statics are imported directly.

## Layout

```
common/      config, grid, normalize, tracking  (vendored from downscaling-temperature@1d2c3bb)
             solar.py   declination, zenith/azimuth, analytic daily TOA
             daily.py   THE daily aggregation: local solar day, night handling, 10% daylight rule
             regrid.py  cached sparse maps: bilinear, enclosing cell, conservative binning
             splits.py  year blocks, holdout + buffer, the three test cells
             units.py   -> kWh/m2/day
data/        download_power, download_nsrdb, statics (+ terrain.py: slope/aspect/horizon/SVF/beam
             shading), stations (Mesonet/SURFRAD/ARM/CSV + BSRN QC), build_store, synthetic,
             check_discontinuity
features/    builder.py: plan-§4 features, training-only climatologies, holdout fill, physical clip
models/      baselines (B0-B3), xgb (M1), swin (M2/M3; backbone = temperature encoder_swin.py)
training/    train_xgb, train_swin, sampling
evaluation/  evaluate (cells x strata, skill, maps, spectra, stations), metrics, spectral
scripts/     run_pipeline.py: phases 3-6 in sequence
```

## Setup

```bash
conda activate downscale            # or: pip install -e ".[dev]"
pytest                              # 46 tests, offline, ~5 s
```

## Running it

```bash
# Phase 1: ingest (no credentials needed)
python -m data.download_power --years $(seq 2001 2021)
python -m data.download_nsrdb --years $(seq 2001 2021)      # ~4 GB/yr over the network
python -m data.build_store                                  # -> data_store/radiation.zarr
python -m data.check_discontinuity

# Phase 2: stations (independent truth)
python -m data.stations download-mesonet --years 2020 2021
python -m data.stations build                               # QC pass rate per station, region counts

# Phases 3-6
python -m scripts.run_pipeline                              # XGB A/B, Swin A/B, Swin quantile, eval
# or individually:
python -m training.train_xgb --track A                      # loss_weight auto: none/toa/energy on dev
python -m training.train_swin --track B
python -m training.train_swin --track A --quantiles 0.1 0.5 0.9
python -m evaluation.evaluate
```

### On TACC Vista (how the real run is done)

```bash
python -m data.statics data_store/statics.npz     # here: needs the temperature store + DEM
scripts/vista_sync.sh push && scripts/vista_sync.sh statics
ssh vista2 'cd $WORK/downscaling-radiation && setsid nohup bash slurm/downloads.sh > logs/downloads.log 2>&1 < /dev/null &'
ssh vista2 'cd $WORK/downscaling-radiation && setsid nohup bash slurm/after_downloads.sh > logs/chain.log 2>&1 < /dev/null &'
scripts/vista_sync.sh pull                        # runs/ and logs/ back here when done
```

Downloads run on the vista2 login node. `after_downloads.sh` checks that the
downloads are complete and then submits the Slurm chain on `gh` (account
ATM23014): build → {xgb, swin A, swin B, swin A quantile} → eval. Data lives on
`$SCRATCH` (`configs/data_vista.yaml`). Measured on Vista: POWER takes about
20 min at 2 workers (POWER returns HTTP 429 at 4 workers). NSRDB takes about
30 min per 4 years at 4 parallel years. A one-year store builds in 26 s.

Do not `pkill -f <pattern>` over ssh when the same command line also contains
the pattern. It kills its own shell.

Smoke test the whole chain offline on the synthetic store, in about 2 minutes on CPU:

```bash
export DOWNSCALE_CONFIG_data=configs/data_synthetic.yaml
python -m data.synthetic && python -m scripts.run_pipeline --smoke
```

Outputs go to `runs/<store_root>/{xgb_A,xgb_B,swin_A,swin_B,swin_A_q,eval}/`.
The `eval/metrics.csv` file has one row per model × cell (temporal / spatial /
both) × stratum (sky, season, terrain, coast). Each row reports RMSE, MBE,
nRMSE, r, KGE, and skill against B0 and B3. The same folder holds
`maps.png/npz`, `spectra.json`, `stations.csv` and `stations_summary.csv`.
The last file includes the model-vs-station minus model-vs-NSRDB gap, which is
the target-quality gap.

## Leakage controls (plan §6), and where they live

- Climatologies (`kt_clim_fine`, B3 ratio, POWER kt anomaly) and the Swin
  normaliser are fit on **training years × training pixels** only. Inside the
  holdout + buffer the fine climatology is filled by Gaussian normalised
  convolution from training pixels. `tests/test_features.py::test_holdout_target_never_leaks`
  scrambles every holdout and non-train target, then asserts that every
  feature and the B3 map stay bit-identical.
- XGBoost rows come from training years × training pixels. The subsample is
  stratified on the **POWER (input)** sky class, never on the target's.
- Swin crops must lie entirely outside holdout + buffer. On a domain too small
  for that (the synthetic one), crops are drawn anywhere and the exclusion zone
  is masked from the loss, with a warning.
- Stations are never used for training, early stopping or selection.
- Track B excludes GOES-derived predictors, because NSRDB is built from GOES clouds.

## Structure and texture (2026-10-22)

The goal is texture **in the right place**, not just texture that looks
realistic. `evaluation/structure.py` scores SSIM, residual correlation against
B1, and, per scale band, both amplitude (sd ratio) and placement (correlation).

- **SSIM loss term (now the Swin default, weight 0.2).** In-region it is free:
  SSIM rises from 0.695 to 0.715 with RMSE unchanged, and fine-band placement
  roughly doubles. On the unseen holdout it costs 6–11% RMSE. That trade was
  accepted.
- **AMSE was rejected.** It adds amplitude without placement, so SSIM and
  residual correlation drop.
- **Post-hoc band-amplitude gains were rejected** (now opt-in with
  `evaluate --amp`). Every gain hit its clip and SSIM fell from 0.41 to 0.29.
- **What placed fine texture needs.** Correlation of each input's cloud
  signal with NSRDB GHI, per band, over 2 × 2 weeks in 2019:

  | input | <30 km | 30–60 | 60–120 | 120–240 |
  |---|---|---|---|---|
  | MODIS, Terra+Aqua (2 snapshots/day) | 0.03–0.04 | 0.04–0.07 | 0.11–0.16 | 0.26–0.31 |
  | GOES-16 cloud mask, 48 scans/day, sunlight-weighted | 0.04–0.06 | 0.06–0.10 | 0.13–0.23 | 0.26–0.39 |

  Even GOES, the satellite NSRDB itself is built from, explains little of
  NSRDB's sub-60 km day-to-day texture. A binary mask carries no optical
  depth, and part of that texture may not be real cloud signal. The ground
  stations decide which.
- **Track C = Track B + GOES cloud mask** (2018+, `configs/data_vista_goes.yaml`,
  with A/B/C trained on the same window). It is flagged as circular with
  NSRDB and judged at **USCRN** stations (7 in-domain, including Austin 33 NW
  inside the holdout) by `nsrdb_mimicry`. That is corr(model − station,
  NSRDB − station), reported alongside skill against B1 measured at both
  stations and NSRDB.
- **Ground truth matters here:** NSRDB reads 0.52 kWh/m²/day (~12%) higher
  than USCRN Austin 33 NW in 2020 (r = 0.985).

### Station results (USCRN, 2020–21, 7 stations, 4,723 station-days)

Every product reads high against USCRN: POWER +0.35, NSRDB +0.48, and the
models about +0.5 kWh/m²/day, because they inherit NSRDB's bias. That bias
cannot be settled from this data. The comparison that can is bias-removed
(centred) skill against B1, with block-bootstrap 90% confidence intervals:

| | all stations | Austin (holdout) |
|---|---|---|
| NSRDB | +9% [+1, +16] | −16% [−32, −3] |
| main window, Swin B + MODIS | **+1% [−4, +6]** | **+6% [−6, +16]** |
| main window, Swin A | −4% [−8, −1] | −1% |
| GOES era, Swin A / B / C | −13% / −6% / −2% | −13% / −8% / −7% |

- **Training on NSRDB can make day-to-day agreement with the ground worse
  than bilinear POWER.** This is strongest with short training (the 18-month
  GOES era) and without satellite cloud inputs. It happens in Track A too, so
  it is not GOES circularity.
- **Satellite clouds move station skill the right way** (A → B → C).
  GOES-era Swin C is the best of its window against both NSRDB (+45%) and the
  stations (−2%, not significant). Its NSRDB mimicry is also the highest
  (0.75), so it should be reported as flagged, not as clean downscaling.
- **Target swap (ERA5-Land, `configs/data_era5l.yaml`):** the same Swin B +
  MODIS trained on ERA5-Land instead of NSRDB scores −46% [−53, −39] at the
  stations (vs +1% for the NSRDB-trained model). Its errors swing to
  ERA5-Land's: mimicry with NSRDB falls from 0.70 to 0.42, and with ERA5-Land
  it rises from 0.41 to 0.75. ERA5-Land itself is −128% day to day at the
  stations. Models inherit their target's errors, so the target sets the
  ceiling.
- **Best overall: main-window Swin B + MODIS.** It is the best on every NSRDB
  metric (SSIM 0.49, residual correlation 0.45, fine-band placement ~10×)
  and the only model at or above B1 at the stations.

## Not done / needs the user

- **Track B MODIS inputs** (MOD/MYD06 cloud fraction, MAIAC AOD) need an
  Earthdata/AppEEARS download. Point `sources.dynamic.*.path` at the
  resulting netCDF stacks. Until then Track B = Track A + POWER PW/AOD/cloud
  (coarse).
- **Oklahoma Mesonet bulk download is blocked from Vista.** On 2026-10-22,
  4 parallel workers tripped the Mesonet CloudFront limit after about 350
  files, and the Vista IP then got HTTP 403 on every request (the workstation
  still got 200). The downloader is now serial with 1 s spacing and stops
  after 20 consecutive 403s. For 2019–21 (50 stations), ask the Mesonet for
  bulk data, or run the serial downloader from another host.
- **Swin config differs from the temperature runs:** window 4 with 3 stages,
  32 px crops. The no-straddle rule leaves no 48 px crop outside the 1°
  buffer on this grid (`configs/model.yaml` explains why).
- **TexMesonet / West Texas Mesonet** have no stable bulk API to target. Add
  exports through `sources.stations.generic_csv`. The Austin holdout has no
  Oklahoma Mesonet station, so its station validation depends on these
  networks.
- Phase 7 (DNI/DHI/LW, a 4 km stage) is not started. DNI/DHI need
  `sources.nsrdb.variables: [ghi, dni, dhi]` and new targets. Hourly would
  change the problem.
- First full real run: 2026-10-22 on Vista (all models, all three test
  cells). Station validation in it is empty: the Mesonet block left no
  station data in the 2020–21 test years.

# Target swap: the production configuration (Track B + MODIS; Swin with SSIM)
# plus Track A, trained against ERA5-Land instead of NSRDB.
#   DATA_CFG=configs/data_vista_era5l.yaml bash slurm/backfill_chain.sh slurm/tasks_era5l.sh
R=runs/data_store_era5l
TASKS=(
  "exA|xgbx --track A|$R/xgb_A/meta.json|00:50:00|3"
  "exB|xgbx --track B|$R/xgb_B/meta.json|00:50:00|3"
  "esB|swinx --track B|$R/swin_B/meta.json|00:35:00|15"
)

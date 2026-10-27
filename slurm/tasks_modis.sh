# Track B with MODIS cloud fraction (main window, 2001-2018 train), SSIM-default Swin.
#   DATA_CFG=configs/data_vista.yaml bash slurm/backfill_chain.sh slurm/tasks_modis.sh
R=runs/data_store
TASKS=(
  "xgbBm|xgbx --track B --tag modis|$R/xgb_B_modis/meta.json|00:50:00|3"
  "swinBm|swinx --track B --tag modis|$R/swin_B_modis/meta.json|00:35:00|15"
)

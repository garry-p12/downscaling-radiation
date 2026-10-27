# Method experiments on the main window (Swin, production settings: Track B +
# MODIS, SSIM loss term). B = clear-sky index normaliser; D = temporal context.
#   bash slurm/backfill_chain.sh slurm/tasks_methods.sh
R=runs/data_store
TASKS=(
  "sBcs|with-train configs/train_clearsky.yaml swinx --track B --tag modis|${R}_clearsky/swin_B_modis/meta.json|00:35:00|20"
  "sBT|swinx --track B+T --tag modis|$R/swin_B_T_modis/meta.json|00:35:00|20"
)

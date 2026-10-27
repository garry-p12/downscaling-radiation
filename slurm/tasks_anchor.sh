# Swin B + MODIS trained on each fold's station-anchored target (method A2).
# Needs slurm/anchor_targets.sh to have finished. One chain per fold, because
# DATA_CFG is per chain:
#   DATA_CFG=configs/data_vista_corr_a.yaml FOLD=a bash slurm/backfill_chain.sh slurm/tasks_anchor.sh
R=runs/data_store_corr_${FOLD}
TASKS=(
  "aS${FOLD}|swinx --track B --tag modis|$R/swin_B_modis/meta.json|00:35:00|20"
  "aX${FOLD}|xgbx --track B --tag modis|$R/xgb_B_modis/meta.json|00:50:00|3"
)

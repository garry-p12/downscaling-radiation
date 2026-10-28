# 5 km grid, GOES window: XGBoost and Swin B + cloud optical depth on one target.
# One chain per target (NSRDB, v1 folds, v2 folds):
#   DATA_CFG=configs/data_vista_5km.yaml          bash slurm/backfill_chain.sh slurm/tasks_5km.sh
#   DATA_CFG=configs/data_vista_5km_corr2_a.yaml  bash slurm/backfill_chain.sh slurm/tasks_5km.sh
# Swin uses configs/train_5km.yaml (64-cell crops = the 320 km of 32 cells at 10 km).
R=runs/$(basename "${DATA_CFG}" .yaml | sed 's/^data_vista/data_store/')
T=$(basename "$R" | sed 's/^data_store_5km//; s/^_//; s/corr2_/2/; s/corr_/1/')
TASKS=(
  "x5${T:-n}|xgbx --track B+cod|$R/xgb_B_cod/meta.json|00:50:00|3"
  "s5${T:-n}|with-train configs/train_5km.yaml swinx --track B+cod|$R/swin_B_cod/meta.json|00:35:00|25"
)

# GOES cloud optical depth inputs (the "+cod" modifier), GOES window.
#   DATA_CFG=configs/data_vista_goes.yaml bash slurm/backfill_chain.sh slurm/tasks_cod.sh
R=runs/data_store_goes
TASKS=(
  "sCcod|swinx --track C+cod|$R/swin_C_cod/meta.json|00:35:00|20"
)
# Reference for the anchored folds: B+cod trained on NSRDB itself.
#   DATA_CFG=configs/data_vista_goes.yaml TASKS_EXTRA=1 bash slurm/backfill_chain.sh slurm/tasks_cod.sh
[ -n "${TASKS_EXTRA:-}" ] && TASKS=("sBcod|swinx --track B+cod|$R/swin_B_cod/meta.json|00:35:00|20")

# Method A + COD: Swin B+cod on each fold's station-anchored target, GOES window.
#   DATA_CFG=configs/data_vista_goes_corr_a.yaml FOLD=a bash slurm/backfill_chain.sh slurm/tasks_anchor_cod.sh
R=runs/data_store_goes_corr_${FOLD}
TASKS=(
  "gS${FOLD}|swinx --track B+cod|$R/swin_B_cod/meta.json|00:35:00|20"
)

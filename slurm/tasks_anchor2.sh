# Swin on the v2 anchored targets (needs slurm/anchor_targets2.sh). One chain per fold and window:
#   DATA_CFG=configs/data_vista_goes_corr2_a.yaml FOLD=a bash slurm/backfill_chain.sh slurm/tasks_anchor2.sh
R=runs/$(basename "${DATA_CFG}" .yaml | sed 's/^data_vista/data_store/')
TASKS=(
  "v2${FOLD}$(basename "$R" | cut -c12-15)|swinx --track ${TRACK:-B+cod}|$R/swin_$(echo ${TRACK:-B+cod} | tr '+' '_')/meta.json|00:35:00|20"
)

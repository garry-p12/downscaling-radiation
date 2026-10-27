# Residual diffusion (method C) on the production mean model (Swin B + MODIS).
#   bash slurm/backfill_chain.sh slurm/tasks_diffusion.sh
# then:  sbatch -A ATM23014 -p gh -t 01:30:00 -J rad_prob slurm/stage.sh prob --runs diff_B
R=runs/data_store
TASKS=(
  "dB|diffx --mean-run swin_B_modis --tag B|$R/diff_B/meta.json|00:35:00|40"
)

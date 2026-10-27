# Texture arms for slurm/backfill_chain.sh:
#   bash slurm/backfill_chain.sh slurm/tasks_texture.sh
# Same data, splits and architecture as swin_A / swin_B; only the loss changes.
R=runs/data_store
TASKS=(
  "swinAamse|swinx --track A --pixel-loss amse --tag amse|$R/swin_A_amse/meta.json|00:35:00|15"
  "swinAssim|swinx --track A --ssim-weight 0.2 --tag ssim|$R/swin_A_ssim/meta.json|00:35:00|15"
  "swinAamss|swinx --track A --pixel-loss amse --ssim-weight 0.2 --tag amse_ssim|$R/swin_A_amse_ssim/meta.json|00:35:00|15"
  "swinBamse|swinx --track B --pixel-loss amse --tag amse|$R/swin_B_amse/meta.json|00:35:00|15"
)

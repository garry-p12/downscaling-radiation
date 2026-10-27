#!/bin/bash
# Build both station-anchored targets (method A2) in one job:
#   sbatch -A ATM23014 -p gh -t 02:00:00 -J rad_anchor slurm/anchor_targets.sh
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
A="TX_Austin_33_NW TX_Monahans_6_ENE TX_Palestine_6_WNW TX_Port_Aransas_32_NNE"
B="TX_Bronte_11_NNE TX_Muleshoe_19_S TX_Panther_Junction_2_N"
bash slurm/stage.sh anchor build-target --exclude $A --fold a
bash slurm/stage.sh anchor build-target --exclude $B --fold b
touch logs/anchor_targets.done

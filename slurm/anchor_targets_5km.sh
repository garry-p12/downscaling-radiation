#!/bin/bash
# Station-anchored targets on the 5 km store: error models fit on 2001-2018
# station-days like at 10 km, applied from 2018 (the GOES-window years).
#   bash slurm/anchor_targets_5km.sh            (login node, or sbatch)
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export DATA_CFG=configs/data_vista_5km_main.yaml
A="TX_Austin_33_NW TX_Monahans_6_ENE TX_Palestine_6_WNW TX_Port_Aransas_32_NNE"
B="TX_Bronte_11_NNE TX_Muleshoe_19_S TX_Panther_Junction_2_N"
S=2018-01-01
bash slurm/stage.sh anchor build-target --exclude $A --fold a --fit-net uscrn --start $S
bash slurm/stage.sh anchor build-target --exclude $B --fold b --fit-net uscrn --start $S
bash slurm/stage.sh anchor build-target --exclude $(cat slurm/fold2_a.txt) --fold 2a --start $S
bash slurm/stage.sh anchor build-target --exclude $(cat slurm/fold2_b.txt) --fold 2b --start $S
touch logs/anchor_targets_5km.done

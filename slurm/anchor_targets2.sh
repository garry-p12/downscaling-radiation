#!/bin/bash
# Station-anchored targets v2 (both networks, two interleaved folds):
#   sbatch -A ATM23014 -p gh-dev -t 02:00:00 -J rad_anchor2 slurm/anchor_targets2.sh
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
bash slurm/stage.sh anchor build-target --exclude $(cat slurm/fold2_a.txt) --fold 2a
bash slurm/stage.sh anchor build-target --exclude $(cat slurm/fold2_b.txt) --fold 2b
touch logs/anchor_targets2.done

#!/bin/bash
# Monte Carlo drop-path samples (evaluation/uncertainty.py sample) for several
# run groups in one job; groups whose mc_*.npz exists are skipped, so a
# timed-out job can simply be resubmitted.
#   RES=5km TARGETS="corr2_a corr2_b corr_a corr_b" sbatch -A ATM23014 -p gh-dev -t 02:00:00 -J rad_uq slurm/uq_sample.sh
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
RES=${RES:-5km}
for c in ${TARGETS:-corr2_a corr2_b corr_a corr_b}; do
  export DATA_CFG=configs/data_vista_${RES}_$c.yaml
  R=runs/data_store_${RES}_$c
  [ -f $R/eval/mc_swin_B_cod.npz ] || bash slurm/stage.sh py -m evaluation.uncertainty sample --members ${MEMBERS:-20}
done

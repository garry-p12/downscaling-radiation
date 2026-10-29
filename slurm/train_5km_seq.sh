#!/bin/bash
# All 5 km anchored-target arms in ONE job, for gh-dev (1 running job per
# user, 2 h) when gh is jammed: XGBoost, Swin, then eval per target. Steps
# whose marker exists are skipped, so a timed-out job can simply be resubmitted.
#   sbatch -A ATM23014 -p gh-dev -t 02:00:00 -J rad_t5seq slurm/train_5km_seq.sh
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MAXMIN=${MAXMIN:-25}
for c in corr_a corr_b corr2_a corr2_b; do
  export DATA_CFG=configs/data_vista_5km_$c.yaml
  R=runs/data_store_5km_$c
  [ -f $R/xgb_B_cod/meta.json ] || bash slurm/stage.sh xgbx --track B+cod
  [ -f $R/swin_B_cod/meta.json ] || bash slurm/stage.sh with-train configs/train_5km.yaml swinx --track B+cod
  [ -f $R/eval/stations_summary.csv ] || bash slurm/stage.sh eval
done

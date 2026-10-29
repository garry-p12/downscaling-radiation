#!/bin/bash
# Several anchored-target arms in ONE job, for gh-dev (1 running job per user,
# 2 h) when gh is jammed: XGBoost, Swin, then eval per target. Steps whose
# marker exists are skipped, so a timed-out job can simply be resubmitted.
#   RES=5km|1km   TARGETS="nsrdb corr_a corr_b corr2_a corr2_b" (nsrdb = the base config)
#   sbatch -A ATM23014 -p gh-dev -t 02:00:00 -J rad_t5seq slurm/train_seq.sh
#   RES=1km TARGETS="nsrdb corr2_a" sbatch ... -J rad_t1seq slurm/train_seq.sh
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MAXMIN=${MAXMIN:-25}
RES=${RES:-5km}
for c in ${TARGETS:-corr_a corr_b corr2_a corr2_b}; do
  if [ "$c" = nsrdb ]; then
    export DATA_CFG=configs/data_vista_$RES.yaml; R=runs/data_store_$RES
  else
    export DATA_CFG=configs/data_vista_${RES}_$c.yaml; R=runs/data_store_${RES}_$c
  fi
  [ -f $R/xgb_B_cod/meta.json ] || bash slurm/stage.sh xgbx --track B+cod
  [ -f $R/swin_B_cod/meta.json ] || bash slurm/stage.sh with-train configs/train_$RES.yaml swinx --track B+cod
  # eval only once both models are complete (Swin may stop at MAXMIN and resume next job)
  [ -f $R/xgb_B_cod/meta.json ] && [ -f $R/swin_B_cod/meta.json ] || continue
  [ -f $R/eval/stations_summary.csv ] || bash slurm/stage.sh eval
done

#!/bin/bash
# Compute-node stages. Submit with the account/partition used in the
# temperature project:
#
#   sbatch -A ATM23014 -p gh -t 01:00:00 -J rad_build slurm/stage.sh build
#   sbatch ... -J rad_xgb   slurm/stage.sh xgb
#   sbatch ... -J rad_swin  slurm/stage.sh swin A      (or B, or Aq)
#   sbatch ... -J rad_eval  slurm/stage.sh eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export DOWNSCALE_CONFIG_data=${DATA_CFG:-configs/data_vista.yaml}
PY="${WORK}/miniforge3/envs/dsc/bin/python"
# Optional prefix:  with-train configs/train_clearsky.yaml <stage> ...
if [ "${1:-}" = "with-train" ]; then
  export DOWNSCALE_CONFIG_train="$2"; shift 2
fi
echo "=== $(hostname) stage=$* train=${DOWNSCALE_CONFIG_train:-default} $(date)"
case "$1" in
  build)
    $PY -u -m data.build_store && $PY -u -m data.check_discontinuity
    $PY -u -m data.stations build || echo "!!! stations build failed (no station data?)"
    ;;
  xgbx)
    shift; $PY -u -m training.train_xgb "$@" ;;
  xgb)
    if [ -n "${2:-}" ]; then
      $PY -u -m training.train_xgb --track "$2"
    else
      $PY -u -m training.train_xgb --track A && $PY -u -m training.train_xgb --track B
    fi ;;
  swin)
    # MAXMIN: stop cleanly before the job limit; train_swin resumes from last.pt.
    MM=(); [ -n "${MAXMIN:-}" ] && MM=(--max-minutes "$MAXMIN")
    case "${2:-A}" in
      A)  $PY -u -m training.train_swin --track A "${MM[@]}" ;;
      B)  $PY -u -m training.train_swin --track B "${MM[@]}" ;;
      Aq) $PY -u -m training.train_swin --track A --quantiles 0.1 0.5 0.9 "${MM[@]}" ;;
    esac ;;
  swinx)
    # Arbitrary train_swin arguments, e.g.  swinx --track A --pixel-loss amse --tag amse
    shift; MM=(); [ -n "${MAXMIN:-}" ] && MM=(--max-minutes "$MAXMIN")
    $PY -u -m training.train_swin "$@" "${MM[@]}" ;;
  diffx)
    # Residual diffusion (method C), resumable:  diffx --mean-run swin_B_modis --tag B
    shift; MM=(); [ -n "${MAXMIN:-}" ] && MM=(--max-minutes "$MAXMIN")
    $PY -u -m training.train_diffusion "$@" "${MM[@]}" ;;
  anchor)
    # Station anchoring (method A):  anchor build-target --exclude ... --fold a
    shift; $PY -u -m evaluation.station_anchor "$@" ;;
  prob)
    shift; $PY -u -m evaluation.probabilistic "$@" ;;
  eval)
    $PY -u -m evaluation.evaluate ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
rc=$?
echo "=== exit $rc $(date)"
exit $rc

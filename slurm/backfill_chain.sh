#!/bin/bash
# Backfill-friendly driver, run on the vista2 login node under nohup.
#
# Short jobs fit the gaps the scheduler leaves before big reservations; 2-3 h
# jobs sit behind them. So every training task is submitted as a SHORT job and
# resubmitted until its completion marker (meta.json) exists: Swin resumes
# from last.pt (train_swin --max-minutes), XGBoost runs one track per job.
# When all training is done (or a task exhausts its attempts) eval is submitted.
#
#   ssh vista2 'cd $WORK/downscaling-radiation && setsid nohup bash slurm/backfill_chain.sh > logs/backfill.log 2>&1 < /dev/null &'
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS=runs/data_store
export DATA_CFG=${DATA_CFG:-configs/data_vista.yaml}
ACC="-A ATM23014"
PART=${PART:-gh}
sb_() { sbatch --parsable "$@" 2>/dev/null | tail -n 1; }

# name | stage args | marker | time limit | max attempts
# A task file (bash defining TASKS=(...)) may be given as $1 to run other arms.
TASKS=(
  "xgbA|xgb A|$RUNS/xgb_A/meta.json|00:50:00|3"
  "xgbB|xgb B|$RUNS/xgb_B/meta.json|00:50:00|3"
  "swinA|swin A|$RUNS/swin_A/meta.json|00:35:00|15"
  "swinB|swin B|$RUNS/swin_B/meta.json|00:35:00|15"
  "swinAq|swin Aq|$RUNS/swin_A_q/meta.json|00:35:00|15"
)
[ -n "${1:-}" ] && source "$1"
declare -A tries

queued() { squeue -u "$USER" -h -n "rad_$1" | grep -q .; }

while true; do
  pending=0
  for t in "${TASKS[@]}"; do
    IFS='|' read -r name args marker limit maxn <<< "$t"
    [ -f "$marker" ] && continue
    n=${tries[$name]:-0}
    if [ "$n" -ge "$maxn" ]; then queued "$name" && pending=1; continue; fi
    pending=1
    queued "$name" && continue
    tries[$name]=$((n + 1))
    # shellcheck disable=SC2086
    id=$(sb_ $ACC -p "$PART" -t "$limit" -J "rad_$name" \
         --export=ALL,MAXMIN=25 slurm/stage.sh $args)
    echo "$(date +%H:%M:%S) submit $name attempt $((n + 1))/$maxn -> $id"
  done
  [ "$pending" -eq 0 ] && break
  sleep 120
done

for t in "${TASKS[@]}"; do
  IFS='|' read -r name _ marker _ _ <<< "$t"
  [ -f "$marker" ] && echo "done: $name" || echo "!!! gave up: $name"
done
[ -n "${NO_EVAL:-}" ] && exit 0
e=$(sb_ $ACC -p "$PART" -t 01:00:00 -J rad_eval slurm/stage.sh eval)
echo "$(date +%H:%M:%S) submit eval -> $e"
while queued eval; do sleep 60; done
echo "eval finished $(date)"; tail -5 logs/rad_eval_"$e".out

#!/bin/bash
# vista2 login node, under nohup: wait for the three ERA5-Land fetch processes,
# add era5l_ghi to the store, launch the target-swap training chain (+ eval).
set -uo pipefail
cd "$(dirname "$0")/.."
export DOWNSCALE_CONFIG_data=configs/data_vista.yaml OMP_NUM_THREADS=4
PY="nice -n 10 ${WORK}/miniforge3/envs/dsc/bin/python -W ignore -u"
echo "=== $(date +%H:%M:%S) waiting for ERA5-Land"
until [ "$(grep -l 'fetch done' logs/dl_era5l_*.log 2>/dev/null | wc -l)" -ge 3 ]; do sleep 300; done
grep -h "fetch done" logs/dl_era5l_*.log
n=$(ls /scratch/11755/gurup12/downscaling-radiation/data_store/raw/era5_land/era5l_ssrd_*.nc | wc -l)
echo "monthly files: $n / 252"
[ "$n" -lt 252 ] && { echo "!!! incomplete; re-run the fetch, then this script"; exit 1; }
echo "=== $(date +%H:%M:%S) add to store"
$PY -m data.download_era5_land add-to-store || { echo "!!! add failed"; exit 1; }
echo "=== $(date +%H:%M:%S) launching chain"
DATA_CFG=configs/data_vista_era5l.yaml setsid nohup bash slurm/backfill_chain.sh slurm/tasks_era5l.sh \
  > logs/backfill_era5l.log 2>&1 < /dev/null &
echo "=== done"

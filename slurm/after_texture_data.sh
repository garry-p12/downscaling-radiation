#!/bin/bash
# vista2 login node, under nohup. Waits for the USCRN, GOES and MODIS downloads,
# adds their products to the store IN SEQUENCE (both append to the same zarr),
# then launches the two training chains:
#   main window : Track B + MODIS                      (slurm/tasks_modis.sh)
#   GOES era    : Tracks A, B, C on 2018-19 train      (slurm/tasks_goes.sh)
# each ending in an eval with the station / NSRDB-mimicry check.
#
#   setsid nohup bash slurm/after_texture_data.sh > logs/texture_data.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
export DOWNSCALE_CONFIG_data=configs/data_vista.yaml
PY="nice -n 10 ${WORK}/miniforge3/envs/dsc/bin/python -W ignore -u"
export OMP_NUM_THREADS=4
step() { echo "=== $(date +%H:%M:%S) $*"; }

step "waiting for USCRN"
until grep -q "USCRN: fetched" logs/dl_uscrn.log 2>/dev/null; do sleep 120; done
step "stations build"; $PY -m data.stations build || echo "!!! stations build failed"

step "waiting for GOES days"
until grep -q "days done" logs/dl_goes.log 2>/dev/null; do sleep 300; done
step "GOES assemble + add"; $PY -m data.goes_cloud assemble && $PY -m data.goes_cloud add-to-store \
  || { echo "!!! GOES add failed"; exit 1; }

step "waiting for MODIS"
until grep -q "all downloaded" logs/dl_modis.log 2>/dev/null; do sleep 300; done
step "MODIS aggregate + add"; $PY -m data.modis_cloud aggregate && $PY -m data.modis_cloud add-to-store \
  || { echo "!!! MODIS add failed"; exit 1; }

step "launching training chains"
DATA_CFG=configs/data_vista.yaml setsid nohup bash slurm/backfill_chain.sh slurm/tasks_modis.sh \
  > logs/backfill_modis.log 2>&1 < /dev/null &
DATA_CFG=configs/data_vista_goes.yaml setsid nohup bash slurm/backfill_chain.sh slurm/tasks_goes.sh \
  > logs/backfill_goes.log 2>&1 < /dev/null &
step "done"

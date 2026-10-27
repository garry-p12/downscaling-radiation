#!/bin/bash
# Phase 1-2 downloads, run on the vista2 LOGIN node under nohup (compute nodes
# have no guaranteed outbound internet; the user asked for vista2):
#
#   ssh vista2 'cd $WORK/downscaling-radiation && nohup bash slurm/downloads.sh > logs/downloads.log 2>&1 &'
#
# Every step skips files that already exist, so re-running resumes.
# Parallelism is deliberately modest: login nodes cap user processes (login2
# once ran out of slots), and POWER rate-limits per client.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export DOWNSCALE_CONFIG_data=configs/data_vista.yaml
PY="${WORK}/miniforge3/envs/dsc/bin/python"
YEARS=$(seq 2001 2021)
echo "=== $(hostname) start $(date)"

( $PY -u -m data.download_power --years $YEARS --workers 2 > logs/dl_power.log 2>&1
  echo "power exit $? $(date)" >> logs/dl_power.log ) &
( $PY -u -m data.stations download-mesonet --years 2019 2020 2021 > logs/dl_mesonet.log 2>&1
  echo "mesonet exit $? $(date)" >> logs/dl_mesonet.log ) &
$PY -u -m data.download_nsrdb --years $YEARS --workers 4 > logs/dl_nsrdb.log 2>&1
echo "nsrdb exit $? $(date)" >> logs/dl_nsrdb.log
wait
echo "=== done $(date)"

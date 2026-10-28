#!/bin/bash
# Native statics on Vista: DEM mosaic, then the 10 km check grid and the 5 km grid.
#   sbatch -A ATM23014 -p gh-dev -t 01:30:00 -J rad_statics slurm/statics_5km.sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
cd "$(dirname "$0")/.." 2>/dev/null || cd "$SLURM_SUBMIT_DIR"
PY=$WORK/miniforge3/envs/dsc/bin/python
RAW=/scratch/11755/gurup12/downscaling-radiation/data_store/raw
SCR=/scratch/11755/gurup12/downscaling-radiation
[ -f $RAW/dem/dem_3s.tif ] || $PY -m data.statics $RAW/dem/dem_3s.tif --mosaic-dem $RAW/dem/tiles
DOWNSCALE_CONFIG_data=configs/data_vista_native10.yaml $PY -m data.statics $SCR/statics_native10.npz
DOWNSCALE_CONFIG_data=configs/data_vista_5km.yaml $PY -m data.statics $SCR/statics_5km.npz
echo "=== done $(date)"

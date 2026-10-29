#!/bin/bash
# 1 km store on Vista (configs/data_vista_1km.yaml), resumable: every step is
# skipped when its output exists, so a timed-out job is simply resubmitted.
#   statics -> NSRDB store -> stations -> MODIS -> GOES (needs the daily files
#   from `data.goes_cloud days`, run on vista2) -> anchored targets from 5 km
#
#   sbatch -A ATM23014 -p gh-dev -t 02:00:00 -J rad_build1 slurm/build_1km.sh
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export DOWNSCALE_CONFIG_data=configs/data_vista_1km.yaml
PY=$WORK/miniforge3/envs/dsc/bin/python
SCR=/scratch/11755/gurup12/downscaling-radiation
Z=$SCR/data_store_1km/radiation.zarr
have() { [ -f "$Z/$1/zarr.json" ]; }
step() { echo "=== $* $(date +%H:%M:%S)"; }

[ -f $SCR/statics_1km.npz ] || { step statics; $PY -u -m data.statics $SCR/statics_1km.npz; }
have nsrdb_ghi || { step store; $PY -u -m data.build_store; }
[ -f $SCR/data_store_1km/raw/stations/stations.csv ] || {
  step stations; $PY -u -m data.stations regrid-meta --from $SCR/data_store/raw/stations; }
have dyn_modis_cf || { step modis; $PY -u -m data.modis_cloud aggregate && $PY -u -m data.modis_cloud add-to-store; }
for p in acm cod; do
  v=$([ $p = acm ] && echo dyn_goes_cf || echo dyn_goes_tcld)
  have $v && continue
  n=$(ls $SCR/data_store_1km/raw/$([ $p = acm ] && echo goes || echo goes_cod)/daily 2>/dev/null | wc -l)
  [ "$n" -ge 1461 ] || { echo "!!! GOES $p: $n/1461 daily files, run data.goes_cloud days first"; exit 3; }
  step goes $p; $PY -u -m data.goes_cloud --product $p assemble && $PY -u -m data.goes_cloud --product $p add-to-store
done
for f in a b 2a 2b; do
  have nsrdb_ghi_corr_$f || { step transfer $f
    $PY -u -m evaluation.station_anchor transfer-target --from $SCR/data_store_5km/radiation.zarr --fold $f; }
done
step done

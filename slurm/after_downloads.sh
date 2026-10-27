#!/bin/bash
# Runs on the vista2 login node under nohup: waits for the NSRDB and POWER
# downloads to finish, checks they are complete, then submits the compute
# chain:   build -> {xgb, swin A, swin B, swin Aq} -> eval
#
#   ssh vista2 'cd $WORK/downscaling-radiation && setsid nohup bash slurm/after_downloads.sh > logs/chain.log 2>&1 < /dev/null &'
set -uo pipefail
cd "$(dirname "$0")/.."
RAW=/scratch/11755/gurup12/downscaling-radiation/data_store/raw
ACC="-A ATM23014 -p gh"
NYEARS=21
PY="${WORK}/miniforge3/envs/dsc/bin/python"
# params x years x tiles, from the same tiling the downloader used.
EXPECT_POWER=$(DOWNSCALE_CONFIG_data=configs/data_vista.yaml $PY -c "
from common import load_config; from data.download_power import tiles
c = load_config('data'); s = c['sources']['power']
print(len(s['parameters']) * $NYEARS * len(tiles(c['domain'], float(s['pad_deg']))))")

until grep -q "^nsrdb exit" logs/dl_nsrdb.log 2>/dev/null && grep -q "^power exit" logs/dl_power.log 2>/dev/null; do
    sleep 300
done
echo "downloads finished $(date)"
grep "exit" logs/dl_nsrdb.log logs/dl_power.log

n_ns=$(ls $RAW/nsrdb/nsrdb_ghi_*.nc 2>/dev/null | wc -l)
n_pw=$(ls $RAW/power/*.nc 2>/dev/null | wc -l)
echo "nsrdb years: $n_ns/$NYEARS   power files: $n_pw/$EXPECT_POWER"
if [ "$n_ns" -lt "$NYEARS" ] || [ "$n_pw" -lt "$EXPECT_POWER" ]; then
    echo "!!! incomplete downloads; not submitting. Re-run slurm/power_only.sh or download_nsrdb, then this script."
    exit 1
fi

# TACC's sbatch wrapper prints a banner to STDOUT even with --parsable, so
# take only the last line (the job id).
sb_() { sbatch --parsable "$@" | tail -n 1; }
b=$(sb_ $ACC -t 01:00:00 -J rad_build slurm/stage.sh build)
x=$(sb_ $ACC -t 03:00:00 -J rad_xgb    --dependency=afterok:$b slurm/stage.sh xgb)
sa=$(sb_ $ACC -t 02:00:00 -J rad_swinA  --dependency=afterok:$b slurm/stage.sh swin A)
sb=$(sb_ $ACC -t 02:00:00 -J rad_swinB  --dependency=afterok:$b slurm/stage.sh swin B)
sq=$(sb_ $ACC -t 02:00:00 -J rad_swinAq --dependency=afterok:$b slurm/stage.sh swin Aq)
e=$(sb_ $ACC -t 01:00:00 -J rad_eval   --dependency=afterany:$x:$sa:$sb:$sq slurm/stage.sh eval)
echo "submitted build=$b xgb=$x swinA=$sa swinB=$sb swinAq=$sq eval=$e $(date)"

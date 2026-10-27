#!/bin/bash
# POWER only (resumes; skips existing files). For re-runs after rate limiting.
set -uo pipefail
cd "$(dirname "$0")/.."
export DOWNSCALE_CONFIG_data=configs/data_vista.yaml
PY="${WORK}/miniforge3/envs/dsc/bin/python"
$PY -u -m data.download_power --years $(seq 2001 2021) --workers 2
echo "power exit $? $(date)"

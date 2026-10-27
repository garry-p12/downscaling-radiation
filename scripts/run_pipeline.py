"""Run phases 3-6 on an existing store: discontinuity check, XGBoost and Swin
for both tracks (+ the M3 quantile model), then the full evaluation.

    python -m scripts.run_pipeline                       # real store
    DOWNSCALE_CONFIG_data=configs/data_synthetic.yaml \
        python -m scripts.run_pipeline --smoke           # minutes, CPU

--smoke shrinks XGBoost rows and Swin epochs and uses configs/model_tiny.yaml;
it checks that the code runs, not what the numbers are.

Each stage is a separate process: torch and xgboost each bring their own
OpenMP runtime, and loading both into one process segfaults on macOS.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(*args: str) -> None:
    cmd = [sys.executable, "-m", *args]
    print("\n$ " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def main(smoke: bool, skip_swin: bool) -> None:
    run("data.check_discontinuity")
    xgb_extra = ["--n-rows", "200000"] if smoke else []
    swin_extra = (["--epochs", "2", "--steps-per-epoch", "30",
                   "--model-config", "model_tiny"] if smoke else [])
    for track in ("A", "B"):
        run("training.train_xgb", "--track", track, *xgb_extra)
    if not skip_swin:
        for track in ("A", "B"):
            run("training.train_swin", "--track", track, *swin_extra)
        run("training.train_swin", "--track", "A", "--quantiles", "0.1", "0.5", "0.9",
            *swin_extra)
    run("evaluation.evaluate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-swin", action="store_true")
    a = ap.parse_args()
    main(a.smoke, a.skip_swin)

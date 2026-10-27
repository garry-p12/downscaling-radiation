"""Optional Weights & Biases tracking. No-op when disabled or wandb missing.

Vendored from downscaling-temperature@1d2c3bb (training/tracking.py); default project
name changed to downscaling-radiation.

Keeps the train loop clean: call tracker.log(...) unconditionally; if wandb is
off it does nothing. Enable via configs/train.yaml -> logging.wandb.enabled.
"""
from __future__ import annotations

import warnings


class Tracker:
    def __init__(self, wandb_cfg: dict, full_config: dict, run_meta: dict):
        self.run = None
        self._wandb = None
        if not wandb_cfg.get("enabled", False):
            return
        try:
            import wandb
        except ImportError:
            warnings.warn("wandb.enabled=true but wandb not installed "
                          "(`pip install wandb`); tracking disabled.",
                          RuntimeWarning)
            return
        # Broad except on purpose: tracking is optional, and a multi-hour
        # training run must not die because logging could not start. This also
        # catches the wandb/ RUN-OUTPUT directory in the repo root shadowing an
        # uninstalled wandb as a namespace package, which fails with
        # AttributeError rather than ImportError.
        try:
            self.run = wandb.init(
                project=wandb_cfg.get("project", "downscaling-radiation"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("run_name"),
                mode=wandb_cfg.get("mode", "online"),
                config={**full_config, **run_meta},
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"wandb.init failed ({e!r}); tracking disabled.",
                          RuntimeWarning)
            self.run = None
            return
        self._wandb = wandb

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def watch(self, model, wandb_cfg: dict) -> None:
        if self.enabled and wandb_cfg.get("watch_model", False):
            self._wandb.watch(model, log="all", log_freq=100)

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            self._wandb.log(metrics, step=step)

    def log_figure(self, key: str, fig, step: int | None = None) -> None:
        """Log a matplotlib Figure as a wandb image, then close it."""
        if self.enabled and fig is not None:
            self._wandb.log({key: self._wandb.Image(fig)}, step=step)
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def summary(self, key: str, value) -> None:
        if self.enabled:
            self.run.summary[key] = value

    def finish(self) -> None:
        if self.enabled:
            self._wandb.finish()

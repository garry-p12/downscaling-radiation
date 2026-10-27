"""Shared utilities: config, grids, solar geometry, daily aggregation, splits.

config / grid / normalize / tracking are vendored from downscaling-temperature
so both variables live on bit-identical grids with identical config semantics.
"""
from .config import load_config  # noqa: F401
from .grid import build_target_grid, subset_bbox  # noqa: F401
from .normalize import Normalizer  # noqa: F401

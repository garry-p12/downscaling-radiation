"""Shared training entry plumbing: configs, store, FeatureBuilder, run dirs."""
from __future__ import annotations

import json
from pathlib import Path

from common import load_config
from features.builder import FeatureBuilder


def load_all():
    dcfg = load_config("data")
    tcfg = load_config("train")
    return dcfg, tcfg


def builder(dcfg, tcfg) -> FeatureBuilder:
    """The split is re-derived from the DATA CONFIG, not read from the store, so
    a config with a different time window (e.g. the GOES era) reuses the same
    store without a rebuild."""
    from common.splits import assign_splits
    from data.build_store import open_store

    ds = open_store(dcfg)
    ds = ds.assign(split=("time", assign_splits(ds.time.values, dcfg["time"])))
    return FeatureBuilder(ds, tcfg, dcfg["dataset"].get("cache_dir"),
                          target_var=dcfg.get("target_var", "nsrdb_ghi"))


def runs_root(dcfg, tcfg) -> Path:
    """runs/<run_group>; run_group defaults to the store_root name, so synthetic
    and real runs never collide and an experiment group can be set per config."""
    group = dcfg.get("run_group") or Path(dcfg["store_root"]).name
    return Path(tcfg["logging"]["out_dir"]) / group


def run_dir(dcfg, tcfg, name: str) -> Path:
    d = runs_root(dcfg, tcfg) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))

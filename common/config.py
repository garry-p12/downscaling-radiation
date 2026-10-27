"""YAML config loading with light dot-access, inheritance and interpolation.

Vendored verbatim from downscaling-temperature@1d2c3bb (common/config.py) so the two
projects resolve configs identically. Keep in sync by hand.

Two features exist to keep the per-domain configs from being copy-pasted
forests:

``extends:``   the config names a parent (a stem or path); the parent is loaded
               first and the child deep-merged over it. Only the deltas that
               actually define a domain — bbox, store root, holdout — live in
               the domain file, so a change to the grid, the temporal split or
               the channel order is made once and applies everywhere.

``${key}``     string values may reference a TOP-LEVEL scalar of the same
               (merged) config. Used for ``store_root``, which is the one thing
               that varies across every ``out_dir`` in a domain file and was the
               single largest source of duplication.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MAX_EXTENDS_DEPTH = 8


class Cfg(dict):
    """dict with attribute access; nested dicts wrapped recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(val) if isinstance(val, dict) else val

    __setattr__ = dict.__setitem__  # type: ignore[assignment]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; ``override`` wins. Lists replace, never append.

    Lists replace because every list in these configs is an ordered contract
    (channel order, layer depths) — appending would silently corrupt it.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _interpolate(node: Any, top: dict) -> Any:
    """Substitute ``${key}`` in every string, resolving against top-level keys."""
    if isinstance(node, dict):
        return {k: _interpolate(v, top) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, top) for v in node]
    if isinstance(node, str):
        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key not in top:
                raise KeyError(f"config references ${{{key}}}, which is not a "
                               f"top-level key of the merged config")
            return str(top[key])
        return _VAR.sub(sub, node)
    return node


def _resolve_path(name: str) -> Path:
    """Config stem -> configs/<stem>.yaml, honouring the env override."""
    path = Path(name)
    if path.suffix:
        return path
    override = os.environ.get(f"DOWNSCALE_CONFIG_{name}")
    return Path(override) if override else CONFIG_DIR / f"{name}.yaml"


def _load_raw(path: Path, depth: int = 0) -> dict:
    """Load one YAML and splice in its ``extends:`` ancestry (parent first)."""
    if depth > _MAX_EXTENDS_DEPTH:
        raise RecursionError(f"extends chain too deep (cycle?) at {path}")
    with open(path) as f:
        node = yaml.safe_load(f) or {}
    parents = node.pop("extends", None)
    if not parents:
        return node
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        ppath = Path(parent)
        if not ppath.is_absolute() and not ppath.exists():
            # Resolve relative to the child's own directory, then configs/.
            cand = path.parent / parent
            ppath = cand if cand.exists() else _resolve_path(parent)
        merged = deep_merge(merged, _load_raw(ppath, depth + 1))
    return deep_merge(merged, node)


def load_config(name: str) -> Cfg:
    """Load a single config by stem ('data', 'model', 'train') or path.

    An environment variable ``DOWNSCALE_CONFIG_<name>`` overrides the file for
    that stem, so a second domain can be run without editing (and having to
    restore) the primary configs:

        DOWNSCALE_CONFIG_data=configs/data_austin.yaml python -m ...

    Every module calls ``load_config("data")`` by stem, so this redirects the
    whole pipeline consistently rather than per-call-site.
    """
    node = _load_raw(_resolve_path(name))
    return Cfg(_interpolate(node, node))


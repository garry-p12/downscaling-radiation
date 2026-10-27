"""Config inheritance resolves the synthetic override without recursion."""
from __future__ import annotations

from common.config import load_config
from tests.conftest import REPO_ROOT


def test_synthetic_extends_base(monkeypatch):
    monkeypatch.setenv("DOWNSCALE_CONFIG_data", str(REPO_ROOT / "configs/data_synthetic.yaml"))
    cfg = load_config("data")
    assert cfg["domain"]["name"] == "synth"
    assert cfg["sources"]["power"]["time_standard"] == "LST"     # inherited
    assert cfg["dataset"]["out_zarr"].startswith("data_store_synth")

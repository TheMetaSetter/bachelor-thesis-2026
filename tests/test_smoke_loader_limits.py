from __future__ import annotations

from src.core.config import load_experiment_config
from src.data.loaders import build_smd_dataloaders


def test_smoke_data_config_applies_split_window_limits() -> None:
    experiment_config = load_experiment_config("configs/experiment/smd_smoke_test.yaml")
    data_bundle = build_smd_dataloaders(experiment_config["data"])

    assert len(data_bundle["datasets"]["train"]) == 256
    assert len(data_bundle["datasets"]["val"]) == 128
    assert len(data_bundle["datasets"]["test"]) == 128

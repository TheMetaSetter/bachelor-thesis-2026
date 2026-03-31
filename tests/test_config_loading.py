from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_experiment_config


def test_load_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_vertical_slice.yaml")
    assert loaded_config["data"]["window_size"] == 100
    assert loaded_config["model"]["model_name"] == "reconstruction_mlp_ae"


def test_load_experiment_config_rejects_missing_required_keys(tmp_path: Path) -> None:
    invalid_experiment_path = tmp_path / "invalid_experiment.yaml"
    invalid_experiment_path.write_text("experiment_name: broken\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(invalid_experiment_path)

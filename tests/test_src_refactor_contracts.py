from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.core.config import load_yaml_config
from src.core.registry import MODEL_BUILDERS, clear_registry
from src.core.runtime_components import (
    register_offline_runtime_components,
    register_online_runtime_components,
)
from src.metrics.pointwise import compute_pointwise_metrics
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "src_refactor_contracts.json"


def _snapshot() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_registry_and_model_output_surface_match_snapshot() -> None:
    snapshot = _snapshot()
    clear_registry()
    register_offline_runtime_components()
    register_online_runtime_components()
    assert sorted(MODEL_BUILDERS) == snapshot["registry_names"]
    model = ReconstructionMLPAutoencoder(4, 8, 3)
    batch = {
        "x": torch.zeros(2, 5, 4),
        "point_labels": torch.zeros(2, 5, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{}, {}],
    }
    assert sorted(model(batch)) == snapshot["model_output_keys"]
    assert sorted(model.state_dict()) == snapshot["reconstruction_state_dict_keys"]


def test_config_key_trees_and_metric_surface_match_snapshot() -> None:
    snapshot = _snapshot()
    paths = {
        "offline_thesis": "configs/experiment/benchmark_smoke/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__smoke.yaml",
        "redlamp": "configs/experiment/comparative/baseline/smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml",
        "online": "configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml",
    }
    for name, path in paths.items():
        config = load_yaml_config(path)
        expected = snapshot["config_key_trees"][name]
        assert sorted(config) == expected["top_level"]
        for section, keys in expected.get("nested", {}).items():
            assert sorted(config[section]) == keys

    metrics = compute_pointwise_metrics(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), threshold=0.5
    )
    assert sorted(metrics) == snapshot["pointwise_metric_keys"]

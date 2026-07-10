from __future__ import annotations

from scripts.train import (
    build_model_from_experiment_config,
    register_runtime_components,
)
from src.models.redlamp_baseline import RedLampBaseline


def test_redlamp_baseline_constructs_from_training_runtime_with_data_window_size() -> (
    None
):
    experiment_config = {
        "data": {
            "window_size": 20,
        },
        "model": {
            "model_name": "redlamp_baseline",
            "input_dim": 4,
        },
        "task": {
            "task_name": "multitask_tsad",
            "classification_label_mode": "redlamp_multiclass",
            "train_balance_classes": True,
        },
    }

    register_runtime_components()
    model = build_model_from_experiment_config(experiment_config)

    assert isinstance(model, RedLampBaseline)
    assert model.window_size == 20

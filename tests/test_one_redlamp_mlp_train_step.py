from __future__ import annotations

import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline


def test_one_redlamp_mlp_train_step_backpropagates() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    step_output = model.training_step(batch)
    step_output["loss"].backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_redlamp_synthetic_validation_step_exposes_synthetic_anomaly_mask() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    model.prepare_synthetic_validation_epoch()
    step_output = model.synthetic_validation_step(batch)

    assert "synthetic_anomaly_mask" in step_output["batch"]
    assert step_output["batch"]["synthetic_anomaly_mask"].shape == step_output[
        "batch"
    ]["x"].shape[:2]

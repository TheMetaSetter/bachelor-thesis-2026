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
    assert (
        step_output["batch"]["synthetic_anomaly_mask"].shape
        == step_output["batch"]["x"].shape[:2]
    )


def test_redlamp_realistic_validation_is_deterministic_after_rng_reset() -> None:
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

    model.prepare_realistic_validation_epoch(anomaly_probability=0.5)
    first_step = model.realistic_validation_step(batch)
    model.prepare_realistic_validation_epoch(anomaly_probability=0.5)
    second_step = model.realistic_validation_step(batch)

    assert torch.equal(first_step["batch"]["x"], second_step["batch"]["x"])
    assert torch.equal(
        first_step["batch"]["classification_labels"],
        second_step["batch"]["classification_labels"],
    )
    assert torch.equal(
        first_step["batch"]["synthetic_anomaly_mask"],
        second_step["batch"]["synthetic_anomaly_mask"],
    )
    assert (
        first_step["batch"]["augmentation_metadata"]
        == second_step["batch"]["augmentation_metadata"]
    )
    assert "val_realistic_loss" in first_step["log"]
    assert "val_realistic_classification_loss" in first_step["log"]
    assert "val_realistic_classification_accuracy" in first_step["log"]


def test_redlamp_total_loss_supports_explicit_reconstruction_weight() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
        lambda_recon=0.9,
        lambda_cls=0.1,
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    step_output = model.training_step(batch)

    expected_loss = (
        0.9 * step_output["loss_terms"]["reconstruction_loss"]
        + 0.1 * step_output["loss_terms"]["classification_loss"]
    )

    assert torch.isclose(step_output["loss"], expected_loss)


def test_redlamp_clean_validation_loss_excludes_classification_term() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        lambda_recon=0.9,
        lambda_cls=0.1,
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    step_output = model.validation_step(batch)

    expected_loss = 0.9 * step_output["loss_terms"]["reconstruction_loss"]

    assert torch.isclose(step_output["loss"], expected_loss)
    assert step_output["log"]["val_loss"] == expected_loss.item()
    assert "val_classification_loss" not in step_output["log"]
    assert "val_classification_accuracy" not in step_output["log"]

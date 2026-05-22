from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_classification_path_disabled_removes_logits_and_classification_logs() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=32,
        hidden_dim=16,
        num_classes=2,
        classification_label_mode="binary",
        enable_classification_path=False,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
    )
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }
    step_output = model.training_step(batch)

    assert step_output["outputs"]["logits"] is None
    assert step_output["loss_terms"]["reconstruction_loss"].item() >= 0.0
    assert step_output["loss_terms"]["classification_loss"].item() == 0.0
    assert not any("classification_" in metric_name for metric_name in step_output["log"])

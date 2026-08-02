from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_active_thesis_model_returns_documented_shapes() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=2,
        dropout=0.0,
        training_phase="stage_a_multitask_pretraining",
        bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    batch = {
        "x": torch.randn(4, 100, 38),
        "point_labels": torch.zeros(4, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(4)],
    }

    outputs = model(batch)

    assert outputs["hidden"].shape == (4, 100, 16)
    assert outputs["recon"].shape == (4, 100, 38)
    assert outputs["point_scores"].shape == (4, 100)
    assert outputs["window_scores"].shape == (4,)

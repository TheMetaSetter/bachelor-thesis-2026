from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_one_forward_and_backward_step_runs() -> None:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = model.training_step(batch)
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    assert step_output["loss"].item() >= 0.0
    assert "train_loss" in step_output["log"]

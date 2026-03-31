from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel
from src.tasks.multitask_tsad_task import MultitaskTSADTask


def test_one_multitask_forward_and_backward_step_runs() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=4,
        discrete_enabled=True,
        discrete_codebook_size=8,
        fusion_mode="average",
    )
    task = MultitaskTSADTask(
        reconstruction_loss_weight=1.0,
        classification_loss_weight=1.0,
        prototype_loss_weight=0.01,
        use_synthetic_augmentation=True,
        anomaly_probability=1.0,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    step_output = task.training_step(model, batch)
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    assert step_output["loss"].item() >= 0.0
    assert step_output["loss_terms"]["classification_loss"].item() >= 0.0
    assert step_output["batch"]["classification_labels"].sum().item() == 2

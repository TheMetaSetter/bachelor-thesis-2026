from __future__ import annotations

import torch

from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.tasks.reconstruction_task import ReconstructionTask


def test_one_forward_and_backward_step_runs() -> None:
    model = ReconstructionMLPAutoencoder(input_dim=38, encoder_dim=64, hidden_dim=16, dropout=0.0)
    task = ReconstructionTask()
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

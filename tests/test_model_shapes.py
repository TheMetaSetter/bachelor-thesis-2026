from __future__ import annotations

import torch

from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder


def test_reconstruction_model_returns_documented_shapes() -> None:
    model = ReconstructionMLPAutoencoder(input_dim=38, encoder_dim=64, hidden_dim=16, dropout=0.0)
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

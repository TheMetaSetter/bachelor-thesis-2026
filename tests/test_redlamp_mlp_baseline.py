from __future__ import annotations

import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline


def test_redlamp_mlp_baseline_forward_contract_and_mlp_depth() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        dropout=0.0,
        anomaly_probability=1.0,
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    outputs = model(batch)

    assert outputs["recon"].shape == (2, 20, 4)
    assert outputs["hidden"].shape == (2, 20, 16)
    assert outputs["pooled"].shape == (2, 16)
    assert outputs["logits"].shape == (2, len(REDLAMP_MULTICLASS_CLASS_NAMES))
    assert outputs["point_scores"].shape == (2, 20)
    assert torch.allclose(
        outputs["aux"]["class_probabilities"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )

    assert sum(isinstance(layer, torch.nn.Linear) for layer in model.encoder) == 3
    assert sum(isinstance(layer, torch.nn.Linear) for layer in model.decoder) == 3
    assert (
        sum(isinstance(layer, torch.nn.Linear) for layer in model.classification_head)
        == 3
    )

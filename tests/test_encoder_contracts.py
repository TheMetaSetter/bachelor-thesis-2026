from __future__ import annotations

from typing import Any

import torch

from src.models.base_encoder import BaseEncoder
from src.models.modules.continuous_prototypes import ContinuousPrototypeLookup
from src.models.modules.discrete_prototypes import DiscretePrototypeLookup
from src.models.modules.fusion import TaskFusion


class DummyEncoder(BaseEncoder):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        hidden = self.projection(batch["x"])
        return {
            "hidden": hidden,
            "pooled": hidden.mean(dim=1),
            "aux": {"encoder_name": "dummy"},
        }


def test_base_encoder_and_placeholder_modules_preserve_time_major_shapes() -> None:
    batch = {"x": torch.randn(3, 100, 38)}
    encoder = DummyEncoder(input_dim=38, hidden_dim=16)
    continuous_module = ContinuousPrototypeLookup(hidden_dim=16, num_prototypes=0, enabled=False)
    discrete_module = DiscretePrototypeLookup(hidden_dim=16, codebook_size=0, enabled=False)
    fusion_module = TaskFusion(mode="identity")

    encoder_outputs = encoder(batch)
    continuous_outputs = continuous_module(encoder_outputs["hidden"])
    discrete_outputs = discrete_module(encoder_outputs["hidden"])
    fusion_outputs = fusion_module(
        base_hidden=encoder_outputs["hidden"],
        continuous_branch=continuous_outputs,
        discrete_branch=discrete_outputs,
    )

    assert encoder_outputs["hidden"].shape == (3, 100, 16)
    assert encoder_outputs["pooled"].shape == (3, 16)
    assert isinstance(encoder_outputs["aux"], dict)

    assert continuous_outputs["hidden"].shape == (3, 100, 16)
    assert continuous_outputs["prototype_context"].shape == (3, 100, 16)
    assert continuous_outputs["prototype_weights"] is None
    assert isinstance(continuous_outputs["aux"], dict)

    assert discrete_outputs["hidden"].shape == (3, 100, 16)
    assert discrete_outputs["quantized_hidden"].shape == (3, 100, 16)
    assert discrete_outputs["code_indices"] is None
    assert isinstance(discrete_outputs["aux"], dict)

    assert fusion_outputs["hidden_reconstruction"].shape == (3, 100, 16)
    assert fusion_outputs["hidden_classification"].shape == (3, 100, 16)
    assert isinstance(fusion_outputs["aux"], dict)

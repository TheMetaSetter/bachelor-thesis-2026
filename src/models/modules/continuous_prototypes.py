from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ContinuousPrototypeLookup(nn.Module):
    def __init__(self, hidden_dim: int, num_prototypes: int = 0, enabled: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.enabled = enabled

        if enabled and num_prototypes > 0:
            self.prototype_bank = nn.Parameter(torch.randn(num_prototypes, hidden_dim))
        else:
            self.register_parameter("prototype_bank", None)

    def forward(self, hidden: torch.Tensor) -> dict[str, Any]:
        if hidden.ndim != 3:
            raise ValueError("ContinuousPrototypeLookup expects hidden with shape [B, L, H]")

        outputs = {
            "hidden": hidden,
            "prototype_context": hidden,
            "prototype_weights": None,
            "aux": {
                "branch_name": "continuous",
                "enabled": self.enabled,
                "num_prototypes": self.num_prototypes,
            },
        }

        if self.enabled and self.prototype_bank is not None:
            attention_logits = torch.einsum("blh,ph->blp", hidden, self.prototype_bank)
            prototype_weights = torch.softmax(attention_logits, dim=-1)
            prototype_context = torch.einsum("blp,ph->blh", prototype_weights, self.prototype_bank)
            outputs["prototype_context"] = prototype_context
            outputs["prototype_weights"] = prototype_weights

        return outputs

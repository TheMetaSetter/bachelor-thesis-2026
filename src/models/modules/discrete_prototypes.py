from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DiscretePrototypeLookup(nn.Module):
    def __init__(self, hidden_dim: int, codebook_size: int = 0, enabled: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.enabled = enabled

        if enabled and codebook_size > 0:
            self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_dim))
        else:
            self.register_parameter("codebook", None)

    def forward(self, hidden: torch.Tensor) -> dict[str, Any]:
        if hidden.ndim != 3:
            raise ValueError("DiscretePrototypeLookup expects hidden with shape [B, L, H]")

        outputs = {
            "hidden": hidden,
            "quantized_hidden": hidden,
            "code_indices": None,
            "aux": {
                "branch_name": "discrete",
                "enabled": self.enabled,
                "codebook_size": self.codebook_size,
            },
        }

        if self.enabled and self.codebook is not None:
            pairwise_distances = torch.cdist(hidden.reshape(-1, self.hidden_dim), self.codebook)
            code_indices = torch.argmin(pairwise_distances, dim=-1).reshape(hidden.shape[0], hidden.shape[1])
            quantized_hidden = self.codebook[code_indices]
            outputs["quantized_hidden"] = quantized_hidden
            outputs["code_indices"] = code_indices

        return outputs

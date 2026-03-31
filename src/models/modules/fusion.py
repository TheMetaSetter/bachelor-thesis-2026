from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TaskFusion(nn.Module):
    def __init__(self, mode: str = "identity") -> None:
        super().__init__()
        self.mode = mode

    def forward(
        self,
        base_hidden: torch.Tensor,
        continuous_branch: dict[str, Any] | None = None,
        discrete_branch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if base_hidden.ndim != 3:
            raise ValueError("TaskFusion expects base_hidden with shape [B, L, H]")

        reconstruction_hidden = base_hidden
        classification_hidden = base_hidden

        if self.mode == "average":
            branch_hidden_tensors = [base_hidden]
            if continuous_branch is not None:
                branch_hidden_tensors.append(continuous_branch["prototype_context"])
            if discrete_branch is not None:
                branch_hidden_tensors.append(discrete_branch["quantized_hidden"])
            stacked_hidden = torch.stack(branch_hidden_tensors, dim=0)
            averaged_hidden = stacked_hidden.mean(dim=0)
            reconstruction_hidden = averaged_hidden
            classification_hidden = averaged_hidden

        return {
            "hidden_reconstruction": reconstruction_hidden,
            "hidden_classification": classification_hidden,
            "aux": {
                "fusion_mode": self.mode,
                "used_continuous_branch": continuous_branch is not None,
                "used_discrete_branch": discrete_branch is not None,
            },
        }

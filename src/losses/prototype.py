from __future__ import annotations

from typing import Any

import torch


def compute_prototype_regularization(aux_outputs: dict[str, Any] | None) -> torch.Tensor:
    if not aux_outputs:
        return torch.tensor(0.0)

    reference_tensor: torch.Tensor | None = None
    for value in aux_outputs.values():
        if isinstance(value, torch.Tensor):
            reference_tensor = value
            break
        if isinstance(value, dict):
            for nested_value in value.values():
                if isinstance(nested_value, torch.Tensor):
                    reference_tensor = nested_value
                    break
        if reference_tensor is not None:
            break

    if reference_tensor is None:
        return torch.tensor(0.0)

    regularization = torch.tensor(0.0, device=reference_tensor.device)
    continuous_branch = aux_outputs.get("continuous_branch")
    if isinstance(continuous_branch, dict):
        prototype_weights = continuous_branch.get("prototype_weights")
        if isinstance(prototype_weights, torch.Tensor):
            regularization = regularization + prototype_weights.pow(2).mean()

    discrete_branch = aux_outputs.get("discrete_branch")
    if isinstance(discrete_branch, dict):
        quantized_hidden = discrete_branch.get("quantized_hidden")
        hidden = discrete_branch.get("hidden")
        if isinstance(quantized_hidden, torch.Tensor) and isinstance(hidden, torch.Tensor):
            regularization = regularization + torch.mean((quantized_hidden - hidden) ** 2)

    return regularization

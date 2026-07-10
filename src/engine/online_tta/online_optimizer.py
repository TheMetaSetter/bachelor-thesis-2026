from __future__ import annotations

from typing import Iterable

import torch


def _resolve_projector_module(model: torch.nn.Module) -> torch.nn.Module:
    projector = getattr(model, "online_mlp_projector", None)
    if projector is None:
        projector = getattr(model, "projector", None)
    if projector is None:
        raise AttributeError("Model does not expose online_mlp_projector or projector")
    if not isinstance(projector, torch.nn.Module):
        raise TypeError("Resolved projector attribute must be a torch.nn.Module")
    return projector


def collect_projector_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    projector = _resolve_projector_module(model)
    return list(projector.parameters())


def build_online_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Create a fresh per-event AdamW optimizer for the projector only."""
    assert_only_projector_is_trainable(model)
    return torch.optim.AdamW(
        collect_projector_parameters(model),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def clip_projector_gradients(model: torch.nn.Module, max_norm: float = 0.5) -> float:
    """Clip projector gradients and return the pre-clipping norm."""
    parameters = collect_projector_parameters(model)
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    return float(norm.detach().cpu())


def assert_only_projector_is_trainable(model: torch.nn.Module) -> None:
    projector_parameter_ids = {
        id(parameter) for parameter in collect_projector_parameters(model)
    }
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad and id(parameter) not in projector_parameter_ids:
            raise ValueError(
                "Only online_mlp_projector parameters may require gradients, "
                f"but {parameter_name!r} is trainable"
            )

    missing_projector_parameters = [
        parameter_name
        for parameter_name, parameter in _resolve_projector_module(
            model
        ).named_parameters()
        if not parameter.requires_grad
    ]
    if missing_projector_parameters:
        raise ValueError(
            "Some online_mlp_projector parameters are frozen unexpectedly: "
            f"{missing_projector_parameters}"
        )

    if not trainable_names:
        raise ValueError("No trainable parameters were found on the online model")

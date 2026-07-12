from __future__ import annotations

from typing import Any

import torch


def move_initialization_batch_to_device(
    batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def maybe_initialize_memories_from_loader(
    model: Any,
    train_loader: Any,
    *,
    device: str,
) -> bool:
    if model.memory_initialized:
        return False
    if model._is_bootstrap_active():
        return False
    if not model._phase_uses_prototype_path():
        return False

    model.memory_ready_for_initialization = True
    token_pool = model._collect_memory_initialization_token_pool_from_loader(
        train_loader,
        device,
    )
    if token_pool["continuous_hidden_tokens"].numel() == 0:
        raise ValueError("memory initialization requires at least one normal token")

    model._initialize_memory_buffers_from_token_pool(
        continuous_hidden_tokens=token_pool["continuous_hidden_tokens"],
        discrete_hidden_tokens_by_class=token_pool["discrete_hidden_tokens_by_class"],
    )
    model.mark_memories_initialized(model.current_epoch_index + 1)
    return True

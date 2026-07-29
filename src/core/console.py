from __future__ import annotations

"""Small helpers for always-on runtime console instrumentation.

The codebase keeps using plain print statements, but these helpers centralize
formatting so the console stays readable even when many runtime surfaces emit
debug context on every batch and step.
"""

from pathlib import Path
import os
from typing import Any

import torch
import torch.nn as nn


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def format_console_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return summarize_tensor(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        preview_items = []
        for index, (key, item_value) in enumerate(value.items()):
            if index >= 8:
                preview_items.append("...")
                break
            preview_items.append(f"{key}={format_console_value(item_value)}")
        return "{" + ", ".join(preview_items) + "}"
    if isinstance(value, (list, tuple)):
        preview_values = [format_console_value(item) for item in list(value)[:8]]
        if len(value) > 8:
            preview_values.append("...")
        bracket_open, bracket_close = (
            ("[", "]") if isinstance(value, list) else ("(", ")")
        )
        return bracket_open + ", ".join(preview_values) + bracket_close
    return _format_scalar(value)


def console_print(prefix: str, message: str, **fields: Any) -> None:
    if _env_flag("THESIS_CONSOLE_QUIET"):
        return
    ordered_fields = ", ".join(
        f"{field_name}={format_console_value(field_value)}"
        for field_name, field_value in fields.items()
    )
    if ordered_fields:
        print(f"[{prefix}] {message} | {ordered_fields}")
        return
    print(f"[{prefix}] {message}")


def debug_print(prefix: str, message: str, **fields: Any) -> None:
    if not _env_flag("THESIS_DEBUG_VERIFICATION_INIT"):
        return
    ordered_fields = ", ".join(
        f"{field_name}={format_console_value(field_value)}"
        for field_name, field_value in fields.items()
    )
    if ordered_fields:
        print(f"[{prefix}][DEBUG] {message} | {ordered_fields}")
        return
    print(f"[{prefix}][DEBUG] {message}")


def debug_print_if(flag_name: str, prefix: str, message: str, **fields: Any) -> None:
    if not _env_flag(flag_name):
        return
    ordered_fields = ", ".join(
        f"{field_name}={format_console_value(field_value)}"
        for field_name, field_value in fields.items()
    )
    if ordered_fields:
        print(f"[{prefix}][DEBUG] {message} | {ordered_fields}")
        return
    print(f"[{prefix}][DEBUG] {message}")


def summarize_tensor(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def summarize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    batch_summary: dict[str, Any] = {
        "x": summarize_tensor(batch.get("x")),
        "point_labels": summarize_tensor(batch.get("point_labels")),
        "mask": summarize_tensor(batch.get("mask")),
        "timestamps": summarize_tensor(batch.get("timestamps")),
        "meta_count": len(batch.get("meta", [])),
    }
    if "classification_labels" in batch:
        batch_summary["classification_labels"] = summarize_tensor(
            batch.get("classification_labels")
        )
    if "synthetic_anomaly_mask" in batch:
        batch_summary["synthetic_anomaly_mask"] = summarize_tensor(
            batch.get("synthetic_anomaly_mask")
        )
    return batch_summary


def summarize_label_distribution(labels: torch.Tensor | None) -> dict[str, int] | str:
    if labels is None:
        return "None"
    flattened_labels = labels.detach().cpu().reshape(-1)
    unique_values, counts = torch.unique(flattened_labels, return_counts=True)
    return {
        str(int(unique_value.item())): int(count.item())
        for unique_value, count in zip(unique_values, counts)
    }


def count_parameters(component: nn.Module | nn.Parameter | torch.Tensor | None) -> int:
    if component is None:
        return 0
    if isinstance(component, nn.Module):
        return sum(parameter.numel() for parameter in component.parameters())
    if isinstance(component, nn.Parameter):
        return int(component.numel())
    if isinstance(component, torch.Tensor):
        return int(component.numel())
    return 0


def count_trainable_parameters(
    component: nn.Module | nn.Parameter | torch.Tensor | None,
) -> int:
    if component is None:
        return 0
    if isinstance(component, nn.Module):
        return sum(
            parameter.numel()
            for parameter in component.parameters()
            if parameter.requires_grad
        )
    if isinstance(component, nn.Parameter):
        return int(component.numel()) if component.requires_grad else 0
    return 0


def print_parameter_summary(
    prefix: str,
    model_name: str,
    model: nn.Module,
    components: dict[str, nn.Module | nn.Parameter | torch.Tensor | None],
    **extra_fields: Any,
) -> None:
    console_print(
        prefix,
        f"Parameter summary for {model_name}",
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        **extra_fields,
    )
    for component_name, component in components.items():
        console_print(
            prefix,
            f"Component parameters: {component_name}",
            total_parameters=count_parameters(component),
            trainable_parameters=count_trainable_parameters(component),
        )

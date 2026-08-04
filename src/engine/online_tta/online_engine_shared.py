from __future__ import annotations

"""Shared construction and runtime-state helpers for THESIS online TTA."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.core.console import console_print
from src.core.registry import build_model
from src.engine.online_tta.online_optimizer import collect_projector_parameters


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(path)


def _validate_single_window_online_batch(batch: dict[str, Any]) -> None:
    if int(batch["x"].shape[0]) != 1:
        raise ValueError(
            "online benchmark batches must contain exactly one causal window"
        )
    if len(batch.get("meta", [])) != 1:
        raise ValueError("online benchmark batches must carry exactly one meta row")


def _sync_online_runtime_state(
    *,
    runtime_state,
    active_ewma_point_scores: dict[int, float],
    record: dict[str, Any],
    hard_old_guard,
    verification_buffer,
) -> None:
    runtime_state.replace_active_ewma_point_scores(active_ewma_point_scores)
    runtime_state.advance_cursor(1)
    runtime_state.append_verification_history(record)
    runtime_state.hard_old_intervals = hard_old_guard.intervals()
    runtime_state.verification_entries = verification_buffer.items()


def _load_model_kwargs(experiment_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs = {
        key: value
        for key, value in experiment_config["model"].items()
        if key != "model_name"
    }
    model_kwargs.update(
        {
            key: value
            for key, value in experiment_config["task"].items()
            if key
            in {
                "reference_checkpoint_path",
                "online_variant",
                "warm_start_projector",
                "target_param_group",
                "clean_stream_only",
                "reset_policy",
                "reset_alignment_threshold",
            }
        }
    )
    return model_kwargs


def _build_model_from_experiment_config(
    experiment_config: dict[str, Any],
) -> torch.nn.Module:
    model_name = experiment_config["model"]["model_name"]
    model_kwargs = _load_model_kwargs(experiment_config)
    model_kwargs["online_variant"] = str(experiment_config["online_variant"])
    console_print(
        "MODEL",
        "Building online TTA model",
        model_name=model_name,
        model_kwargs_keys=sorted(model_kwargs.keys()),
    )
    return build_model(model_name, **model_kwargs)


def _build_optimizer_from_experiment_config(
    model: torch.nn.Module,
    experiment_config: dict[str, Any],
) -> torch.optim.Optimizer:
    optimizer_config = experiment_config["optimizer"]
    optimizer_name = str(optimizer_config.get("optimizer_name", "adamw"))
    target_param_group = str(experiment_config["task"]["target_param_group"])
    if target_param_group != "projector_params":
        raise ValueError(
            "The Phase 4 online TTA core supports only target_param_group='projector_params'"
        )
    optimizer_parameters = collect_projector_parameters(model)
    optimizer_kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(optimizer_parameters, **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

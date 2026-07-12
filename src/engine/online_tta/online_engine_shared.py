from __future__ import annotations

"""Shared calibration and artifact helpers for THESIS online TTA."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.console import console_print
from src.core.registry import build_dataset, build_model
from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta.online_calibration import (
    build_online_stream as _build_online_stream,
    collect_clean_validation_scores as _collect_clean_validation_scores,
)
from src.engine.thresholding import (
    select_clean_validation_point_threshold,
    select_online_ewma_threshold,
)
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)
from src.engine.online_tta.signature_verification import signature_window_to_dict


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


def _serialize_recurrent_signatures(
    recurrent_signatures: set[tuple[int, ...]]
) -> list[dict[str, Any]]:
    return [
        {"signature": [int(value) for value in signature]}
        for signature in sorted(recurrent_signatures)
    ]


def _sync_online_runtime_state(
    *,
    runtime_state,
    previous_ewma_score: float | None,
    signature_history: list[Any],
    recurrent_signatures: set[tuple[int, ...]],
    record: dict[str, Any],
    hard_old_guard,
    verification_buffer,
) -> None:
    runtime_state.record_previous_ewma_score(previous_ewma_score)
    runtime_state.advance_cursor(1)
    runtime_state.signature_history = [
        signature_window_to_dict(window) for window in signature_history
    ]
    runtime_state.append_recurrent_signatures(
        _serialize_recurrent_signatures(recurrent_signatures)
    )
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


def _build_threshold_artifact_from_scores(
    *,
    calibration_scores: dict[str, list[float]],
    entity_id: str,
    online_variant: str,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    window_size: int,
) -> dict[str, Any]:
    offline_point_threshold = select_clean_validation_point_threshold(
        np.asarray(calibration_scores["offline_point"], dtype=float),
        quantile=float(protocol_config["offline_threshold_quantile"]),
    )
    online_ewma_point_threshold = select_online_ewma_threshold(
        np.asarray(calibration_scores["ewma"], dtype=float),
        quantile=float(protocol_config["online_threshold_quantile"]),
    )
    return build_threshold_artifact(
        method_name="THESIS",
        variant_name=online_variant,
        entity_id=entity_id,
        seed=int(experiment_config["seed"]),
        window_size=window_size,
        offline_point_threshold=offline_point_threshold,
        online_ewma_point_threshold=online_ewma_point_threshold,
        input_window_threshold=float(np.quantile(calibration_scores["input_window"], 0.99)),
        latent_window_low_threshold=float(
            np.quantile(calibration_scores["latent_window"], 0.95)
        ),
        latent_window_high_threshold=float(
            np.quantile(calibration_scores["latent_window"], 0.99)
        ),
        quantile=float(protocol_config["online_threshold_quantile"]),
        ewma_current_weight=float(protocol_config["online_ewma_current_weight"]),
        ewma_previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        created_by="src/engine/online_tta/online_engine.py",
        config_path=str(experiment_config.get("experiment_name", "unknown")),
        resolved_config_sha256=CheckpointManager._stable_json_digest(experiment_config),
    )


def calibrate_online_threshold_artifact(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
    device: str,
) -> dict[str, Any]:
    window_size = int(protocol_config["window_size"])
    batch_size = int(experiment_config["data"]["batch_size"])
    view_noise_std = float(experiment_config["task"].get("view_noise_std", 0.0))
    view_dropout_probability = float(
        experiment_config["task"].get("view_dropout_probability", 0.0)
    )
    calibration_scores = _collect_clean_validation_scores(
        model=model,
        clean_validation_sequences=clean_validation_sequences,
        window_size=window_size,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
        device=device,
        current_weight=float(protocol_config["online_ewma_current_weight"]),
        previous_weight=float(protocol_config["online_ewma_previous_weight"]),
    )
    entity_id = (
        str(clean_validation_sequences[0]["meta"]["entity_id"])
        if clean_validation_sequences
        else "unknown"
    )
    return _build_threshold_artifact_from_scores(
        calibration_scores=calibration_scores,
        entity_id=entity_id,
        online_variant=online_variant,
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        window_size=window_size,
    )


def calibrate_entity_thresholds(
    *,
    model: torch.nn.Module,
    clean_validation_sequence: dict[str, Any],
    entity_id: str,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    actual = str(clean_validation_sequence.get("meta", {}).get("entity_id", ""))
    if actual != entity_id:
        raise ValueError(f"validation entity {actual!r} does not match {entity_id!r}")
    return calibrate_online_threshold_artifact(
        model=model,
        clean_validation_sequences=[clean_validation_sequence],
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=str(experiment_config.get("online_variant", "A0")),
        device=device,
    )


def calibrate_entity_threshold_artifacts(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
    device: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for sequence in clean_validation_sequences:
        entity_id = str(sequence.get("meta", {}).get("entity_id", ""))
        if not entity_id:
            raise ValueError("clean validation sequence is missing entity_id")
        if entity_id in grouped:
            continue
        grouped[entity_id] = calibrate_entity_thresholds(
            model=model,
            clean_validation_sequence=sequence,
            entity_id=entity_id,
            experiment_config={**experiment_config, "online_variant": online_variant},
            protocol_config=protocol_config,
            device=device,
        )
    return grouped

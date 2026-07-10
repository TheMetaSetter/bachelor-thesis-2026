from __future__ import annotations

"""THESIS online TTA core.

₍₍⚞(˶˃ ꒳ ˂˶)⚟⁾⁾ Online flow

clean validation
  -> stride-1 causal endpoint scores
  -> EWMA calibration
  -> threshold artifact
  -> test stream
  -> triage
  -> projector-only update
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.console import console_print
from src.core.registry import build_dataset, build_model
from src.core.runtime_components import register_online_runtime_components
from src.core.seed import seed_everything
from src.data.stream import OnlineWindowBatcher, SMDOnlineStream
from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta.online_losses import (
    compute_a1_pnn_reconstruction_loss,
    compute_a2_hard_old_reconstruction_loss,
    compute_a2_online_contrastive_loss,
    compute_hard_old_hinge_loss,
    compute_masked_pnn_reconstruction_loss,
)
from src.engine.online_tta.online_optimizer import (
    assert_only_projector_is_trainable,
    build_online_optimizer,
    clip_projector_gradients,
    collect_projector_parameters,
)
from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.triage import classify_online_window
from src.engine.online_tta.ttl_buffer import TTLBuffer
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.thresholding import (
    select_clean_validation_point_threshold,
    select_online_ewma_threshold,
)
from src.protocols.point_scores import (
    ewma_scores,
    window_scores_to_causal_endpoint_scores,
)
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)


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


def _build_online_stream(
    *,
    sequences: list[dict[str, Any]],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
) -> OnlineWindowBatcher:
    stream = SMDOnlineStream(
        sequences=sequences,
        window_size=window_size,
        stride=1,
        clean_stream_only=True,
        stream_window_mode="sliding_stride_1",
    )
    return OnlineWindowBatcher(
        stream=stream,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
    )


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _run_stride1_sequence_scores(
    *,
    model: torch.nn.Module,
    sequence: dict[str, Any],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
) -> np.ndarray:
    batcher = _build_online_stream(
        sequences=[sequence],
        window_size=window_size,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
    )
    sequence_length = int(sequence["x"].shape[0])
    window_scores: list[float] = []

    for batch in batcher:
        batch_on_device = _move_batch_to_device(batch, device)
        model.eval()
        with torch.no_grad():
            outputs = model.forward(batch_on_device)
        window_scores.append(float(outputs["point_scores"][0, -1].detach().cpu()))

    return window_scores_to_causal_endpoint_scores(
        window_scores=window_scores,
        sequence_length=sequence_length,
        window_size=window_size,
    )


def _collect_clean_validation_scores(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
    current_weight: float,
    previous_weight: float,
) -> tuple[list[float], list[float]]:
    clean_validation_point_scores: list[float] = []
    clean_validation_ewma_scores: list[float] = []
    for sequence in clean_validation_sequences:
        causal_point_scores = _run_stride1_sequence_scores(
            model=model,
            sequence=sequence,
            window_size=window_size,
            batch_size=batch_size,
            view_noise_std=view_noise_std,
            view_dropout_probability=view_dropout_probability,
            device=device,
        )
        smoothed_scores = ewma_scores(
            causal_point_scores,
            current_weight=current_weight,
            previous_weight=previous_weight,
        )
        clean_validation_point_scores.extend(
            float(score) for score in causal_point_scores if not np.isnan(score)
        )
        clean_validation_ewma_scores.extend(
            float(score) for score in smoothed_scores if not np.isnan(score)
        )
    return clean_validation_point_scores, clean_validation_ewma_scores


def _build_threshold_artifact_from_scores(
    *,
    clean_validation_point_scores: list[float],
    clean_validation_ewma_scores: list[float],
    entity_id: str,
    online_variant: str,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    window_size: int,
) -> dict[str, Any]:
    offline_point_threshold = select_clean_validation_point_threshold(
        np.asarray(clean_validation_point_scores, dtype=float),
        quantile=float(protocol_config["offline_threshold_quantile"]),
    )
    online_ewma_point_threshold = select_online_ewma_threshold(
        np.asarray(clean_validation_ewma_scores, dtype=float),
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
        quantile=float(protocol_config["online_threshold_quantile"]),
        ewma_current_weight=float(protocol_config["online_ewma_current_weight"]),
        ewma_previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        created_by="src/engine/online_tta/online_engine.py",
        config_path=str(experiment_config.get("experiment_name", "unknown")),
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
    clean_validation_point_scores, clean_validation_ewma_scores = (
        _collect_clean_validation_scores(
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
    )
    entity_id = (
        str(clean_validation_sequences[0]["meta"]["entity_id"])
        if clean_validation_sequences
        else "unknown"
    )
    return _build_threshold_artifact_from_scores(
        clean_validation_point_scores=clean_validation_point_scores,
        clean_validation_ewma_scores=clean_validation_ewma_scores,
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
    """Calibrate exactly one entity without reading labels or other entities."""
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
    """Return independent threshold artifacts keyed by validation entity."""
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


def _build_triage_thresholds(online_ewma_threshold: float) -> dict[str, float]:
    threshold = float(online_ewma_threshold)
    return {
        "input_window_threshold": threshold,
        "latent_window_low_threshold": threshold * 0.5,
        "latent_window_high_threshold": threshold,
        # Legacy names remain for consumers that inspect the context directly.
        "strong_anomaly_threshold": threshold,
        "hard_old_normality_threshold": threshold * 0.5,
        "pnn_candidate_input_threshold": threshold * 0.75,
        "pnn_candidate_latent_threshold": threshold * 0.75,
    }


def _step_result(
    *,
    record: dict[str, Any],
    did_update: bool,
    loss_total: float | None,
) -> dict[str, Any]:
    record["did_update"] = did_update
    record["loss_total"] = loss_total
    return {
        "record": record,
        "online_variant": record["online_variant"],
        "did_update": did_update,
        "loss_total": loss_total,
    }


def _process_online_window(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    triage_thresholds: dict[str, float],
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
    previous_ewma_score: float | None,
    device: str,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    batch_on_device, raw_point_score, latent_window_score, ewma_point_score = (
        _score_online_window(
            model=model,
            batch=batch,
            previous_ewma_score=previous_ewma_score,
            ewma_current_weight=ewma_current_weight,
            ewma_previous_weight=ewma_previous_weight,
            device=device,
        )
    )
    triage_decision = classify_online_window(
        input_window_score=raw_point_score,
        latent_window_score=latent_window_score,
        thresholds=triage_thresholds,
    )
    _update_online_window_buffers(
        batch_on_device=batch_on_device,
        raw_point_score=raw_point_score,
        triage_decision=triage_decision,
        verification_buffer=verification_buffer,
        ttl_buffer=ttl_buffer,
    )
    event_optimizer = optimizer
    if online_variant != "A0":
        event_optimizer = build_online_optimizer(model)
    step_result = execute_online_tta_step(
        model=model,
        optimizer=event_optimizer,
        batch=batch_on_device,
        online_variant=online_variant,
        threshold_value=threshold_value,
        ewma_point_score=ewma_point_score,
        raw_point_score=raw_point_score,
        latent_window_score=latent_window_score,
        triage_decision=triage_decision,
    )
    record, metric = _build_online_window_outputs(
        step_result=step_result,
        threshold_value=threshold_value,
        raw_point_score=raw_point_score,
        ewma_point_score=ewma_point_score,
        triage_decision=triage_decision,
        verification_buffer=verification_buffer,
        ttl_buffer=ttl_buffer,
    )
    return ewma_point_score, metric, record


def _score_online_window(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    previous_ewma_score: float | None,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    device: str,
) -> tuple[dict[str, Any], float, float, float]:
    batch_on_device = _move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        pre_outputs = model.forward(batch_on_device)
    raw_point_score = float(pre_outputs["point_scores"][0, -1].detach().cpu())
    latent_value = pre_outputs["aux"].get("latent_window_score")
    if latent_value is None:
        latent_value = pre_outputs["window_scores"]
    latent_window_score = float(torch.as_tensor(latent_value).mean().detach().cpu())
    if previous_ewma_score is None:
        ewma_point_score = raw_point_score
    else:
        ewma_point_score = (
            ewma_current_weight * raw_point_score
            + ewma_previous_weight * previous_ewma_score
        )
    return batch_on_device, raw_point_score, latent_window_score, ewma_point_score


def _update_online_window_buffers(
    *,
    batch_on_device: dict[str, Any],
    raw_point_score: float,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
) -> None:
    if triage_decision == "pnn_candidate":
        verification_buffer.add(
            {
                "window_start": int(batch_on_device["meta"][0]["start_index"]),
                "window_end": int(batch_on_device["meta"][0]["end_index"]),
                "point_score": raw_point_score,
            }
        )
    if triage_decision != "strong_anomaly":
        ttl_buffer.add(
            item=int(batch_on_device["meta"][0]["end_index"]) - 1,
            current_step=int(batch_on_device["meta"][0]["stream_step"]),
        )


def _build_online_window_outputs(
    *,
    step_result: dict[str, Any],
    threshold_value: float,
    raw_point_score: float,
    ewma_point_score: float,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = dict(step_result["record"])
    record["threshold"] = float(threshold_value)
    record["input_window_score"] = float(raw_point_score)
    record["verification_cycle_ready"] = verification_buffer.should_verify()
    metric = {
        "online/step": 0,
        "online/raw_point_score": raw_point_score,
        "online/ewma_point_score": ewma_point_score,
        "online/threshold": float(threshold_value),
        "online/prediction": record["prediction"],
        "online/did_update": record["did_update"],
        "online/loss_total": record["loss_total"],
        "online/triage_decision": triage_decision,
        "online/input_window_score": raw_point_score,
        "online/latent_window_score": float(step_result["record"].get("latent_window_score", 0.0)),
        "online/num_buffer_admitted_windows": len(verification_buffer),
        "online/num_buffer_rejected_overlap_windows": 0,
        "online/num_points_removed_by_discrete_anom_filter": 0,
        "online/num_points_remaining_for_signature": 0,
        "online/num_recurrent_signatures": 0,
        "online/num_pseudo_new_normality_points": 0,
        "online/loss_hard_recon": step_result["record"].get("loss_total") if triage_decision == "hard_old_normality" else None,
        "online/loss_pnn_recon": step_result["record"].get("loss_total") if triage_decision == "pnn_candidate" else None,
        "online/loss_contrastive": None,
        "online/projector_grad_norm": None,
        "online/source_encoder_grad_norm": 0.0,
        "online/source_memory_grad_norm": 0.0,
        "online/recon_head_grad_norm": 0.0,
        "online/classification_head_grad_norm": 0.0,
        "online/verification_buffer_size": len(verification_buffer),
        "online/ttl_buffer_size": len(ttl_buffer),
    }
    return record, metric


def _compute_step_scores(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    ewma_point_score: float | None,
    raw_point_score: float | None,
    latent_window_score: float | None,
) -> tuple[float, float, float]:
    if raw_point_score is None or latent_window_score is None:
        model.eval()
        with torch.no_grad():
            scoring_outputs = model.forward(batch)
        if raw_point_score is None:
            raw_point_score = float(
                scoring_outputs["point_scores"][0, -1].detach().cpu()
            )
        if latent_window_score is None:
            latent_window_score = float(
                torch.as_tensor(
                    scoring_outputs["aux"].get(
                        "latent_window_score", scoring_outputs["window_scores"]
                    )
                ).mean().detach().cpu()
            )
    if ewma_point_score is None:
        ewma_point_score = float(raw_point_score)
    return (
        float(ewma_point_score),
        float(raw_point_score),
        float(latent_window_score),
    )


def _build_step_record(
    *,
    batch: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    ewma_point_score: float,
    raw_point_score: float,
    latent_window_score: float,
    triage_decision: str | None,
) -> dict[str, Any]:
    meta = batch["meta"][0]
    return {
        "entity_id": str(meta["entity_id"]),
        "point_index": int(meta["end_index"]) - 1,
        "window_start_index": int(meta["start_index"]),
        "window_end_index": int(meta["end_index"]),
        "raw_point_score": raw_point_score,
        "ewma_point_score": ewma_point_score,
        "latent_window_score": latent_window_score,
        "threshold": float(threshold_value),
        "prediction": int(ewma_point_score > float(threshold_value)),
        "online_variant": online_variant,
        "triage_decision": triage_decision,
        "did_update": False,
        "loss_total": None,
    }


def _run_online_variant_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    online_variant: str,
    triage_decision: str | None,
    training_outputs: dict[str, Any],
    threshold_value: float,
) -> torch.Tensor | None:
    if online_variant == "A1":
        if triage_decision != "pnn_candidate":
            return None
        pnn_mask = batch.get("pnn_mask")
        if pnn_mask is not None:
            loss_total = compute_masked_pnn_reconstruction_loss(
                training_outputs["recon"], batch["x"], pnn_mask
            )
        else:
            loss_total = compute_a1_pnn_reconstruction_loss(
                training_outputs["recon"], batch["x"], mask=batch.get("mask")
            )
    elif online_variant == "A2":
        window_score = training_outputs["window_scores"].mean()
        reconstruction_loss = compute_hard_old_hinge_loss(
            window_score, threshold_value
        )
        if triage_decision in {"gray_zone", "pnn_candidate"}:
            contrastive_loss = compute_a2_online_contrastive_loss(
                training_outputs["aux"]["reference_hidden"],
                training_outputs["aux"]["projected_hidden"],
            )
            loss_total = reconstruction_loss + 0.1 * contrastive_loss
        else:
            loss_total = reconstruction_loss
    else:
        raise ValueError("online_variant must be one of: A0, A1, A2")

    loss_total.backward()
    clip_projector_gradients(model)
    optimizer.step()
    return loss_total


def execute_online_tta_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    batch: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    ewma_point_score: float | None = None,
    raw_point_score: float | None = None,
    latent_window_score: float | None = None,
    triage_decision: str | None,
) -> dict[str, Any]:
    ewma_point_score, raw_point_score, latent_window_score = _compute_step_scores(
        model=model,
        batch=batch,
        ewma_point_score=ewma_point_score,
        raw_point_score=raw_point_score,
        latent_window_score=latent_window_score,
    )
    record = _build_step_record(
        batch=batch,
        online_variant=online_variant,
        threshold_value=threshold_value,
        ewma_point_score=ewma_point_score,
        raw_point_score=raw_point_score,
        latent_window_score=latent_window_score,
        triage_decision=triage_decision,
    )
    if optimizer is None or online_variant == "A0":
        return _step_result(record=record, did_update=False, loss_total=None)
    if triage_decision == "strong_anomaly":
        return _step_result(record=record, did_update=False, loss_total=None)
    model.train()
    optimizer.zero_grad()
    training_outputs = model.forward(batch)
    loss_total = _run_online_variant_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        online_variant=online_variant,
        triage_decision=triage_decision,
        training_outputs=training_outputs,
        threshold_value=threshold_value,
    )
    if loss_total is None:
        return _step_result(record=record, did_update=False, loss_total=None)
    return _step_result(
        record=record, did_update=True, loss_total=float(loss_total.detach().cpu())
    )


def _run_online_sequence(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sequence: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    protocol_config: dict[str, Any],
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
    max_online_steps: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batcher = _build_online_stream(
        sequences=[sequence],
        window_size=int(protocol_config["window_size"]),
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
    )
    metric_history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    previous_ewma_score: float | None = None
    ewma_current_weight = float(protocol_config["online_ewma_current_weight"])
    ewma_previous_weight = float(protocol_config["online_ewma_previous_weight"])
    triage_thresholds = _build_triage_thresholds(threshold_value)

    for batch in batcher:
        if max_online_steps is not None and len(metric_history) >= max_online_steps:
            break
        previous_ewma_score, metric, record = _process_online_window(
            model=model,
            optimizer=optimizer,
            batch=batch,
            online_variant=online_variant,
            threshold_value=threshold_value,
            ewma_current_weight=ewma_current_weight,
            ewma_previous_weight=ewma_previous_weight,
            triage_thresholds=triage_thresholds,
            verification_buffer=verification_buffer,
            ttl_buffer=ttl_buffer,
            previous_ewma_score=previous_ewma_score,
            device=device,
        )
        metric["online/step"] = len(metric_history) + 1
        records.append(record)
        metric_history.append(metric)

    return metric_history, records


def _build_online_execution_context(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return _build_dry_run_online_context(online_variant=online_variant)
    return _build_runtime_online_context(
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
    )


def _build_dry_run_online_context(*, online_variant: str) -> dict[str, Any]:
    return {
        "benchmark_status": "dry_run",
        "created_at_utc": _utc_now_iso(),
        "online_variant": online_variant,
        "threshold_artifact": None,
        "threshold_artifact_path": None,
        "final_checkpoint_path": None,
        "metric_history": [],
        "records": [],
        "online_metrics_path": None,
        "online_records_path": None,
    }


def _build_runtime_online_context(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
) -> dict[str, Any]:
    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"], experiment_config["data"]
    )
    model = _build_model_from_experiment_config(experiment_config)
    optimizer = _build_optimizer_from_experiment_config(model, experiment_config)
    assert_only_projector_is_trainable(model)
    checkpoint_manager = CheckpointManager(
        Path(str(experiment_config["checkpoint_dir"]))
    )
    threshold_artifact = calibrate_online_threshold_artifact(
        model=model,
        clean_validation_sequences=data_bundle["scaled_sequences"]["val"],
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        device=str(experiment_config["device"]),
    )
    output_dir = Path(str(experiment_config["output_dir"]))
    threshold_path = output_dir / "thresholds" / "online_thresholds.json"
    write_threshold_artifact(threshold_artifact, threshold_path)
    return {
        "benchmark_status": "completed",
        "created_at_utc": _utc_now_iso(),
        "online_variant": online_variant,
        "data_bundle": data_bundle,
        "model": model,
        "optimizer": optimizer,
        "checkpoint_manager": checkpoint_manager,
        "threshold_artifact": threshold_artifact,
        "threshold_artifact_path": str(threshold_path),
        "threshold_value": float(
            threshold_artifact["thresholds"]["online_ewma_point"]["value"]
        ),
        "output_dir": output_dir,
        "batch_size": int(experiment_config["data"]["batch_size"]),
        "view_noise_std": float(experiment_config["task"].get("view_noise_std", 0.0)),
        "view_dropout_probability": float(
            experiment_config["task"].get("view_dropout_probability", 0.0)
        ),
        "device": str(experiment_config["device"]),
        "max_online_steps": int(experiment_config["task"].get("max_online_steps", 0)),
        "verification_buffer": VerificationBuffer(max_size=64, non_overlap_gap=0),
        "hard_old_guard": NonOverlapGuard(max_size=1),
        "ttl_buffer": TTLBuffer(ttl_steps=int(protocol_config["window_size"])),
    }


def _run_online_execution_sequences(
    *,
    context: dict[str, Any],
    protocol_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    max_online_steps = int(context.get("max_online_steps", 0))
    max_online_steps_limit = max_online_steps if max_online_steps > 0 else None
    for sequence in context["data_bundle"]["scaled_sequences"]["test"]:
        if max_online_steps_limit is not None and len(metric_history) >= max_online_steps_limit:
            break
        sequence_metric_history, sequence_records = _run_online_sequence(
            model=context["model"],
            optimizer=context["optimizer"],
            sequence=sequence,
            online_variant=context["online_variant"],
            threshold_value=context["threshold_value"],
            protocol_config=protocol_config,
            batch_size=context["batch_size"],
            view_noise_std=context["view_noise_std"],
            view_dropout_probability=context["view_dropout_probability"],
            device=context["device"],
            verification_buffer=context["verification_buffer"],
            ttl_buffer=context["ttl_buffer"],
            max_online_steps=(
                None
                if max_online_steps_limit is None
                else max_online_steps_limit - len(metric_history)
            ),
        )
        metric_history.extend(sequence_metric_history)
        records.extend(sequence_records)
    return metric_history, records


def _finalize_online_execution(
    *,
    context: dict[str, Any],
    experiment_config: dict[str, Any],
    metric_history: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = context["output_dir"]
    metrics_path = _write_json(output_dir / "online_metrics.json", metric_history)
    records_path = _write_json(output_dir / "online_records.json", records)
    final_checkpoint_path = context["checkpoint_manager"].save_checkpoint(
        checkpoint_name="online_final.pt",
        model=context["model"],
        optimizer=context["optimizer"],
        scheduler=None,
        scaler_state=context["data_bundle"]["scaler"].state_dict(),
        config=experiment_config,
        epoch=len(metric_history),
        metric_history=metric_history,
        extra_state={
            "threshold_artifact": context["threshold_artifact"],
            "threshold_artifact_path": context["threshold_artifact_path"],
            "online_variant": context["online_variant"],
            "verification_buffer_size": len(context["verification_buffer"]),
            "ttl_buffer_size": len(context["ttl_buffer"]),
            "verification_buffer_entries": context["verification_buffer"].items(),
            "hard_old_guard_intervals": context["hard_old_guard"].intervals(),
        },
    )
    return {
        "benchmark_status": "completed",
        "created_at_utc": context["created_at_utc"],
        "online_variant": context["online_variant"],
        "threshold_artifact": context["threshold_artifact"],
        "threshold_artifact_path": context["threshold_artifact_path"],
        "final_checkpoint_path": str(final_checkpoint_path),
        "metric_history": metric_history,
        "records": records,
        "online_metrics_path": metrics_path,
        "online_records_path": records_path,
        "threshold_source": "clean_validation_stride1_ewma",
    }


def run_thesis_online_tta_experiment(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
    dry_run: bool,
) -> dict[str, Any]:
    if online_variant not in {"A0", "A1", "A2"}:
        raise ValueError("online_variant must be one of: A0, A1, A2")

    seed_everything(int(experiment_config["seed"]))
    register_online_runtime_components()
    context = _build_online_execution_context(
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        dry_run=dry_run,
    )
    if context["benchmark_status"] == "dry_run":
        return context

    metric_history, records = _run_online_execution_sequences(
        context=context,
        protocol_config=protocol_config,
    )
    return _finalize_online_execution(
        context=context,
        experiment_config=experiment_config,
        metric_history=metric_history,
        records=records,
    )

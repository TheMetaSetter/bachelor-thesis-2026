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
from src.core.artifact_integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from src.core.registry import build_dataset, build_model
from src.core.runtime_components import register_online_runtime_components
from src.core.seed import seed_everything
from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta.online_calibration import (
    build_online_stream as _build_online_stream,
    collect_clean_validation_scores as _collect_clean_validation_scores,
    move_batch_to_device as _move_batch_to_device,
)
from src.engine.online_tta.online_losses import (
    compute_a1_pnn_reconstruction_loss,
    compute_a2_hard_old_reconstruction_loss,
    compute_hard_old_hinge_loss,
    compute_masked_pnn_reconstruction_loss,
    compute_token_multi_positive_info_nce,
)
from src.engine.online_tta.online_optimizer import (
    assert_only_projector_is_trainable,
    build_online_optimizer,
    clip_projector_gradients,
    collect_projector_parameters,
)
from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.signature_verification import (
    PrototypeVerificationMetadata,
    SignatureWindow,
    build_pnn_token_mask,
    filter_known_anomaly_tokens,
    find_recurrent_signatures,
    ordered_continuous_signature,
    signature_window_to_dict,
)
from src.engine.online_tta.triage import classify_online_window
from src.engine.online_tta.ttl_buffer import TTLBuffer
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_cycle import VerificationCycleController
from src.engine.online_tta.verification_adapter import (
    VerificationResult,
    build_entry_batch,
    verify_buffer_entries,
)
from src.engine.online_tta.runtime_state import (
    OnlineRuntimeState,
    build_online_runtime_state,
)
from src.engine.thresholding import (
    select_clean_validation_point_threshold,
    select_online_ewma_threshold,
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


def _validate_single_window_online_batch(batch: dict[str, Any]) -> None:
    # The online benchmark is defined on one causal window at a time. That
    # assumption is easy to miss because other parts of the code can still score
    # batched windows, so we fail fast here instead of silently dropping items.
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
    runtime_state: OnlineRuntimeState,
    previous_ewma_score: float | None,
    signature_history: list[SignatureWindow],
    recurrent_signatures: set[tuple[int, ...]],
    record: dict[str, Any],
    hard_old_guard: NonOverlapGuard,
    verification_buffer: VerificationBuffer,
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
        input_window_threshold=float(
            np.quantile(calibration_scores["input_window"], 0.99)
        ),
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


def _build_triage_thresholds(
    online_ewma_threshold: float,
    threshold_artifact: dict[str, Any] | None = None,
) -> dict[str, float]:
    if threshold_artifact is not None:
        thresholds = threshold_artifact["thresholds"]
        return {
            "input_window_threshold": float(thresholds["input_window"]["value"]),
            "latent_window_low_threshold": float(
                thresholds["latent_window_low"]["value"]
            ),
            "latent_window_high_threshold": float(
                thresholds["latent_window_high"]["value"]
            ),
        }
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
    signature_history: list[SignatureWindow],
    verification_controller: VerificationCycleController,
    hard_old_guard: NonOverlapGuard,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    event = _prepare_online_window_event(
        model=model,
        batch=batch,
        online_variant=online_variant,
        previous_ewma_score=previous_ewma_score,
        ewma_current_weight=ewma_current_weight,
        ewma_previous_weight=ewma_previous_weight,
        triage_thresholds=triage_thresholds,
        signature_history=signature_history,
        hard_old_guard=hard_old_guard, device=device,
    )
    admitted, rejected = _admit_and_verify_online_window(
        model=model,
        event=event,
        online_variant=online_variant,
        threshold_value=threshold_value,
        verification_buffer=verification_buffer,
        ttl_buffer=ttl_buffer,
        verification_controller=verification_controller,
        device=device,
    )
    step_result = _execute_window_event_step(
        model=model,
        optimizer=optimizer,
        event=event,
        online_variant=online_variant,
        threshold_value=threshold_value,
        hard_old_guard=hard_old_guard,
    )
    record, metric = _build_event_window_outputs(step_result, event, threshold_value, verification_buffer, ttl_buffer)
    return _finalize_window_result(event, metric, record, admitted, rejected)


def _finalize_window_result(
    event: dict[str, Any],
    metric: dict[str, Any],
    record: dict[str, Any],
    admitted: int,
    rejected: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    metric.update(event["signature_diagnostics"])
    metric["online/num_buffer_admitted_windows"] = int(admitted)
    metric["online/num_buffer_rejected_overlap_windows"] = int(rejected)
    return event["ewma_point_score"], metric, record


def _build_event_window_outputs(
    step_result: dict[str, Any],
    event: dict[str, Any],
    threshold_value: float,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _build_online_window_outputs(
        step_result=step_result,
        threshold_value=threshold_value,
        raw_point_score=event["raw_point_score"],
        input_window_score=event["input_window_score"],
        ewma_point_score=event["ewma_point_score"],
        triage_decision=event["triage_decision"],
        verification_buffer=verification_buffer,
        ttl_buffer=ttl_buffer,
    )


def _prepare_online_window_event(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    online_variant: str,
    previous_ewma_score: float | None,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    triage_thresholds: dict[str, float],
    signature_history: list[SignatureWindow],
    hard_old_guard: NonOverlapGuard,
    device: str,
) -> dict[str, Any]:
    (
        batch_on_device,
        raw_point_score,
        input_window_score,
        latent_window_score,
        ewma_point_score,
        scoring_outputs,
    ) = _score_online_window(
        model=model,
        batch=batch,
        online_variant=online_variant,
        previous_ewma_score=previous_ewma_score,
        ewma_current_weight=ewma_current_weight,
        ewma_previous_weight=ewma_previous_weight,
        device=device,
    )
    signature_diagnostics, recurrent_signatures = _attach_event_pnn_mask(
        model, scoring_outputs, batch_on_device, online_variant, signature_history
    )
    triage_decision = _classify_event_window(
        input_window_score=input_window_score,
        latent_window_score=latent_window_score,
        thresholds=triage_thresholds,
        batch=batch_on_device,
        hard_old_guard=hard_old_guard,
    )
    return {
        "batch": batch_on_device,
        "raw_point_score": raw_point_score,
        "input_window_score": input_window_score,
        "latent_window_score": latent_window_score,
        "ewma_point_score": ewma_point_score,
        "triage_decision": triage_decision,
        "signature_diagnostics": signature_diagnostics,
        "recurrent_signatures": recurrent_signatures,
    }


def _admit_and_verify_online_window(
    *,
    model: torch.nn.Module,
    event: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    raw_point_score: float,
    input_window_score: float,
    latent_window_score: float,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
    verification_controller: VerificationCycleController,
    device: str,
) -> tuple[int, int]:
    admitted, rejected = _update_online_window_buffers(
        batch_on_device=event["batch"],
        raw_point_score=event["raw_point_score"],
        input_window_score=event["input_window_score"],
        latent_window_score=event["latent_window_score"],
        triage_decision=event["triage_decision"],
        verification_buffer=verification_buffer,
        ttl_buffer=ttl_buffer,
    )
    verification_controller.maybe_run(
        lambda entries: _verify_and_adapt_entries(
            model=model,
            entries=entries,
            online_variant=online_variant,
            threshold_value=threshold_value,
            device=device,
        )
    )
    return admitted, rejected


def _execute_window_event_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    event: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    ewma_point_score: float,
    input_window_score: float,
    latent_window_score: float,
    triage_decision: str,
    hard_old_guard: NonOverlapGuard,
) -> dict[str, Any]:
    event_optimizer = optimizer
    if online_variant != "A0":
        event_optimizer = build_online_optimizer(model)
    step_result = execute_online_tta_step(
        model=model,
        optimizer=event_optimizer,
        batch=event["batch"],
        online_variant=online_variant,
        threshold_value=threshold_value,
        ewma_point_score=event["ewma_point_score"],
        raw_point_score=event["input_window_score"],
        latent_window_score=event["latent_window_score"],
        triage_decision=event["triage_decision"],
    )
    if step_result["did_update"] and event["triage_decision"] == "hard_old_normality":
        hard_old_guard.add(_window_interval(event["batch"]))
    return step_result


def _window_interval(batch: dict[str, Any]) -> tuple[int, int]:
    meta = batch["meta"][0]
    return int(meta["start_index"]), int(meta["end_index"])


def _attach_event_pnn_mask(
    model: torch.nn.Module,
    scoring_outputs: dict[str, Any],
    batch: dict[str, Any],
    online_variant: str,
    signature_history: list[SignatureWindow],
) -> tuple[dict[str, int], set[tuple[int, ...]]]:
    if online_variant == "A0":
        return {}, set()
    pnn_mask, diagnostics = _build_event_pnn_mask(
        model=model,
        scoring_outputs=scoring_outputs,
        batch=batch,
        signature_history=signature_history,
    )
    if pnn_mask is not None:
        batch["pnn_mask"] = pnn_mask
    recurrent_signatures = find_recurrent_signatures(signature_history)
    return diagnostics, recurrent_signatures


def _classify_event_window(
    *,
    input_window_score: float,
    latent_window_score: float,
    thresholds: dict[str, float],
    batch: dict[str, Any],
    hard_old_guard: NonOverlapGuard,
) -> str:
    decision = classify_online_window(
        input_window_score=input_window_score,
        latent_window_score=latent_window_score,
        thresholds=thresholds,
    )
    if decision == "hard_old_normality" and not hard_old_guard.accept(
        _window_interval(batch)
    ):
        return "gray_zone"
    return decision


def _verify_and_adapt_entries(
    *,
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    online_variant: str,
    threshold_value: float,
    device: str,
) -> dict[str, VerificationResult]:
    """Verify the shared buffer and adapt only entries with a non-empty PNN mask."""
    candidates = verify_buffer_entries(model, entries, device)
    finalized: dict[str, VerificationResult] = {}
    for entry in entries:
        entry_id = str(entry["entry_id"])
        candidate = candidates[entry_id]
        if online_variant == "A0" or not candidate.adapted:
            finalized[entry_id] = VerificationResult(
                False,
                candidate.pseudo_normal_points,
                candidate.reason,
                candidate.pnn_mask,
            )
            continue
        batch = build_entry_batch(entry, device)
        batch["pnn_mask"] = candidate.pnn_mask.to(device)
        if candidate.recurrent_signature_ids is not None:
            batch["recurrent_signature_ids"] = (
                candidate.recurrent_signature_ids.to(device)
            )
        if candidate.known_anomaly_mask is not None:
            batch["known_anomaly_mask"] = candidate.known_anomaly_mask.to(device)
        step = execute_online_tta_step(
            model=model,
            optimizer=build_online_optimizer(model),
            batch=batch,
            online_variant=online_variant,
            threshold_value=threshold_value,
            ewma_point_score=float(entry["point_score"]),
            raw_point_score=float(entry["input_window_score"]),
            latent_window_score=float(entry["latent_window_score"]),
            triage_decision="pnn_verified",
        )
        finalized[entry_id] = VerificationResult(
            bool(step["did_update"]),
            candidate.pseudo_normal_points,
            "adapted" if step["did_update"] else "update_skipped",
            candidate.pnn_mask,
        )
    return finalized


def _score_online_window(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    online_variant: str,
    previous_ewma_score: float | None,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    device: str,
) -> tuple[dict[str, Any], float, float, float, float, dict[str, Any]]:
    batch_on_device = _move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        if online_variant == "A0" and hasattr(model, "forward_source"):
            pre_outputs = model.forward_source(batch_on_device)
        else:
            pre_outputs = model.forward(batch_on_device)
    raw_point_score = float(pre_outputs["point_scores"][0, -1].detach().cpu())
    latent_value = pre_outputs["aux"].get("latent_window_score")
    if latent_value is None:
        latent_value = pre_outputs["window_scores"]
    latent_window_score = float(torch.as_tensor(latent_value).mean().detach().cpu())
    input_window_score = float(
        torch.mean((pre_outputs["recon"] - batch_on_device["x"]) ** 2).detach().cpu()
    )
    if previous_ewma_score is None:
        ewma_point_score = raw_point_score
    else:
        ewma_point_score = (
            ewma_current_weight * raw_point_score
            + ewma_previous_weight * previous_ewma_score
        )
    return (
        batch_on_device,
        raw_point_score,
        input_window_score,
        latent_window_score,
        ewma_point_score,
        pre_outputs,
    )


def _build_event_pnn_mask(
    *,
    model: torch.nn.Module,
    scoring_outputs: dict[str, Any],
    batch: dict[str, Any],
    signature_history: list[SignatureWindow],
) -> tuple[torch.Tensor | None, dict[str, int]]:
    """Build a causal PNN mask from frozen prototype buffers."""
    hidden = scoring_outputs["aux"].get("reference_hidden")
    if hidden is None:
        hidden = scoring_outputs["hidden"]
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("online scoring outputs must expose frozen source hidden")
    hidden = hidden.detach()
    reference = getattr(model, "reference_encoder", None)
    inner_model = getattr(reference, "model", None)
    metadata = PrototypeVerificationMetadata.from_model(inner_model)
    codebook = metadata.codebook
    prototypes = getattr(inner_model, "continuous_prototype_bank", None)
    if not isinstance(prototypes, torch.Tensor):
        raise ValueError("online reference model lacks continuous prototype bank")
    known_anomaly = filter_known_anomaly_tokens(
        hidden,
        codebook.to(hidden.device),
        metadata.anomalous_codeword_mask.to(hidden.device),
        metadata.anomaly_radii.to(hidden.device),
    )
    signatures = ordered_continuous_signature(hidden, prototypes, topk=3)
    meta = batch["meta"][0]
    window = SignatureWindow(
        str(meta["entity_id"]),
        int(meta["start_index"]),
        int(meta["end_index"]),
        signatures,
    )
    recurrent = find_recurrent_signatures([*signature_history, window])
    signature_history.append(window)
    mask = build_pnn_token_mask(signatures, recurrent, known_anomaly)
    return mask, {
        "online/num_points_removed_by_discrete_anom_filter": int(
            known_anomaly.sum().item()
        ),
        "online/num_points_remaining_for_signature": int((~known_anomaly).sum().item()),
        "online/num_recurrent_signatures": len(recurrent),
        "online/num_pseudo_new_normality_points": int(mask.sum().item()),
    }


def _update_online_window_buffers(
    *,
    batch_on_device: dict[str, Any],
    raw_point_score: float,
    input_window_score: float,
    latent_window_score: float,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
) -> tuple[bool, bool]:
    admitted = False
    rejected = False
    if triage_decision == "gray_zone":
        admitted = verification_buffer.try_admit(
            {
                "entry_id": f"window-{int(batch_on_device['meta'][0]['stream_step'])}",
                "window_start": int(batch_on_device["meta"][0]["start_index"]),
                "window_end": int(batch_on_device["meta"][0]["end_index"]),
                "point_score": raw_point_score,
                "input_window_score": input_window_score,
                "latent_window_score": latent_window_score,
                "entity_id": str(batch_on_device["meta"][0]["entity_id"]),
                "stream_step": int(batch_on_device["meta"][0]["stream_step"]),
                "window": batch_on_device["x"][0].detach().cpu().tolist(),
            }
        )
        rejected = not admitted
    if triage_decision != "strong_anomaly":
        ttl_buffer.add(
            item=int(batch_on_device["meta"][0]["end_index"]) - 1,
            current_step=int(batch_on_device["meta"][0]["stream_step"]),
        )
    return admitted, rejected


def _build_online_window_outputs(
    *,
    step_result: dict[str, Any],
    threshold_value: float,
    raw_point_score: float,
    input_window_score: float,
    ewma_point_score: float,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
    ttl_buffer: TTLBuffer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = dict(step_result["record"])
    record["threshold"] = float(threshold_value)
    record["input_window_score"] = float(input_window_score)
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
        "online/input_window_score": input_window_score,
        "online/latent_window_score": float(
            step_result["record"].get("latent_window_score", 0.0)
        ),
        "online/num_buffer_admitted_windows": len(verification_buffer),
        "online/num_buffer_rejected_overlap_windows": 0,
        "online/num_points_removed_by_discrete_anom_filter": 0,
        "online/num_points_remaining_for_signature": 0,
        "online/num_recurrent_signatures": 0,
        "online/num_pseudo_new_normality_points": 0,
        "online/loss_hard_recon": record.get("reconstruction_loss")
        if triage_decision == "hard_old_normality"
        else None,
        "online/loss_pnn_recon": record.get("reconstruction_loss")
        if triage_decision == "pnn_verified"
        else None,
        "online/loss_contrastive": record.get("contrastive_loss"),
        "online/projector_grad_norm": record.get("projector_grad_norm"),
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
    online_variant: str,
    ewma_point_score: float | None,
    raw_point_score: float | None,
    latent_window_score: float | None,
) -> tuple[float, float, float]:
    if raw_point_score is None or latent_window_score is None:
        model.eval()
        with torch.no_grad():
            if online_variant == "A0" and hasattr(model, "forward_source"):
                scoring_outputs = model.forward_source(batch)
            else:
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
                )
                .mean()
                .detach()
                .cpu()
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
        if triage_decision != "pnn_verified":
            return None
        pnn_mask = batch.get("pnn_mask")
        if pnn_mask is not None and (
            not isinstance(pnn_mask, torch.Tensor) or not bool(pnn_mask.any())
        ):
            return None
        if isinstance(pnn_mask, torch.Tensor):
            loss_total = compute_masked_pnn_reconstruction_loss(
                training_outputs["recon"], batch["x"], pnn_mask
            )
        else:
            loss_total = compute_a1_pnn_reconstruction_loss(
                training_outputs["recon"], batch["x"], mask=batch.get("mask")
            )
        reconstruction_loss = loss_total
        contrastive_loss = loss_total.new_zeros(())
    elif online_variant == "A2":
        if triage_decision == "pnn_verified":
            pnn_mask = batch.get("pnn_mask")
            if not isinstance(pnn_mask, torch.Tensor) or not bool(pnn_mask.any()):
                return None
            reconstruction_loss = compute_masked_pnn_reconstruction_loss(
                training_outputs["recon"], batch["x"], pnn_mask
            )
        elif triage_decision == "hard_old_normality":
            reconstruction_loss = compute_hard_old_hinge_loss(
                training_outputs["window_scores"].mean(), threshold_value
            )
        else:
            return None
        contrastive_loss = reconstruction_loss.new_zeros(())
        if triage_decision in {"hard_old_normality", "pnn_verified"}:
            source_model = model.reference_encoder.model
            metadata = PrototypeVerificationMetadata.from_model(source_model)
            anomalous_codewords = metadata.codebook[
                metadata.anomalous_codeword_mask
            ].to(reconstruction_loss.device)
            contrastive_loss = compute_token_multi_positive_info_nce(
                training_outputs["aux"]["projected_hidden"],
                training_outputs["aux"]["reference_hidden"],
                anomalous_codewords,
                pnn_mask=batch.get("pnn_mask")
                if triage_decision == "pnn_verified"
                else None,
                recurrent_signature_ids=batch.get("recurrent_signature_ids"),
                known_anomaly_mask=batch.get("known_anomaly_mask"),
            )
            loss_total = reconstruction_loss + 0.1 * contrastive_loss
        else:
            loss_total = reconstruction_loss
    else:
        raise ValueError("online_variant must be one of: A0, A1, A2")

    loss_total.backward()
    gradient_norm = clip_projector_gradients(model)
    optimizer.step()
    model._last_online_diagnostics = {
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "projector_grad_norm": gradient_norm,
    }
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
        online_variant=online_variant,
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
    record.update(getattr(model, "_last_online_diagnostics", {}))
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
    runtime_state: OnlineRuntimeState | None = None,
    max_online_steps: int | None,
    hard_old_guard: NonOverlapGuard | None = None,
    signature_history: list[SignatureWindow] | None = None,
    threshold_artifact: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hard_old_guard is None:
        hard_old_guard = NonOverlapGuard(max_size=1)
    if signature_history is None:
        signature_history = []
    if runtime_state is None:
        runtime_state = build_online_runtime_state(
            entity_id=str(sequence["meta"]["entity_id"]),
            online_variant=online_variant,
            threshold_artifact={
                "entity_id": str(sequence["meta"]["entity_id"]),
                "thresholds": {},
            },
        )
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
    verification_controller = VerificationCycleController(verification_buffer)
    ewma_current_weight = float(protocol_config["online_ewma_current_weight"])
    ewma_previous_weight = float(protocol_config["online_ewma_previous_weight"])
    triage_thresholds = _build_triage_thresholds(threshold_value, threshold_artifact)

    for batch in batcher:
        _validate_single_window_online_batch(batch)
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
            signature_history=signature_history,
            verification_controller=verification_controller,
            hard_old_guard=hard_old_guard,
        )
        metric["online/step"] = len(metric_history) + 1
        _sync_online_runtime_state(
            runtime_state=runtime_state,
            previous_ewma_score=previous_ewma_score,
            signature_history=signature_history,
            recurrent_signatures=find_recurrent_signatures(signature_history),
            record=record,
            hard_old_guard=hard_old_guard,
            verification_buffer=verification_buffer,
        )
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
    threshold_artifacts = calibrate_entity_threshold_artifacts(
        model=model,
        clean_validation_sequences=data_bundle["scaled_sequences"]["val"],
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        device=str(experiment_config["device"]),
    )
    output_dir = Path(str(experiment_config["output_dir"]))
    threshold_paths: dict[str, str] = {}
    for entity_id, artifact in threshold_artifacts.items():
        path = output_dir / "thresholds" / entity_id / "online_thresholds.json"
        write_threshold_artifact(artifact, path)
        threshold_paths[entity_id] = str(path)
    first_entity = next(iter(threshold_artifacts))
    threshold_artifact = threshold_artifacts[first_entity]
    threshold_path = threshold_paths[first_entity]
    batch_size = int(experiment_config["data"]["batch_size"])
    if batch_size != 1:
        raise ValueError(
            "online benchmark startup expects batch_size=1 so each step is one causal window"
        )
    # The runtime state keeps the causal cursor and EWMA trail alongside the
    # artifact provenance. That makes resume validation explicit instead of
    # relying on hidden local variables.
    runtime_state = build_online_runtime_state(
        entity_id=str(threshold_artifact["entity_id"]),
        online_variant=online_variant,
        threshold_artifact=threshold_artifact,
        checkpoint_path=str(checkpoint_manager.checkpoint_dir / "online_final.pt"),
        threshold_artifact_path=str(threshold_path),
    )
    return {
        "benchmark_status": "completed",
        "created_at_utc": _utc_now_iso(),
        "online_variant": online_variant,
        "data_bundle": data_bundle,
        "model": model,
        "optimizer": optimizer,
        "checkpoint_manager": checkpoint_manager,
        "threshold_artifact": threshold_artifact,
        "threshold_artifacts": threshold_artifacts,
        "threshold_paths": threshold_paths,
        "threshold_artifact_path": str(threshold_path),
        "threshold_value": float(
            threshold_artifact["thresholds"]["online_ewma_point"]["value"]
        ),
        "runtime_state": runtime_state,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "view_noise_std": float(experiment_config["task"].get("view_noise_std", 0.0)),
        "view_dropout_probability": float(
            experiment_config["task"].get("view_dropout_probability", 0.0)
        ),
        "device": str(experiment_config["device"]),
        "max_online_steps": int(experiment_config["task"].get("max_online_steps", 0)),
        "verification_buffer": VerificationBuffer(max_size=64, non_overlap_gap=0),
        "hard_old_guard": NonOverlapGuard(max_size=1),
        "signature_history": [],
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
        if (
            max_online_steps_limit is not None
            and len(metric_history) >= max_online_steps_limit
        ):
            break
        entity_id = str(sequence["meta"]["entity_id"])
        artifact = context["threshold_artifacts"].get(entity_id)
        if artifact is None:
            raise KeyError(f"No threshold artifact for test entity {entity_id}")
        sequence_metric_history, sequence_records = _run_online_sequence(
            model=context["model"],
            optimizer=context["optimizer"],
            sequence=sequence,
            online_variant=context["online_variant"],
            threshold_value=float(artifact["thresholds"]["online_ewma_point"]["value"]),
            protocol_config=protocol_config,
            batch_size=context["batch_size"],
            view_noise_std=context["view_noise_std"],
            view_dropout_probability=context["view_dropout_probability"],
            device=context["device"],
            verification_buffer=context["verification_buffer"],
            ttl_buffer=context["ttl_buffer"],
            runtime_state=context.get("runtime_state"),
            hard_old_guard=context["hard_old_guard"],
            signature_history=context["signature_history"],
            max_online_steps=(
                None
                if max_online_steps_limit is None
                else max_online_steps_limit - len(metric_history)
            ),
            threshold_artifact=artifact,
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
    window_size = int(experiment_config["data"]["window_size"])
    expected_windows = sum(
        max(0, int(sequence["x"].shape[0]) - window_size + 1)
        for sequence in context["data_bundle"]["scaled_sequences"]["test"]
    )
    smoke_limit = context.get("max_online_steps")
    expected_processed = (
        min(expected_windows, int(smoke_limit))
        if smoke_limit is not None and int(smoke_limit) > 0
        else expected_windows
    )
    coverage_status = (
        "complete" if len(records) == expected_processed else "incomplete"
    )
    metrics_path = _write_json(output_dir / "online_metrics.json", metric_history)
    records_path = _write_json(output_dir / "online_records.json", records)
    final_checkpoint_path = context["checkpoint_manager"].save_checkpoint(
        checkpoint_name="online_final.pt",
        model=context["model"],
        optimizer=None,
        scheduler=None,
        scaler_state=context["data_bundle"]["scaler"].state_dict(),
        config=experiment_config,
        epoch=len(metric_history),
        metric_history=metric_history,
        extra_state={
            "threshold_artifact": context["threshold_artifact"],
            "threshold_artifact_path": context["threshold_artifact_path"],
            "online_variant": context["online_variant"],
            "stream_cursor": context["runtime_state"].stream_cursor,
            "previous_ewma_score": context["runtime_state"].previous_ewma_score,
            "signature_history": context["runtime_state"].signature_history,
            "recurrent_signatures": context["runtime_state"].recurrent_signatures,
            "verification_buffer_size": len(context["verification_buffer"]),
            "ttl_buffer_size": len(context["ttl_buffer"]),
            "verification_buffer_entries": context["verification_buffer"].items(),
            "verification_history": context["runtime_state"].verification_history,
            "hard_old_guard_intervals": context["hard_old_guard"].intervals(),
            "online_runtime_state": context["runtime_state"].to_dict(),
        },
    )
    artifact_identity = {
        "entity_id": str(context["threshold_artifact"]["entity_id"]),
        "online_variant": str(context["online_variant"]),
        "experiment_name": str(experiment_config["experiment_name"]),
    }
    artifact_manifest = build_artifact_manifest(
        {
            "checkpoint": final_checkpoint_path,
            "metrics": metrics_path,
            "records": records_path,
            "threshold": context["threshold_artifact_path"],
        },
        identity=artifact_identity,
        provenance={
            "threshold_artifact_path": context["threshold_artifact_path"],
            "threshold_artifact_sha256": CheckpointManager._stable_json_digest(
                context["threshold_artifact"]
            ),
            "resolved_experiment_config_sha256": CheckpointManager._stable_json_digest(
                experiment_config
            ),
            "online_variant": context["online_variant"],
        },
    )
    artifact_manifest_path = write_artifact_manifest(
        output_dir / "online_artifact_manifest.json", artifact_manifest
    )
    expected_provenance = artifact_manifest.get("provenance")
    artifact_integrity_status = (
        "verified"
        if verify_artifact_manifest(
            artifact_manifest,
            artifact_identity,
            expected_provenance=expected_provenance,
        )
        else "failed"
    )
    execution_complete = (
        coverage_status == "complete" and artifact_integrity_status == "verified"
    )
    return {
        "benchmark_status": "completed" if execution_complete else "failed",
        "experiment_status": "complete" if execution_complete else "incomplete",
        "matrix_status": "matrix_ready",
        "runtime_protocol_status": "full_spec_v2",
        "stream_coverage_status": coverage_status,
        "artifact_integrity_status": artifact_integrity_status,
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_path": str(artifact_manifest_path),
        "metric_availability_status": "recorded",
        "expected_windows": expected_windows,
        "processed_windows": len(records),
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

from __future__ import annotations

"""Window scoring and verification helpers for THESIS online TTA."""

from typing import Any

import torch

from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.online_calibration import (
    move_batch_to_device as _move_batch_to_device,
)
from src.engine.online_tta.timing_debug import OnlineTtaTimingLogger
from src.engine.online_tta.online_engine_step import execute_online_tta_step
from src.engine.online_tta.online_optimizer import build_online_optimizer
from src.engine.online_tta.point_ewma import update_window_point_ewma
from src.engine.online_tta.verification_adapter import (
    VerificationResult,
    build_entry_batch,
    verify_buffer_entries,
)
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_cycle import VerificationCycleController


def _verify_and_adapt_entries(
    *,
    model: torch.nn.Module,
    entries: list[dict[str, Any]],
    online_variant: str,
    threshold_value: float,
    device: str,
    source_hidden_by_entry_id: dict[str, torch.Tensor] | None = None,
) -> dict[str, VerificationResult]:
    candidates = verify_buffer_entries(
        model,
        entries,
        device,
        source_hidden_by_entry_id=source_hidden_by_entry_id,
    )
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
            batch["recurrent_signature_ids"] = candidate.recurrent_signature_ids.to(
                device
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
    previous_ewma_point_scores: dict[int, float],
    ewma_current_weight: float,
    ewma_previous_weight: float,
    device: str,
    timing_logger: OnlineTtaTimingLogger | None = None,
) -> tuple[
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    torch.Tensor,
    dict[int, float],
    dict[str, Any],
]:
    timing_logger = timing_logger or OnlineTtaTimingLogger(enabled=False, device=device)
    batch_on_device = timing_logger.measure(
        "host_to_cuda", lambda: _move_batch_to_device(batch, device)
    )
    model.eval()
    pre_outputs = timing_logger.measure(
        "model_forward",
        lambda: _forward_online_window(model, batch_on_device, online_variant),
    )
    window_point_scores, raw_point_scores, input_window_score, latent_window_score = (
        timing_logger.measure(
            "score_extraction",
            lambda: _extract_online_window_scores(pre_outputs, batch_on_device),
        )
    )

    current_window_ewma_point_scores, active_ewma_point_scores = (
        update_window_point_ewma(
            previous_scores=previous_ewma_point_scores,
            absolute_indices=batch_on_device["absolute_indices"][0],
            window_point_scores=window_point_scores,
            current_weight=ewma_current_weight,
            previous_weight=ewma_previous_weight,
        )
    )

    return (
        batch_on_device,
        window_point_scores,
        raw_point_scores,
        input_window_score,
        latent_window_score,
        current_window_ewma_point_scores,
        active_ewma_point_scores,
        pre_outputs,
    )


def _forward_online_window(
    model: torch.nn.Module,
    batch_on_device: dict[str, Any],
    online_variant: str,
) -> dict[str, Any]:
    with torch.no_grad():
        if online_variant == "A0" and hasattr(model, "forward_source"):
            return model.forward_source(batch_on_device)
        return model.forward(batch_on_device)


def _extract_online_window_scores(
    outputs: dict[str, Any], batch_on_device: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    window_point_scores = outputs["point_scores"][0].detach()
    scoring_aux = outputs["aux"].get("scoring", outputs["aux"])
    raw_point_scores = scoring_aux.get("raw_point_scores")
    if not isinstance(raw_point_scores, torch.Tensor):
        raise ValueError("online model must expose raw point scores under aux.scoring")
    raw_point_scores = raw_point_scores[0].detach()
    latent_value = outputs["aux"].get("latent_window_score")
    if latent_value is None:
        latent_value = outputs["window_scores"]
    latent_window_score = float(torch.as_tensor(latent_value).mean().detach().cpu())
    input_window_score = float(
        torch.mean((outputs["recon"] - batch_on_device["x"]) ** 2).detach().cpu()
    )
    return (
        window_point_scores,
        raw_point_scores,
        input_window_score,
        latent_window_score,
    )


def _update_online_window_buffers(
    *,
    batch_on_device: dict[str, Any],
    raw_point_score: float,
    input_window_score: float,
    latent_window_score: float,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
) -> tuple[bool, bool]:
    admitted = False
    rejected = False
    if triage_decision == "gray_zone":
        admitted = verification_buffer.try_admit(
            {
                "entry_id": f"window-{int(batch_on_device['meta'][0]['stream_step'])}",
                "start_index": int(batch_on_device["meta"][0]["start_index"]),
                "end_index": int(batch_on_device["meta"][0]["end_index"]),
                "window_start": int(batch_on_device["meta"][0]["start_index"]),
                "window_end": int(batch_on_device["meta"][0]["end_index"]),
                "point_score": raw_point_score,
                "input_window_score": input_window_score,
                "latent_window_score": latent_window_score,
                "entity_id": str(batch_on_device["meta"][0]["entity_id"]),
                "stream_step": int(batch_on_device["meta"][0]["stream_step"]),
                "window": batch_on_device["x"][0].detach().cpu().tolist(),
                "x": batch_on_device["x"][0].detach().cpu().tolist(),
                "admitted_at_cursor": int(batch_on_device["meta"][0]["stream_step"]),
            }
        )
        rejected = not admitted
    return admitted, rejected


def _build_online_window_outputs(
    *,
    step_result: dict[str, Any],
    threshold_value: float,
    absolute_indices: torch.Tensor,
    window_point_scores: torch.Tensor,
    raw_point_scores: torch.Tensor,
    input_window_score: float,
    current_window_ewma_point_scores: torch.Tensor,
    triage_decision: str,
    verification_buffer: VerificationBuffer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = dict(step_result["record"])
    record["causal_window"] = {
        "absolute_indices": [
            int(index) for index in absolute_indices.detach().cpu().tolist()
        ]
    }
    record["window_point_scores"] = [
        float(score) for score in window_point_scores.detach().cpu().tolist()
    ]
    record["raw_window_point_scores"] = [
        float(score) for score in raw_point_scores.detach().cpu().tolist()
    ]
    record["current_window_ewma_point_scores"] = [
        float(score)
        for score in current_window_ewma_point_scores.detach().cpu().tolist()
    ]
    record["window_point_predictions"] = [
        int(score > threshold_value)
        for score in current_window_ewma_point_scores.detach().cpu().tolist()
    ]
    record["threshold"] = float(threshold_value)
    record["input_window_score"] = float(input_window_score)
    record["verification_cycle_ready"] = verification_buffer.should_verify()
    metric = {
        "online/step": 0,
        "online/raw_point_score": float(raw_point_scores[-1].detach().cpu()),
        "online/ewma_point_score": float(
            current_window_ewma_point_scores[-1].detach().cpu()
        ),
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
    }
    return record, metric

from __future__ import annotations

"""Window scoring and verification helpers for THESIS online TTA."""

from typing import Any

import torch

from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.online_calibration import move_batch_to_device as _move_batch_to_device
from src.engine.online_tta.online_engine_step import execute_online_tta_step
from src.engine.online_tta.online_optimizer import build_online_optimizer
from src.engine.online_tta.signature_verification import (
    PrototypeVerificationMetadata,
    SignatureWindow,
    build_pnn_token_mask,
    filter_known_anomaly_tokens,
    find_recurrent_signatures,
    ordered_continuous_signature,
)
from src.engine.online_tta.ttl_buffer import TTLBuffer
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
) -> dict[str, VerificationResult]:
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

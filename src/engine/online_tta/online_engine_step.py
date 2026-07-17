from __future__ import annotations

"""Single-step online TTA update helpers."""

from typing import Any

import torch

from src.engine.online_tta.online_losses import (
    compute_a1_pnn_reconstruction_loss,
    compute_a2_hard_old_reconstruction_loss,
    compute_hard_old_hinge_loss,
    compute_masked_pnn_reconstruction_loss,
    compute_token_multi_positive_info_nce,
)
from src.engine.online_tta.online_optimizer import clip_projector_gradients
from src.engine.online_tta.signature_verification import PrototypeVerificationMetadata


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

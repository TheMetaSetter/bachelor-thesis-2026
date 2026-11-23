"""Apply MUSTARD online update rules to windows that have already been scored."""

import torch

from .contrastive import compute_token_multi_positive_info_nce
from .signature_verification import (
    SignatureWindow,
    build_pnn_token_mask,
    filter_known_anomaly_tokens,
    find_recurrent_signatures,
    ordered_continuous_signature,
)
from ...scoring import reconstruction_scores


def _result(loss=None, *, reason):
    return {"did_update": loss is not None, "loss": loss, "reason": reason}


def _verified_entries(method, entries):
    records, windows = [], []
    for entry in entries:
        with torch.no_grad():
            hidden = method.encoder(entry["x"])
            anomalies = filter_known_anomaly_tokens(
                hidden,
                method.memory["discrete"],
                method.memory["anomalous_mask"],
                method.memory["radii"],
            )
            signatures = ordered_continuous_signature(
                hidden,
                method.memory["continuous"],
                topk=method.parameters_config["verification"]["signature_top_k"],
            )
        windows.append(
            SignatureWindow(
                method.config.get("entity", ""),
                entry["window_start"],
                entry["window_end"],
                signatures,
            )
        )
        records.append((entry, signatures, anomalies))
    recurrent = find_recurrent_signatures(windows)
    for entry, signatures, anomalies in records:
        yield (
            entry,
            build_pnn_token_mask(signatures, recurrent, anomalies),
            signatures,
            anomalies,
        )


def _verification_cycle(method, batch, context):
    positions = batch["indices"]
    start, stop = int(positions[0, 0]), int(positions[0, -1]) + 1
    buffer = method.verification_buffer
    entry = {
        "entry_id": str(start),
        "window_start": start,
        "window_end": stop,
        "x": batch["x"].detach().clone(),
    }
    if not buffer.try_admit(entry):
        return _result(reason="overlap")
    if not buffer.should_verify():
        return _result(reason="waiting_for_verification")
    losses = []
    for candidate, mask, signatures, anomalies in _verified_entries(
        method, buffer.items()
    ):
        outcome = _apply_update(
            method,
            candidate["x"],
            context
            | {
                "decision": "pnn_verified",
                "mask": mask,
                "signatures": signatures,
                "anomalous_mask": anomalies,
            },
        )
        buffer.mark_verification_result(candidate["entry_id"], outcome["did_update"])
        if outcome["did_update"]:
            losses.append(outcome["loss"])
    buffer.finish_verification_cycle()
    return _result(
        sum(losses) / len(losses) if losses else None, reason="verification_cycle"
    )


def _apply_update(method, x, context):
    parameters = method.parameters_config
    settings = method.config["method"]["update"]
    mask = context.get("mask")
    decision = context["decision"]
    if decision == "pnn_verified" and (mask is None or not bool(mask.any())):
        return _result(reason="empty_verification_mask")
    if "recorded_score" not in context:
        raise ValueError("MUSTARD update requires a recorded pre-update score")
    loss = None
    for _ in range(settings["steps"]):
        output = method._forward(x)
        reconstruction = output["reconstruction"]
        if decision == "pnn_verified":
            loss = (reconstruction - x).square()[mask.bool()].mean()
        else:
            errors = reconstruction_scores(
                x,
                reconstruction,
                {
                    "score_space": "raw_input",
                    "scaler": method.config["scaler"],
                },
            )
            loss = (
                (errors.mean() - context["calibration"]["input_window_threshold"])
                .clamp_min(0)
                .square()
            )
        if method.variant == "A2":
            signatures = context.get("signatures")
            signature_tensor = (
                None
                if signatures is None
                else torch.as_tensor(signatures, device=x.device)
            )
            loss = loss + parameters[
                "contrastive_weight"
            ] * compute_token_multi_positive_info_nce(
                output["projected"],
                output["hidden"],
                method.memory["discrete"][method.memory["anomalous_mask"]],
                pnn_mask=mask,
                recurrent_signature_ids=signature_tensor,
                known_anomaly_mask=context.get("anomalous_mask"),
                temperature=parameters["contrastive_temperature"],
            )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite MUSTARD update loss")
        method.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if settings["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                method.projector.parameters(),
                settings["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
        method.optimizer.step()
    return _result(float(loss.detach()), reason=decision)


def update_window(method, batch: dict, context: dict) -> dict:
    """Use triage and verification to select a permitted projector update."""
    if method.variant == "A0":
        return _result(reason="frozen")
    if "labels" in batch or "point_labels" in batch:
        raise ValueError("MUSTARD adaptation cannot receive test labels")
    decision = context["decision"]
    if decision == "gray_zone":
        return _verification_cycle(method, batch, context)
    if decision == "pnn_verified" or (
        decision == "hard_old_normality" and method.variant == "A2"
    ):
        return _apply_update(method, batch["x"], context)
    return _result(reason=decision)

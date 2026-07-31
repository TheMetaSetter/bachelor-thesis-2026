from __future__ import annotations

"""Online window orchestration helpers for THESIS."""

from typing import Any

from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.online_optimizer import build_online_optimizer
from src.engine.online_tta.triage import classify_online_window
from src.engine.online_tta.online_engine_step import execute_online_tta_step
from src.engine.online_tta.online_engine_window_metrics import (
    _build_event_pnn_mask,
    _build_online_window_outputs,
    _score_online_window,
    _update_online_window_buffers,
    _verify_and_adapt_entries,
)
from src.engine.online_tta.signature_verification import (
    find_recurrent_signatures,
)
from src.engine.online_tta.timing_debug import OnlineTtaTimingLogger
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_cycle import VerificationCycleController


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
        "strong_anomaly_threshold": threshold,
        "hard_old_normality_threshold": threshold * 0.5,
        "pnn_candidate_input_threshold": threshold * 0.75,
        "pnn_candidate_latent_threshold": threshold * 0.75,
    }


def _process_online_window(
    *,
    model,
    optimizer,
    batch: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    triage_thresholds: dict[str, float],
    verification_buffer: VerificationBuffer,
    previous_ewma_score: float | None,
    device: str,
    signature_history: list[Any],
    verification_controller: VerificationCycleController,
    hard_old_guard: NonOverlapGuard,
    timing_logger: OnlineTtaTimingLogger | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    timing_logger = timing_logger or OnlineTtaTimingLogger(enabled=False, device=device)
    timing_logger.set_window(batch)
    event = timing_logger.measure(
        "prepare_event",
        lambda: _prepare_online_window_event(
            model=model, batch=batch, online_variant=online_variant,
            previous_ewma_score=previous_ewma_score,
            ewma_current_weight=ewma_current_weight,
            ewma_previous_weight=ewma_previous_weight,
            triage_thresholds=triage_thresholds,
            signature_history=signature_history, hard_old_guard=hard_old_guard,
            device=device, timing_logger=timing_logger,
        ),
    )
    admitted, rejected = timing_logger.measure(
        "buffer_and_verification",
        lambda: _admit_and_verify_online_window(
            model=model, event=event, online_variant=online_variant,
            threshold_value=threshold_value, verification_buffer=verification_buffer,
            verification_controller=verification_controller,
            device=device,
        ),
    )
    step_result = timing_logger.measure(
        "adaptation_step",
        lambda: _execute_window_event_step(
            model=model, optimizer=optimizer, event=event,
            online_variant=online_variant, threshold_value=threshold_value,
            hard_old_guard=hard_old_guard,
        ),
    )
    record, metric = timing_logger.measure(
        "build_outputs",
        lambda: _build_event_window_outputs(
            step_result, event, threshold_value, verification_buffer
        ),
    )
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _build_online_window_outputs(
        step_result=step_result,
        threshold_value=threshold_value,
        raw_point_score=event["raw_point_score"],
        input_window_score=event["input_window_score"],
        ewma_point_score=event["ewma_point_score"],
        triage_decision=event["triage_decision"],
        verification_buffer=verification_buffer,
    )


def _prepare_online_window_event(
    *,
    model,
    batch: dict[str, Any],
    online_variant: str,
    previous_ewma_score: float | None,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    triage_thresholds: dict[str, float],
    signature_history: list[Any],
    hard_old_guard: NonOverlapGuard,
    device: str,
    timing_logger: OnlineTtaTimingLogger,
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
        timing_logger=timing_logger,
    )
    signature_diagnostics, recurrent_signatures = timing_logger.measure(
        "pnn_mask",
        lambda: _attach_event_pnn_mask(
            model, scoring_outputs, batch_on_device, online_variant, signature_history
        ),
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
    model,
    event: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    verification_buffer: VerificationBuffer,
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
    model,
    optimizer,
    event: dict[str, Any],
    online_variant: str,
    threshold_value: float,
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
    model,
    scoring_outputs: dict[str, Any],
    batch: dict[str, Any],
    online_variant: str,
    signature_history: list[Any],
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

from __future__ import annotations

"""One causal-window event in the THESIS online runtime."""

from typing import Any

from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.online_engine_step import execute_online_tta_step
from src.engine.online_tta.online_engine_window_metrics import (
    _build_online_window_outputs,
    _score_online_window,
    _update_online_window_buffers,
    _verify_and_adapt_entries,
)
from src.engine.online_tta.online_optimizer import build_online_optimizer
from src.engine.online_tta.timing_debug import OnlineTtaTimingLogger
from src.engine.online_tta.triage import classify_online_window
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_cycle import VerificationCycleController


def _build_triage_thresholds(
    online_ewma_threshold: float,
    threshold_artifact: dict[str, Any] | None = None,
) -> dict[str, float]:
    if threshold_artifact is None:
        raise ValueError("A1/A2 require a validated threshold_artifact")
    thresholds = threshold_artifact["thresholds"]
    return {
        "input_window_threshold": float(thresholds["input_window"]["value"]),
        "latent_window_low_threshold": float(thresholds["latent_window_low"]["value"]),
        "latent_window_high_threshold": float(thresholds["latent_window_high"]["value"]),
    }


def _window_interval(batch: dict[str, Any]) -> tuple[int, int]:
    meta = batch["meta"][0]
    return int(meta["start_index"]), int(meta["end_index"])


def _prepare_online_window_event(
    *,
    model,
    batch: dict[str, Any],
    online_variant: str,
    previous_ewma_point_scores: dict[int, float],
    ewma_current_weight: float,
    ewma_previous_weight: float,
    triage_thresholds: dict[str, float] | None,
    hard_old_guard: NonOverlapGuard,
    device: str,
    timing_logger: OnlineTtaTimingLogger,
) -> dict[str, Any]:
    scored = _score_online_window(
        model=model,
        batch=batch,
        online_variant=online_variant,
        previous_ewma_point_scores=previous_ewma_point_scores,
        ewma_current_weight=ewma_current_weight,
        ewma_previous_weight=ewma_previous_weight,
        device=device,
        timing_logger=timing_logger,
    )
    (
        batch_on_device,
        point_scores,
        input_score,
        latent_score,
        ewma_scores,
        active_scores,
        scoring_outputs,
    ) = scored
    triage_region = None
    hard_old_is_admissible = False
    if online_variant != "A0":
        if triage_thresholds is None:
            raise ValueError("A1/A2 require triage thresholds")
        triage_region = classify_online_window(input_score, latent_score, triage_thresholds)
        hard_old_is_admissible = (
            triage_region == "hard_old_normality"
            and hard_old_guard.accept(_window_interval(batch_on_device))
        )
    return {
        "batch": batch_on_device,
        "window_point_scores": point_scores,
        "input_window_score": input_score,
        "latent_window_score": latent_score,
        "current_window_ewma_point_scores": ewma_scores,
        "active_ewma_point_scores": active_scores,
        "source_hidden": scoring_outputs.get("aux", {}).get("reference_hidden"),
        "triage_region": triage_region,
        "hard_old_is_admissible": hard_old_is_admissible,
    }


def _run_current_window_action(
    *,
    model,
    optimizer,
    event: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    hard_old_guard: NonOverlapGuard,
) -> dict[str, Any]:
    triage_region = event["triage_region"]
    accepted_hard_old = (
        triage_region == "hard_old_normality" and event["hard_old_is_admissible"]
    )
    step = execute_online_tta_step(
        model=model,
        optimizer=optimizer if accepted_hard_old else None,
        batch=event["batch"],
        online_variant=online_variant,
        threshold_value=threshold_value,
        ewma_point_score=float(event["current_window_ewma_point_scores"][-1]),
        raw_point_score=float(event["window_point_scores"][-1]),
        latent_window_score=event["latent_window_score"],
        triage_decision="hard_old_normality" if accepted_hard_old else triage_region,
    )
    if step["did_update"] and accepted_hard_old:
        hard_old_guard.add(_window_interval(event["batch"]))
    return step


def _admit_and_verify_gray_zone(
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
        raw_point_score=float(event["window_point_scores"][-1]),
        input_window_score=event["input_window_score"],
        latent_window_score=event["latent_window_score"],
        triage_decision=event["triage_region"],
        verification_buffer=verification_buffer,
    )
    source_hidden_by_entry_id: dict[str, Any] = {}
    source_hidden = event.get("source_hidden")
    if admitted and source_hidden is not None:
        source_hidden_by_entry_id[
            f"window-{int(event['batch']['meta'][0]['stream_step'])}"
        ] = source_hidden
    verification_controller.maybe_run(
        lambda entries: _verify_and_adapt_entries(
            model=model,
            entries=entries,
            online_variant=online_variant,
            threshold_value=threshold_value,
            device=device,
            source_hidden_by_entry_id=source_hidden_by_entry_id,
        )
    )
    return int(admitted), int(rejected)


def _build_event_outputs(
    event: dict[str, Any],
    step: dict[str, Any],
    threshold_value: float,
    verification_buffer: VerificationBuffer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _build_online_window_outputs(
        step_result=step,
        threshold_value=threshold_value,
        absolute_indices=event["batch"]["absolute_indices"][0],
        window_point_scores=event["window_point_scores"],
        input_window_score=event["input_window_score"],
        current_window_ewma_point_scores=event["current_window_ewma_point_scores"],
        triage_decision=event["triage_region"],
        verification_buffer=verification_buffer,
    )


def _process_online_window(
    *,
    model,
    optimizer,
    batch: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    triage_thresholds: dict[str, float] | None,
    verification_buffer: VerificationBuffer,
    previous_ewma_point_scores: dict[int, float],
    device: str,
    verification_controller: VerificationCycleController,
    hard_old_guard: NonOverlapGuard,
    timing_logger: OnlineTtaTimingLogger | None = None,
) -> tuple[dict[int, float], dict[str, Any], dict[str, Any]]:
    timing_logger = timing_logger or OnlineTtaTimingLogger(enabled=False, device=device)
    timing_logger.set_window(batch)
    event = timing_logger.measure(
        "prepare_event",
        lambda: _prepare_online_window_event(
            model=model, batch=batch, online_variant=online_variant,
            previous_ewma_point_scores=previous_ewma_point_scores,
            ewma_current_weight=ewma_current_weight,
            ewma_previous_weight=ewma_previous_weight,
            triage_thresholds=triage_thresholds, hard_old_guard=hard_old_guard,
            device=device, timing_logger=timing_logger,
        ),
    )
    step = timing_logger.measure(
        "adaptation_step",
        lambda: _run_current_window_action(
            model=model, optimizer=optimizer, event=event, online_variant=online_variant,
            threshold_value=threshold_value, hard_old_guard=hard_old_guard,
        ),
    )
    admitted, rejected = (0, 0)
    if online_variant != "A0":
        admitted, rejected = timing_logger.measure(
            "buffer_and_verification",
            lambda: _admit_and_verify_gray_zone(
                model=model, event=event, online_variant=online_variant,
                threshold_value=threshold_value, verification_buffer=verification_buffer,
                verification_controller=verification_controller, device=device,
            ),
        )
    record, metric = _build_event_outputs(event, step, threshold_value, verification_buffer)
    metric["online/num_buffer_admitted_windows"] = admitted
    metric["online/num_buffer_rejected_overlap_windows"] = rejected
    return event["active_ewma_point_scores"], metric, record

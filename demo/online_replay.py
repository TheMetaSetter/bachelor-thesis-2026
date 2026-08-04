from __future__ import annotations

"""Build online replay state from saved benchmark artifacts."""

from pathlib import Path
from typing import Any

import numpy as np

from demo.demo_state import OnlineReplayState
from demo.loaders import (
    load_demo_test_sequence,
    load_report_payload,
    load_sequence_values,
)


def _iter_completed_windows(
    controller: Any,
    window_size: int,
) -> list[tuple[int, Any, dict[str, Any], list[Any]]]:
    if window_size < 1:
        raise ValueError("window_size must be positive")

    points: list[Any] = []
    completed_windows: list[tuple[int, Any, dict[str, Any], list[Any]]] = []
    for stream_index, item in enumerate(controller):
        value = item.get("x") if isinstance(item, dict) else item
        metadata = item.get("meta", {}) if isinstance(item, dict) else {}
        points.append(value)
        if len(points) < window_size:
            continue
        completed_windows.append(
            (
                stream_index + 1,
                value,
                dict(metadata) if isinstance(metadata, dict) else {},
                list(points[-window_size:]),
            )
        )
    return completed_windows


def consume_online_stream(
    controller: Any,
    window_size: int,
    score_callback: Any,
) -> list[dict[str, Any]]:
    """Consume points and score only completed causal windows."""
    outputs: list[dict[str, Any]] = []
    for end_index, _value, _metadata, window in _iter_completed_windows(
        controller,
        window_size,
    ):
        outputs.append({"end_index": end_index, "score": score_callback(window)})
    return outputs


def run_live_online_replay(
    controller: Any,
    window_size: int,
    score_callback: Any,
) -> list[dict[str, Any]]:
    """Score causal windows without exposing labels to the callback."""
    outputs: list[dict[str, Any]] = []
    for end_index, value, metadata, window in _iter_completed_windows(
        controller,
        window_size,
    ):
        payload = {
            "x": value,
            "meta": dict(metadata) if isinstance(metadata, dict) else {},
            "window": window,
            "end_index": end_index,
        }
        outputs.append({"end_index": end_index, "score": score_callback(payload)})
    return outputs


def _load_threshold_artifact(
    report: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    online_execution = report.get("online_execution")
    if not isinstance(online_execution, dict):
        raise ValueError("Report is missing online_execution")
    artifact = online_execution.get("threshold_artifact")
    if isinstance(artifact, dict):
        return artifact
    artifact_paths = report.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        threshold_path = artifact_paths.get("thresholds")
        if threshold_path is not None:
            return load_report_payload(Path(threshold_path))
    raise ValueError(
        f"Report {report_path} does not contain an online threshold artifact"
    )


def _thesis_vector_series(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    latest_values: dict[int, tuple[float, float, bool]] = {}
    for record in records:
        causal_window = record.get("causal_window")
        if not isinstance(causal_window, dict):
            raise ValueError("THESIS record is missing causal_window")
        absolute_indices = causal_window.get("absolute_indices")
        point_scores = record.get("window_point_scores")
        ewma_scores = record.get("current_window_ewma_point_scores")
        predictions = record.get("window_point_predictions")
        if not all(isinstance(values, list) for values in (
            absolute_indices, point_scores, ewma_scores, predictions
        )):
            raise ValueError("THESIS record is missing a required point vector")
        if not (
            len(absolute_indices)
            == len(point_scores)
            == len(ewma_scores)
            == len(predictions)
        ):
            raise ValueError("THESIS point vectors must have one shared length")
        for index, point_score, ewma_score, prediction in zip(
            absolute_indices, point_scores, ewma_scores, predictions
        ):
            latest_values[int(index)] = (
                float(point_score),
                float(ewma_score),
                bool(prediction),
            )
    indices = np.asarray(sorted(latest_values), dtype=np.int64)
    raw_scores = np.asarray(
        [latest_values[int(index)][0] for index in indices], dtype=np.float64
    )
    ewma_scores = np.asarray(
        [latest_values[int(index)][1] for index in indices], dtype=np.float64
    )
    predictions = np.asarray(
        [latest_values[int(index)][2] for index in indices], dtype=bool
    )
    return indices, raw_scores, ewma_scores, predictions


def _baseline_endpoint_series(
    records: list[dict[str, Any]], threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_scores = np.asarray(
        [record.get("raw_point_score", 0.0) for record in records],
        dtype=np.float64,
    )
    ewma_scores = np.asarray(
        [record.get("ewma_point_score", 0.0) for record in records],
        dtype=np.float64,
    )
    return (
        np.arange(ewma_scores.shape[0], dtype=np.int64),
        raw_scores,
        ewma_scores,
        np.asarray(ewma_scores > threshold, dtype=bool),
    )


def build_online_replay_state(report_path: str | Path) -> OnlineReplayState:
    path = Path(report_path)
    report = load_report_payload(path)
    online_execution = report.get("online_execution")
    if not isinstance(online_execution, dict):
        raise ValueError("Report is missing online_execution")
    threshold_artifact = _load_threshold_artifact(report, path)
    threshold = float(online_execution.get("threshold_value", 0.0))
    test_sequence = load_demo_test_sequence(path)
    raw_values = load_sequence_values(test_sequence)
    records = list(online_execution.get("records", []))
    method = str(online_execution.get("baseline_name", report.get("baseline_name", "unknown")))
    if method == "THESIS":
        score_indices, raw_point_scores, ewma_point_scores, predicted_mask = (
            _thesis_vector_series(records)
        )
    else:
        score_indices, raw_point_scores, ewma_point_scores, predicted_mask = (
            _baseline_endpoint_series(records, threshold)
        )
    metrics = {
        "num_records": len(records),
        "num_metric_rows": len(online_execution.get("metric_history", [])),
        "num_updates": sum(1 for record in records if record.get("did_update")),
        "threshold_value": threshold,
    }
    return OnlineReplayState(
        report_path=path,
        method=method,
        variant=str(
            online_execution.get(
                "online_variant", report.get("online_variant", "unknown")
            )
        ),
        entity_id=str(
            online_execution.get("entity_id", report.get("entity_id", "unknown"))
        ),
        seed=int(report.get("seed", 0)),
        threshold=threshold,
        threshold_source=str(online_execution.get("threshold_source", "unknown")),
        point_rule=str(
            threshold_artifact["thresholds"]["offline_point"].get(
                "score_rule", "unknown"
            )
        ),
        smoothing_rule=str(
            threshold_artifact["thresholds"]["online_ewma_point"].get(
                "score_rule", "none"
            )
        ),
        raw_values=raw_values,
        score_indices=score_indices,
        raw_point_scores=raw_point_scores,
        ewma_point_scores=ewma_point_scores,
        predicted_mask=np.asarray(predicted_mask, dtype=bool),
        records=records,
        threshold_artifact=threshold_artifact,
        metrics=metrics,
    )

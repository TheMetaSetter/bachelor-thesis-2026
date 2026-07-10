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


def consume_online_stream(
    controller: Any,
    window_size: int,
    score_callback: Any,
) -> list[dict[str, Any]]:
    """Consume points and score only completed causal windows."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    points: list[Any] = []
    outputs: list[dict[str, Any]] = []
    for item in controller:
        value = item.get("x") if isinstance(item, dict) else item
        points.append(value)
        if len(points) < window_size:
            continue
        window = points[-window_size:]
        result = score_callback(window)
        outputs.append({"end_index": len(points), "score": result})
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
    raw_point_scores = np.asarray(
        [record.get("raw_point_score", 0.0) for record in records],
        dtype=np.float64,
    )
    ewma_point_scores = np.asarray(
        [record.get("ewma_point_score", 0.0) for record in records],
        dtype=np.float64,
    )
    predicted_mask = ewma_point_scores > threshold
    metrics = {
        "num_records": len(records),
        "num_metric_rows": len(online_execution.get("metric_history", [])),
        "num_updates": sum(1 for record in records if record.get("did_update")),
        "threshold_value": threshold,
    }
    return OnlineReplayState(
        report_path=path,
        method=str(
            online_execution.get(
                "baseline_name", report.get("baseline_name", "unknown")
            )
        ),
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
        raw_point_scores=raw_point_scores,
        ewma_point_scores=ewma_point_scores,
        predicted_mask=np.asarray(predicted_mask, dtype=bool),
        records=records,
        threshold_artifact=threshold_artifact,
        metrics=metrics,
    )

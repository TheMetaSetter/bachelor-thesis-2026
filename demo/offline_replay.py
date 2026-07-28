from __future__ import annotations

"""Build offline replay state from saved benchmark artifacts."""

from pathlib import Path
from typing import Any

import numpy as np

from demo.demo_state import OfflineReplayState
from demo.loaders import (
    load_demo_test_sequence,
    load_report_payload,
    load_sequence_labels,
    load_sequence_values,
)


def _load_threshold_artifact(
    report: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    artifact_paths = report.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError("Report is missing artifact_paths")
    threshold_path = artifact_paths.get("thresholds")
    if threshold_path is None:
        raise ValueError("Report is missing threshold artifact path")
    return load_report_payload(Path(threshold_path))


def build_offline_replay_state(report_path: str | Path) -> OfflineReplayState:
    path = Path(report_path)
    report = load_report_payload(path)
    threshold_artifact = _load_threshold_artifact(report, path)
    threshold = float(threshold_artifact["thresholds"]["offline_point"]["value"])
    test_sequence = load_demo_test_sequence(path)
    artifact_paths = report["artifact_paths"]
    test_scores_path = Path(artifact_paths["test_scores"])
    scores_npz = np.load(test_scores_path)
    point_scores = np.asarray(scores_npz["point_scores"], dtype=np.float64).reshape(-1)
    point_labels = load_sequence_labels(test_sequence)
    raw_values = load_sequence_values(test_sequence)
    predicted_mask = point_scores > threshold
    covered_point_mask = np.asarray(
        scores_npz["covered_point_mask"], dtype=bool
    ).reshape(-1)
    metrics = dict(report.get("offline_metrics", {}))
    return OfflineReplayState(
        report_path=path,
        method=str(
            threshold_artifact.get(
                "method_name", report.get("baseline_name", "unknown")
            )
        ),
        variant=str(
            threshold_artifact.get(
                "variant_name", report.get("baseline_name", "unknown")
            )
        ),
        entity_id=str(
            threshold_artifact.get("entity_id", report.get("entity_id", "unknown"))
        ),
        seed=int(threshold_artifact.get("seed", report.get("seed", 0))),
        threshold=threshold,
        threshold_source=str(
            threshold_artifact["thresholds"]["offline_point"].get(
                "source_split", "unknown"
            )
        ),
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
        point_scores=point_scores,
        point_labels=point_labels,
        predicted_mask=np.asarray(predicted_mask, dtype=bool),
        covered_point_mask=covered_point_mask,
        threshold_artifact=threshold_artifact,
        metrics=metrics,
    )

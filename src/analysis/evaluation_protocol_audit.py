from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.split_protocol import describe_label_regime
from src.data.split_protocol import summarize_split_point_labels


def _collect_positive_spans(binary_values: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, value in enumerate(binary_values.astype(int).tolist()):
        if value == 1 and start_index is None:
            start_index = index
        elif value == 0 and start_index is not None:
            spans.append((start_index, index))
            start_index = None
    if start_index is not None:
        spans.append((start_index, int(binary_values.shape[0])))
    return spans


def describe_metric_regime_implications(
    *,
    label_regime: str,
    threshold: float | None = None,
    observed_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    threshold_note = (
        f"The observed threshold was {threshold:.6f}."
        if threshold is not None
        else "No observed threshold was provided."
    )
    if label_regime == "all_one":
        implications = [
            "Every evaluated timestep is labeled anomalous, so there are no true negatives in the test vector.",
            "Precision can stay at 1.0 as long as the model predicts at least one positive, because false positives are impossible in an all-positive label regime.",
            "Recall becomes the fraction of anomalous timesteps whose scores exceed the threshold, so a low recall simply means the model predicted only a small positive subset.",
            "PR-AUC can appear perfect or otherwise degenerate in this regime and should not be interpreted as normal model-quality evidence.",
            "ROC-AUC becomes undefined because the evaluator sees only one label class.",
            "VUS-PR and VUS-ROC can also become undefined because the range-based metric code expects both positive and negative labels.",
        ]
        if observed_metrics is not None:
            recall = observed_metrics.get("recall")
            if recall is not None and np.isfinite(recall):
                implications.append(
                    f"The observed recall of {recall:.6f} is therefore consistent with a model that flagged only about {recall * 100:.2f}% of an already-all-anomalous test timeline."
                )
        return {
            "regime_name": "all_positive_test_labels",
            "summary": (
                "The evaluated test vector is all anomalous. This is a protocol-special "
                "single-class regime, not a standard full-timeline anomaly-detection test."
            ),
            "implications": implications,
            "threshold_note": threshold_note,
        }
    if label_regime == "all_zero":
        return {
            "regime_name": "all_negative_test_labels",
            "summary": (
                "The evaluated test vector is all normal. This is also a protocol-special "
                "single-class regime, not a standard full-timeline anomaly-detection test."
            ),
            "implications": [
                "There are no positive labels, so recall is not scientifically informative in the usual anomaly-detection sense.",
                "Any predicted positive becomes a false positive, so precision collapses immediately when the model fires.",
                "PR-AUC, ROC-AUC, and VUS-style metrics become degenerate or undefined because only one label class is present.",
            ],
            "threshold_note": threshold_note,
        }
    return {
        "regime_name": "mixed_test_labels",
        "summary": (
            "The evaluated test vector contains both normal and anomalous labels, so the "
            "pointwise metrics are at least defined in the usual binary sense."
        ),
        "implications": [
            "The remaining question is then whether evaluated coverage truly matches the intended raw test timeline."
        ],
        "threshold_note": threshold_note,
    }


def _build_entity_window_coverage(
    split_sequences: list[dict[str, Any]],
    dataset: Any,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    records_by_sequence_index: dict[int, list[tuple[int, int]]] = {}
    for sequence_index, start_index, end_index in getattr(dataset, "index_records", []):
        records_by_sequence_index.setdefault(int(sequence_index), []).append(
            (int(start_index), int(end_index))
        )

    for sequence_index, sequence in enumerate(split_sequences):
        raw_num_points = int(sequence["x"].shape[0])
        index_records = records_by_sequence_index.get(sequence_index, [])
        coverage_mask = np.zeros(raw_num_points, dtype=np.int64)
        for start_index, end_index in index_records:
            coverage_mask[start_index:end_index] = 1
        if index_records:
            covered_indices = np.flatnonzero(coverage_mask)
            evaluated_start_index = int(covered_indices[0])
            evaluated_end_index = int(covered_indices[-1]) + 1
            evaluated_num_points = int(coverage_mask.sum())
        else:
            evaluated_start_index = 0
            evaluated_end_index = 0
            evaluated_num_points = 0
        entities.append(
            {
                "entity_id": sequence["meta"]["entity_id"],
                "raw_num_points": raw_num_points,
                "evaluated_start_index": evaluated_start_index,
                "evaluated_end_index": evaluated_end_index,
                "evaluated_num_points": evaluated_num_points,
                "is_truncated": evaluated_num_points < raw_num_points,
                "coverage_spans": _collect_positive_spans(coverage_mask),
            }
        )
    return entities


def _summarize_entity_coverage(entities: list[dict[str, Any]]) -> dict[str, int]:
    raw_num_points = int(sum(int(entity["raw_num_points"]) for entity in entities))
    evaluated_num_points = int(
        sum(int(entity["evaluated_num_points"]) for entity in entities)
    )
    return {
        "raw_num_points": raw_num_points,
        "evaluated_num_points": evaluated_num_points,
    }


def build_dataset_protocol_audit_report(
    *,
    data_bundle: dict[str, Any],
    data_config: dict[str, Any],
    evaluation_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    warnings: list[str] = []

    for split_name in ("train", "val", "test"):
        split_sequences = data_bundle["raw_sequences"][split_name]
        label_summary = summarize_split_point_labels(split_sequences)
        entities = _build_entity_window_coverage(
            split_sequences,
            data_bundle["datasets"][split_name],
        )
        coverage_summary = _summarize_entity_coverage(entities)
        configured_limit = data_config.get(f"max_{split_name}_windows")
        is_truncated = any(
            entity["evaluated_num_points"] < entity["raw_num_points"]
            for entity in entities
        )
        truncation_reason = None
        if is_truncated:
            truncation_reason = (
                "max_window_cap"
                if configured_limit is not None
                else "window_stride_remainder"
            )
        if split_name == "test" and is_truncated:
            warnings.append(
                "Test split window coverage is truncated relative to the raw test "
                "timeline. Evaluation artifacts do not cover the full labeled "
                "timeline."
            )
            if truncation_reason == "max_window_cap":
                warnings.append(
                    "Configured "
                    f"max_test_windows={configured_limit} truncates the evaluated "
                    "test timeline. Treat this as a truncated smoke evaluation, "
                    "not a full-timeline test."
                )
            else:
                warnings.append(
                    "Current window size and stride leave an uncovered suffix on "
                    "the raw test timeline. Use benchmark-comparable coverage such "
                    "as test_stride=1 or another setting that still covers the full "
                    "labeled timeline."
                )
        split_reports[split_name] = {
            "num_sequences": len(split_sequences),
            "num_windows": len(data_bundle["datasets"][split_name]),
            "label_regime": label_summary["label_regime"],
            "n_pos": label_summary["n_pos"],
            "n_neg": label_summary["n_neg"],
            "positive_ratio": label_summary["positive_ratio"],
            "raw_num_points": coverage_summary["raw_num_points"],
            "evaluated_num_points": coverage_summary["evaluated_num_points"],
            "is_truncated": is_truncated,
            "truncation_reason": truncation_reason,
            "entities": entities,
        }

    report = {
        "dataset_name": data_bundle["dataset_name"],
        "data_config": data_config,
        "scaler_fit_scope": "train_only_before_windowing",
        "splits": split_reports,
        "warnings": warnings,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    if split_reports["test"]["label_regime"] in {"all_zero", "all_one"}:
        warnings.append(
            "Test labels contain only one class. Pointwise metrics such as ROC-AUC, "
            "PR-AUC, and VUS can become degenerate or misleading."
        )
    report["benchmark_comparability"] = (
        "benchmark_comparable"
        if (
            split_reports["test"]["label_regime"] == "mixed"
            and not split_reports["test"]["is_truncated"]
        )
        else "non_comparable"
    )
    report["protocol_status"] = (
        "benchmark_comparable_full_timeline"
        if report["benchmark_comparability"] == "benchmark_comparable"
        else (
            "truncated_smoke_evaluation"
            if split_reports["test"]["is_truncated"]
            else "single_class_test_labels"
        )
    )

    if evaluation_outputs is not None:
        metrics = evaluation_outputs["metrics"]
        report["evaluation"] = {
            "threshold": float(metrics.get("threshold", float("nan"))),
            "unique_label_count": int(metrics.get("unique_label_count", -1)),
            "is_single_class_label_regime": bool(
                int(metrics.get("is_single_class_label_regime", 0))
            ),
            "raw_num_points": int(metrics.get("raw_num_points", -1)),
            "evaluated_num_points": int(metrics.get("evaluated_num_points", -1)),
            "is_truncated_evaluation": bool(
                int(metrics.get("is_truncated_evaluation", 0))
            ),
            "score_min": float(metrics.get("score_min", float("nan"))),
            "score_max": float(metrics.get("score_max", float("nan"))),
            "score_mean": float(metrics.get("score_mean", float("nan"))),
            "score_std": float(metrics.get("score_std", float("nan"))),
        }
        if report["evaluation"]["is_single_class_label_regime"]:
            warnings.append(
                "Reconstructed evaluation labels still contain only one class after "
                "window aggregation."
            )

    return report


def render_dataset_protocol_audit_markdown(
    report: dict[str, Any],
    *,
    experiment_name: str,
) -> str:
    lines = [
        f"# Evaluation Protocol Audit: {experiment_name}",
        "",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Scaler fit scope: `{report['scaler_fit_scope']}`",
        f"- Benchmark comparability: `{report['benchmark_comparability']}`",
        f"- Protocol status: `{report['protocol_status']}`",
        "",
        "## Split Summary",
        "",
    ]
    split_reports = report.get("splits", {})
    for split_name in ("train", "val", "test"):
        split_report = split_reports.get(split_name)
        if split_report is None:
            lines.extend(
                [
                    f"### {split_name}",
                    "",
                    "- Unavailable in fallback audit mode.",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- Windows: {split_report['num_windows']}",
                f"- Label regime: `{split_report['label_regime']}`",
                f"- Positive ratio: {split_report['positive_ratio']:.6f}",
                f"- Evaluated points: {split_report['evaluated_num_points']}/{split_report['raw_num_points']}",
                f"- Truncated coverage: `{split_report['is_truncated']}`",
                f"- Truncation reason: `{split_report['truncation_reason']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Benchmark Protocol Status",
            "",
            "A benchmark-comparable run must evaluate a future test timeline that "
            "contains both normal and anomalous timesteps after reconstruction.",
            "",
            f"- Benchmark comparability: `{report['benchmark_comparability']}`",
            f"- Protocol status: `{report['protocol_status']}`",
            "",
        ]
    )

    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    evaluation = report.get("evaluation")
    test_split = split_reports.get("test")
    if evaluation is not None:
        lines.extend(
            [
                "## Evaluation Coverage",
                "",
                f"- Evaluated points: {evaluation['evaluated_num_points']}/{evaluation['raw_num_points']}",
                f"- Truncated evaluation artifact: `{evaluation['is_truncated_evaluation']}`",
                "",
            ]
        )
    if test_split is not None:
        implications = describe_metric_regime_implications(
            label_regime=str(test_split["label_regime"]),
            threshold=None if evaluation is None else float(evaluation["threshold"]),
        )
        lines.extend(
            [
                "## Metric Regime Interpretation",
                "",
                implications["summary"],
                "",
            ]
        )
        for implication in implications["implications"]:
            lines.append(f"- {implication}")
        lines.extend(["", f"- {implications['threshold_note']}", ""])

    return "\n".join(lines)


def build_protocol_audit_log_path(
    *,
    experiment_name: str,
    current_date: datetime | None = None,
) -> Path:
    resolved_date = current_date or datetime.now()
    date_folder = resolved_date.strftime("%m-%d-%Y")
    return (
        Path("documents")
        / "logs"
        / date_folder
        / "research"
        / f"{experiment_name}__evaluation_protocol_audit.md"
    )

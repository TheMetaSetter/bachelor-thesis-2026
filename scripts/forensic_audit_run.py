from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.evaluation_protocol_audit import (
    build_dataset_protocol_audit_report,
    describe_metric_regime_implications,
)
from src.core.config import load_experiment_config
from src.core.registry import build_dataset, register_dataset
from src.data.loaders import (
    build_anomaly_archive_dataset_bundle,
    build_smd_dataset_bundle,
)


def _load_observed_metrics(
    *,
    metrics_path: str | None,
    metrics_json: str | None,
) -> dict[str, Any] | None:
    if metrics_path is not None:
        return json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if metrics_json is not None:
        return json.loads(metrics_json)
    return None


def _build_forensic_markdown(
    *,
    experiment_config_path: str,
    experiment_config: dict[str, Any],
    report: dict[str, Any],
    observed_metrics: dict[str, Any] | None,
) -> str:
    test_split = report["splits"]["test"]
    implication_bundle = describe_metric_regime_implications(
        label_regime=str(test_split["label_regime"]),
        threshold=None
        if observed_metrics is None or "threshold" not in observed_metrics
        else float(observed_metrics["threshold"]),
        observed_metrics=observed_metrics,
    )
    observed_metrics_lines: list[str] = []
    if observed_metrics is None:
        observed_metrics_lines.append(
            "- No local `evaluation_metrics.json` was provided in this workspace, so metric values are inferred only when they follow directly from code and label regime."
        )
    else:
        for metric_name in (
            "precision",
            "recall",
            "pr_auc",
            "roc_auc",
            "threshold",
            "vus_pr",
            "vus_roc",
        ):
            if metric_name in observed_metrics:
                observed_metrics_lines.append(
                    f"- `{metric_name}`: {observed_metrics[metric_name]}"
                )

    warning_lines = [f"- {warning}" for warning in report["warnings"]]
    implication_lines = [f"- {line}" for line in implication_bundle["implications"]]
    conclusion_paragraph = (
        "The strongest repository-grounded conclusion is that this run uses a "
        "mixed-label future test timeline, so the label regime itself is not the "
        "main problem. The next forensic questions are evaluated coverage, "
        "threshold choice, and how scores were aggregated back onto the timeline."
    )
    if bool(test_split["is_truncated"]):
        conclusion_paragraph = (
            "The strongest repository-grounded conclusion is that this run is "
            "dominated by truncated early-prefix evaluation rather than by the "
            "full raw test timeline. In plain words, the evaluator only looked at "
            "an early slice of the test series, while the later suffix stayed "
            "outside evaluated coverage. That means weird metric bundles here must "
            "be interpreted first as protocol artifacts of partial coverage, not "
            "immediately as model quality evidence."
        )
    elif str(test_split["label_regime"]) == "all_zero":
        conclusion_paragraph = (
            "The strongest repository-grounded conclusion is that this run uses "
            "an all-normal single-class test vector. In that regime, anomaly "
            "metrics such as recall, PR-AUC, ROC-AUC, and VUS are degenerate or "
            "not scientifically informative in the usual pointwise sense."
        )
    elif str(test_split["label_regime"]) == "mixed":
        conclusion_paragraph = (
            "The strongest repository-grounded conclusion is that the label regime "
            "itself is not degenerate, so the next forensic question is whether "
            "evaluated coverage, threshold selection, and cross-entity aggregation "
            "still match the intended raw test protocol."
        )
    lines = [
        "---",
        f"date: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "researcher: Codex",
        f"topic: \"Forensic audit for {experiment_config['experiment_name']}\"",
        "status: complete",
        "---",
        "",
        f"# Forensic Audit: {experiment_config['experiment_name']}",
        "",
        "## Evaluated Run",
        "",
        f"- Experiment config: `{experiment_config_path}`",
        f"- Dataset: `{report['dataset_name']}`",
        f"- Data config path: `{experiment_config['data_config_path']}`",
        "",
        "## Verified Protocol Facts",
        "",
        f"- `benchmark_comparability`: `{report.get('benchmark_comparability', 'n/a')}`",
        f"- `protocol_status`: `{report.get('protocol_status', 'n/a')}`",
        f"- Test label regime: `{test_split['label_regime']}`",
        f"- Test positive ratio: `{test_split['positive_ratio']:.6f}`",
        f"- Test windows: `{test_split['num_windows']}`",
        f"- Test truncated coverage: `{test_split['is_truncated']}`",
        "",
        "## Observed Metric Bundle",
        "",
        *observed_metrics_lines,
        "",
        "## Causal Interpretation",
        "",
        implication_bundle["summary"],
        "",
        *implication_lines,
        f"- {implication_bundle['threshold_note']}",
        "",
        "## Raw Warnings From Audit Layer",
        "",
        *warning_lines,
        "",
        "## Conclusion",
        "",
        conclusion_paragraph,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    register_dataset("smd", build_smd_dataset_bundle)
    register_dataset("anomaly_archive", build_anomaly_archive_dataset_bundle)
    experiment_config = load_experiment_config(args.experiment_config)
    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"],
        experiment_config["data"],
    )
    observed_metrics = _load_observed_metrics(
        metrics_path=args.metrics_path,
        metrics_json=args.metrics_json,
    )
    report = build_dataset_protocol_audit_report(
        data_bundle=data_bundle,
        data_config=experiment_config["data"],
        evaluation_outputs=None
        if observed_metrics is None
        else {"metrics": observed_metrics},
    )
    markdown = _build_forensic_markdown(
        experiment_config_path=args.experiment_config,
        experiment_config=experiment_config,
        report=report,
        observed_metrics=observed_metrics,
    )
    if args.output_path is None:
        output_path = (
            Path("documents")
            / "logs"
            / datetime.now().strftime("%m-%d-%Y")
            / "research"
            / f"{experiment_config['experiment_name']}__forensic_audit.md"
        )
    else:
        output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

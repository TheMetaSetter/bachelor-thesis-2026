from __future__ import annotations

"""Summarize benchmark reports into one pedagogical table.

₍^. .^₎⟆ Reporting flow

benchmark report
  -> normalize fields
  -> keep method/variant/entity/seed aligned
  -> export JSON + CSV summary
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUMMARY_COLUMNS = [
    "method",
    "variant",
    "entity_id",
    "seed",
    "benchmark_type",
    "row_kind",
    "threshold_source",
    "point_rule",
    "smoothing_rule",
    "test_label_usage",
    "runtime_seconds",
    "metrics",
    "source_report_path",
]


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _report_kind(report: dict[str, Any], report_path: Path) -> str:
    if "online_execution" in report or "online_metrics_path" in report_path.name:
        return "online"
    return "offline"


def _threshold_artifact(report: dict[str, Any]) -> dict[str, Any]:
    if "thresholds" in report and isinstance(report["thresholds"], dict):
        return report
    online_execution = report.get("online_execution")
    if isinstance(online_execution, dict):
        artifact = online_execution.get("threshold_artifact")
        if isinstance(artifact, dict):
            return artifact
    return {}


def _thresholds(artifact: dict[str, Any]) -> dict[str, Any]:
    thresholds = artifact.get("thresholds")
    return thresholds if isinstance(thresholds, dict) else {}


def _extract_method_variant_entity_seed(
    report: dict[str, Any],
    artifact: dict[str, Any],
    report_kind: str,
) -> tuple[str, str, str, int]:
    benchmark_config = report.get("benchmark_config")
    benchmark_config = benchmark_config if isinstance(benchmark_config, dict) else {}
    online_execution = report.get("online_execution")
    online_execution = online_execution if isinstance(online_execution, dict) else {}

    method = (
        artifact.get("method_name")
        or online_execution.get("baseline_name")
        or benchmark_config.get("baseline_name")
        or report.get("baseline_name")
        or "unknown"
    )
    variant = (
        artifact.get("variant_name")
        or online_execution.get("online_variant")
        or benchmark_config.get("online_variant")
        or report.get("online_variant")
        or benchmark_config.get("offline_variant")
        or "unknown"
    )
    entity_id = (
        artifact.get("entity_id")
        or report.get("entity_id")
        or benchmark_config.get("entity_id")
        or "unknown"
    )
    seed = (
        artifact.get("seed") or report.get("seed") or benchmark_config.get("seed") or 0
    )
    if report_kind == "offline" and variant == "unknown":
        variant = method
    return str(method), str(variant), str(entity_id), int(seed)


def _extract_threshold_source(
    report: dict[str, Any],
    artifact: dict[str, Any],
    report_kind: str,
) -> str:
    online_execution = report.get("online_execution")
    online_execution = online_execution if isinstance(online_execution, dict) else {}
    if report_kind == "online":
        return str(
            online_execution.get(
                "threshold_source",
                _thresholds(artifact)
                .get("online_ewma_point", {})
                .get("source_split", "unknown"),
            )
        )
    thresholds = _thresholds(artifact)
    return str(
        thresholds.get("offline_point", {}).get(
            "source_split",
            report.get("threshold_source", "unknown"),
        )
    )


def _extract_point_rule(artifact: dict[str, Any]) -> str:
    thresholds = _thresholds(artifact)
    offline_rule = thresholds.get("offline_point", {}).get("score_rule")
    if offline_rule is not None:
        return str(offline_rule)
    online_rule = thresholds.get("online_ewma_point", {}).get("score_rule")
    if online_rule is not None:
        return str(online_rule)
    return "unknown"


def _extract_smoothing_rule(artifact: dict[str, Any]) -> str:
    thresholds = _thresholds(artifact)
    online_rule = thresholds.get("online_ewma_point", {}).get("score_rule")
    if online_rule is not None:
        return str(online_rule)
    return "none"


def _extract_test_label_usage(report: dict[str, Any], artifact: dict[str, Any]) -> str:
    protocol = report.get("protocol")
    protocol = protocol if isinstance(protocol, dict) else {}
    provenance = artifact.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return str(
        provenance.get(
            "test_label_usage",
            protocol.get("test_label_usage", "unknown"),
        )
    )


def _extract_runtime_seconds(report: dict[str, Any]) -> float | None:
    runtime_seconds = report.get("runtime_seconds")
    if runtime_seconds is not None:
        return _as_float(runtime_seconds)
    online_execution = report.get("online_execution")
    if (
        isinstance(online_execution, dict)
        and online_execution.get("runtime_seconds") is not None
    ):
        return _as_float(online_execution["runtime_seconds"])
    two_stage_execution = report.get("two_stage_execution")
    if (
        isinstance(two_stage_execution, dict)
        and two_stage_execution.get("runtime_seconds") is not None
    ):
        return _as_float(two_stage_execution["runtime_seconds"])
    return None


def _extract_row_kind(report: dict[str, Any], artifact: dict[str, Any]) -> str:
    benchmark_config = report.get("benchmark_config")
    benchmark_config = benchmark_config if isinstance(benchmark_config, dict) else {}
    explicit_row_kind = benchmark_config.get("row_kind") or report.get("row_kind")
    if isinstance(explicit_row_kind, str) and explicit_row_kind:
        return explicit_row_kind

    thresholds = _thresholds(artifact)
    threshold_tokens = " ".join(
        str(value)
        for value in (
            thresholds.get("offline_point", {}).get("score_rule"),
            thresholds.get("online_ewma_point", {}).get("score_rule"),
            report.get("threshold_source"),
            benchmark_config.get("baseline_name"),
            benchmark_config.get("online_variant"),
        )
        if value is not None
    ).lower()
    if "oracle" in threshold_tokens:
        return "oracle"
    if "self_join" in threshold_tokens or "self-join" in threshold_tokens:
        return "non_causal"

    benchmark_comparability = str(
        report.get("benchmark_comparability")
        or benchmark_config.get("benchmark_comparability")
        or report.get("protocol", {}).get("benchmark_comparability", "")
    ).lower()
    if benchmark_comparability == "non_comparable":
        return "non_causal"
    return "regular"


def _extract_metrics(report: dict[str, Any], report_kind: str) -> dict[str, Any]:
    if report_kind == "offline":
        if isinstance(report.get("offline_metrics"), dict):
            return dict(report["offline_metrics"])
        two_stage_execution = report.get("two_stage_execution")
        if isinstance(two_stage_execution, dict):
            return {
                "status": two_stage_execution.get("status"),
                "skip_completed": two_stage_execution.get("skip_completed"),
            }
        return {}

    online_execution = report.get("online_execution")
    if not isinstance(online_execution, dict):
        return {}
    metric_history = online_execution.get("metric_history") or []
    records = online_execution.get("records") or []
    num_updates = sum(1 for record in records if record.get("did_update"))
    metrics: dict[str, Any] = {
        "num_records": len(records),
        "num_metric_rows": len(metric_history),
        "num_updates": num_updates,
    }
    threshold_value = online_execution.get("threshold_value")
    if threshold_value is not None:
        metrics["threshold_value"] = float(threshold_value)
    return metrics


def summarize_report(report_path: Path) -> dict[str, Any]:
    report = _load_json(report_path)
    report_kind = _report_kind(report, report_path)
    artifact = _threshold_artifact(report)
    method, variant, entity_id, seed = _extract_method_variant_entity_seed(
        report, artifact, report_kind
    )
    row = {
        "method": method,
        "variant": variant,
        "entity_id": entity_id,
        "seed": seed,
        "benchmark_type": report_kind,
        "row_kind": _extract_row_kind(report, artifact),
        "threshold_source": _extract_threshold_source(report, artifact, report_kind),
        "point_rule": _extract_point_rule(artifact),
        "smoothing_rule": _extract_smoothing_rule(artifact),
        "test_label_usage": _extract_test_label_usage(report, artifact),
        "runtime_seconds": _extract_runtime_seconds(report),
        "metrics": _extract_metrics(report, report_kind),
        "source_report_path": str(report_path),
    }
    return row


def _iter_report_paths(
    report_paths: Iterable[Path] | None,
    report_dirs: Iterable[Path] | None,
) -> list[Path]:
    paths: list[Path] = []
    if report_paths is not None:
        paths.extend(report_paths)
    if report_dirs is not None:
        for report_dir in report_dirs:
            paths.extend(sorted(report_dir.rglob("*benchmark_report.json")))
    return sorted({path.resolve() for path in paths})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["metrics"] = json.dumps(csv_row["metrics"], sort_keys=True)
            writer.writerow({column: csv_row.get(column) for column in SUMMARY_COLUMNS})


def summarize_benchmark_results(
    *,
    report_paths: Iterable[Path] | None = None,
    report_dirs: Iterable[Path] | None = None,
    output_path: Path,
) -> dict[str, Any]:
    resolved_report_paths = _iter_report_paths(report_paths, report_dirs)
    rows = [summarize_report(report_path) for report_path in resolved_report_paths]
    summary = {
        "created_at_utc": _utc_now_iso(),
        "row_count": len(rows),
        "rows": rows,
        "source_report_paths": [str(path) for path in resolved_report_paths],
    }
    _write_json(output_path, summary)
    _write_csv(output_path.with_suffix(".csv"), rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Path to one benchmark report JSON file.",
    )
    parser.add_argument(
        "--report-dir",
        action="append",
        default=[],
        help="Directory that contains benchmark report JSON files.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/benchmark_summary.json",
        help="Where to write the summary JSON.",
    )
    args = parser.parse_args()
    summarize_benchmark_results(
        report_paths=[Path(path) for path in args.report],
        report_dirs=[Path(path) for path in args.report_dir],
        output_path=Path(args.output_path),
    )


if __name__ == "__main__":
    main()

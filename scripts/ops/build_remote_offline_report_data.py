from __future__ import annotations

"""Build compact offline report rows from a fragmented remote result tree.

The script is streamed to the remote host and prints JSON to stdout. It never
writes to the remote result tree. Identity is resolved from path/config/manifest
and checkpoint evidence before lower-trust threshold or UQ metadata.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_KEYS = ("vus_pr", "affiliation_f1", "vus_roc")
SPLIT_NAMES = ("clean_validation", "synthetic_validation", "test")
VARIANT_PATTERN = re.compile(r"__(O[01])__")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_values(payload: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        value = payload.get(key)
        values[key] = float(value) if isinstance(value, (int, float)) else None
    return values


def path_parts_after(path: Path, anchor: str) -> list[str]:
    parts = list(path.parts)
    return parts[parts.index(anchor) + 1 :]


def parse_basic_identity(metric_path: Path) -> dict[str, Any]:
    if "offline_benchmark" in metric_path.parts:
        tail = path_parts_after(metric_path, "offline_benchmark")
        return {
            "method_name": tail[0],
            "variant_name": None,
            "entity_id": tail[1],
            "seed": int(tail[2].removeprefix("seed")),
            "phase_name": "offline",
            "stage_name": "metric_summary",
        }
    if "redlamp_baseline" in metric_path.parts:
        tail = path_parts_after(metric_path, "redlamp_baseline")
        return {
            "method_name": "redlamp_baseline",
            "variant_name": None,
            "entity_id": tail[0],
            "seed": int(tail[1].removeprefix("seed")),
            "phase_name": "offline",
            "stage_name": "evaluation",
        }
    tail = path_parts_after(metric_path, "thesis")
    return {
        "method_name": "THESIS",
        "variant_name": tail[0],
        "entity_id": tail[1],
        "seed": int(tail[2].removeprefix("seed")),
        "phase_name": "offline",
        "stage_name": "stage_b_fusion_finetuning",
    }


def variant_from_experiment_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = VARIANT_PATTERN.search(value)
    return match.group(1) if match else None


def variant_from_config(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    experiment_variant = payload.get("experiment_variant")
    if experiment_variant == "two_stage_base_v1":
        return "O0"
    if experiment_variant == "two_stage_point_score_supervised_v1":
        return "O1"
    value = payload.get("variant_name")
    return value if isinstance(value, str) else None


def choose_variant(
    evidence: list[tuple[str, str | None]],
) -> tuple[str | None, list[str]]:
    usable = [(source, value) for source, value in evidence if value]
    if not usable:
        return None, ["no trusted variant evidence"]
    counts = Counter(value for _, value in usable)
    resolved = counts.most_common(1)[0][0]
    diagnostics = [f"{source}={value}" for source, value in usable if value != resolved]
    return resolved, diagnostics


def compact_split(split_payload: dict[str, Any] | None) -> dict[str, Any]:
    split_payload = split_payload or {}
    point_summary = split_payload.get("point_score_summary") or {}
    uncertainty = split_payload.get("uncertainty_summary") or {}
    return {
        "mean_of_means": point_summary.get("mean"),
        "mean_of_variances": uncertainty.get("point_anomaly_score_variance_mean"),
        "window_score_mean": (split_payload.get("window_score_summary") or {}).get(
            "mean"
        ),
        "window_variance_mean": uncertainty.get("window_anomaly_score_variance_mean"),
        "trace_audit": split_payload.get("trace_audit"),
    }


def thesis_artifact_paths(metric_path: Path) -> dict[str, Path]:
    stage_root = metric_path.parent
    run_root = stage_root.parent.parent
    return {
        "metric": metric_path,
        "uq": stage_root / "metrics" / "uq_summary.json",
        "config": stage_root / "resolved_experiment_config.json",
        "manifest": run_root / "two_stage" / "two_stage_manifest.json",
        "threshold": stage_root / "thresholds" / "thresholds.json",
        "protocol": stage_root / "protocol" / "resolved_protocol.json",
        "best": stage_root / "checkpoints" / "best.pt",
        "initialization": run_root
        / "two_stage"
        / "initializations"
        / "stage_b_init.pt",
    }


def resolve_thesis_identity(
    identity: dict[str, Any],
    config: dict[str, Any] | None,
    manifest: dict[str, Any],
    uq_run: dict[str, Any],
    threshold: dict[str, Any],
    uq_payload: dict[str, Any],
) -> tuple[str | None, dict[str, Any], list[str]]:
    path_variant = identity["variant_name"]
    trusted_evidence = [
        ("path", path_variant),
        ("resolved_config", variant_from_config(config)),
        (
            "manifest_experiment_name",
            variant_from_experiment_name(manifest.get("experiment_name")),
        ),
        (
            "uq_experiment_name",
            variant_from_experiment_name(uq_run.get("experiment_name")),
        ),
        (
            "checkpoint_path",
            variant_from_experiment_name(str(uq_run.get("checkpoint_path"))),
        ),
    ]
    resolved_variant, conflict_diagnostics = choose_variant(trusted_evidence)
    raw_metadata_variants = {
        "uq_summary": uq_run.get("variant_name"),
        "thresholds": threshold.get("variant_name"),
    }
    diagnostics = list(conflict_diagnostics)
    if any(
        value and value != resolved_variant for value in raw_metadata_variants.values()
    ):
        diagnostics.append("low_trust_variant_metadata_conflict")
    config_path_value = uq_run.get("experiment_config_path")
    if (
        isinstance(config_path_value, str)
        and config_path_value.count("outputs/benchmark") > 1
    ):
        diagnostics.append("duplicated_experiment_config_path_prefix")
    if (uq_payload.get("run_scalar_logs") or {}).get("query/num_samples_eval") is None:
        diagnostics.append("missing_query_num_samples_eval")
    return resolved_variant, raw_metadata_variants, diagnostics


def thesis_coverage(metric: dict[str, Any]) -> dict[str, Any]:
    raw_points = metric.get("raw_num_points")
    evaluated_points = metric.get("evaluated_num_points")
    missing_points = (
        int(raw_points - evaluated_points)
        if isinstance(raw_points, (int, float))
        and isinstance(evaluated_points, (int, float))
        else None
    )
    return {
        "raw_num_points": raw_points,
        "evaluated_num_points": evaluated_points,
        "missing_points": missing_points,
        "policy": "near_complete_tail_gap" if missing_points else "complete",
        "protocol_status_raw": metric.get("protocol_status"),
        "benchmark_comparability_raw": metric.get("benchmark_comparability"),
    }


def thesis_report_eligibility(uq_payload: dict[str, Any]) -> bool:
    split_payloads = uq_payload.get("splits") or {}
    for split_name in ("clean_validation", "test"):
        compact = compact_split(split_payloads.get(split_name))
        if compact["mean_of_means"] is None or compact["mean_of_variances"] is None:
            return False
    return True


def thesis_uq_summary(
    paths: dict[str, Path], uq_payload: dict[str, Any]
) -> dict[str, Any]:
    splits = uq_payload.get("splits") or {}
    return {
        "source_path": str(paths["uq"]),
        "splits": {split: compact_split(splits.get(split)) for split in SPLIT_NAMES},
    }


def validation_testing_comparison(uq_summary: dict[str, Any]) -> dict[str, Any]:
    validation = uq_summary["splits"]["clean_validation"]
    test = uq_summary["splits"]["test"]
    validation_mean = validation["mean_of_means"]
    test_mean = test["mean_of_means"]
    validation_variance = validation["mean_of_variances"]
    test_variance = test["mean_of_variances"]
    return {
        "mean_of_means_validation": validation_mean,
        "mean_of_means_test": test_mean,
        "mean_of_variances_validation": validation_variance,
        "mean_of_variances_test": test_variance,
        "test_minus_validation": {
            "mean_of_means": (
                test_mean - validation_mean
                if test_mean is not None and validation_mean is not None
                else None
            ),
            "mean_of_variances": (
                test_variance - validation_variance
                if test_variance is not None and validation_variance is not None
                else None
            ),
        },
    }


def thesis_provenance(
    metric_path: Path,
    paths: dict[str, Path],
    expected_sha256: Any,
    actual_sha256: str | None,
    diagnostics: list[str],
) -> dict[str, Any]:
    return {
        "metric_path": str(metric_path),
        "uq_path": str(paths["uq"]),
        "protocol_path": str(paths["protocol"]),
        "best_checkpoint_path": str(paths["best"]),
        "initialization_checkpoint_path": str(paths["initialization"]),
        "best_checkpoint_sha256_expected": expected_sha256,
        "best_checkpoint_sha256_actual": actual_sha256,
        "best_checkpoint_sha256_matches": expected_sha256 == actual_sha256,
        "resolved_config_path": str(paths["config"]),
        "manifest_path": str(paths["manifest"]),
        "threshold_path": str(paths["threshold"]),
        "diagnostics": diagnostics,
    }


def build_thesis_record(metric_path: Path) -> dict[str, Any]:
    paths = thesis_artifact_paths(metric_path)
    identity = parse_basic_identity(metric_path)
    uq_payload = load_json(paths["uq"]) or {}
    uq_run = uq_payload.get("run") or {}
    config = load_json(paths["config"])
    manifest = load_json(paths["manifest"]) or {}
    threshold = load_json(paths["threshold"]) or {}
    metric = load_json(paths["metric"]) or {}
    resolved_variant, raw_metadata_variants, diagnostics = resolve_thesis_identity(
        identity,
        config,
        manifest,
        uq_run,
        threshold,
        uq_payload,
    )
    actual_sha256 = sha256_file(paths["best"])
    expected_sha256 = uq_run.get("checkpoint_sha256")
    coverage = thesis_coverage(metric)
    metric_values_by_name = metric_values(metric)
    path_variant = identity["variant_name"]
    uq_summary = thesis_uq_summary(paths, uq_payload)
    return {
        "run_id": f"thesis/{identity['variant_name']}/{identity['entity_id']}/seed{identity['seed']}",
        "identity": {
            **identity,
            "variant_name": resolved_variant,
            "raw_path_variant": path_variant,
            "raw_metadata_variants": raw_metadata_variants,
            "resolution": "path_config_manifest_checkpoint_consensus",
            "identity_conflict": bool(diagnostics),
        },
        "metrics": metric_values_by_name,
        "coverage": coverage,
        "uq_summary": uq_summary,
        "validation_testing_comparison": validation_testing_comparison(uq_summary),
        "provenance": thesis_provenance(
            metric_path, paths, expected_sha256, actual_sha256, diagnostics
        ),
        "table_1_eligible": all(
            value is not None for value in metric_values_by_name.values()
        ),
        "table_2_eligible": thesis_report_eligibility(uq_payload),
    }


def build_simple_record(metric_path: Path) -> dict[str, Any]:
    identity = parse_basic_identity(metric_path)
    metric = load_json(metric_path) or {}
    coverage = {
        "raw_num_points": metric.get("raw_num_points"),
        "evaluated_num_points": metric.get("evaluated_num_points"),
        "protocol_status_raw": metric.get("protocol_status"),
        "benchmark_comparability_raw": metric.get("benchmark_comparability"),
    }
    if metric_path.name == "offline_metrics.json":
        source_kind = "traditional_machine_learning_metric"
        table_2_eligible = False
    else:
        source_kind = "redlamp_baseline_metric"
        table_2_eligible = False
    return {
        "run_id": f"{identity['method_name']}/{identity['entity_id']}/seed{identity['seed']}",
        "identity": {**identity, "resolution": "path", "identity_conflict": False},
        "metrics": metric_values(metric),
        "coverage": coverage,
        "uq_summary": {
            "status": "not_applicable"
            if source_kind.startswith("traditional")
            else "not_available"
        },
        "provenance": {"metric_path": str(metric_path), "source_kind": source_kind},
        "table_1_eligible": all(
            value is not None for value in metric_values(metric).values()
        ),
        "table_2_eligible": table_2_eligible,
    }


def collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metric_path in sorted(root.rglob("offline_metrics.json")):
        if "offline_benchmark" in metric_path.parts:
            records.append(build_simple_record(metric_path))
    for metric_path in sorted(root.rglob("evaluation_metrics.json")):
        if "stage_b_fusion_finetuning" in metric_path.parts:
            records.append(build_thesis_record(metric_path))
        elif "redlamp_baseline" in metric_path.parts:
            records.append(build_simple_record(metric_path))
    return records


def build_payload(root: Path) -> dict[str, Any]:
    records = collect_records(root)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(root),
        "resolution_manifest": "outputs/reporting/offline_phase_tables/identity_reconciliation.json",
        "records": records,
        "summary": {
            "record_count": len(records),
            "table_1_eligible_count": sum(
                record["table_1_eligible"] for record in records
            ),
            "table_2_eligible_count": sum(
                record["table_2_eligible"] for record in records
            ),
            "identity_conflict_count": sum(
                record["identity"].get("identity_conflict", False) for record in records
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_payload(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

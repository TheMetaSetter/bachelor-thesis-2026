from __future__ import annotations

"""Collect thesis offline benchmark data into one compact report payload.

This script keeps row-level raw data so the final paper tables can be rebuilt
with different layouts without re-running evaluation.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCAN_ROOTS = (
    "outputs/eval18",
    "outputs/benchmark/smd/offline_benchmark",
    "outputs/benchmark/smd/redlamp_baseline",
)

DEFAULT_OUTPUT_JSON = Path("outputs/reporting/offline_phase_tables/offline_report_data.json")
DEFAULT_OUTPUT_MD = Path("outputs/reporting/offline_phase_tables/offline_report_data.md")

METRIC_KEYS = ("vus_pr", "affiliation_f1", "vus_roc")
UQ_SPLIT_NAMES = ("clean_validation", "synthetic_validation", "test")


@dataclass(frozen=True)
class RunIdentity:
    method_group: str
    display_group: str
    variant_name: str | None
    entity_id: str | None
    seed: int | None
    source_kind: str


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return _load_json(path)


def _mean(values: list[float | int | None]) -> float | None:
    numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric_values:
        return None
    return float(sum(numeric_values) / len(numeric_values))


def _derive_thesis_identity(run_root: Path) -> RunIdentity | None:
    parts = run_root.parts
    try:
        eval18_index = parts.index("eval18")
    except ValueError:
        return None
    if eval18_index + 1 >= len(parts):
        return None
    alias = parts[eval18_index + 1]
    match = re.fullmatch(r"(o[01])_(m\d+_\d+)_(s\d+)", alias)
    if match is None:
        return None
    variant_token, entity_token, seed_token = match.groups()
    variant_name = variant_token.upper()
    entity_id = entity_token.replace("m", "machine_", 1).replace("_", "_", 1)
    if entity_id.startswith("machine__"):
        entity_id = entity_id.replace("machine__", "machine_", 1)
    seed = int(seed_token.removeprefix("s"))
    display_group = f"Thesis main + {variant_name}"
    return RunIdentity(
        method_group="thesis_main",
        display_group=display_group,
        variant_name=variant_name,
        entity_id=entity_id,
        seed=seed,
        source_kind="thesis_eval18",
    )


def _derive_offline_baseline_identity(run_root: Path) -> RunIdentity | None:
    parts = run_root.parts
    if "offline_benchmark" not in parts:
        return None
    marker = parts.index("offline_benchmark")
    tail = parts[marker + 1 :]
    if len(tail) < 3:
        return None
    method_group = tail[0]
    entity_id = tail[1]
    seed_part = tail[2]
    if not seed_part.startswith("seed"):
        return None
    try:
        seed = int(seed_part.removeprefix("seed"))
    except ValueError:
        return None
    display_group = {
        "redlamp_baseline": "RedLamp + baseline",
        "kmeans_ad": "Traditional ML 1",
        "iforest": "Traditional ML 2",
        "stumpy_channel_ab": "Traditional ML 3",
    }.get(method_group, method_group)
    return RunIdentity(
        method_group=method_group,
        display_group=display_group,
        variant_name=None,
        entity_id=entity_id,
        seed=seed,
        source_kind="offline_benchmark",
    )


def _derive_identity(run_root: Path) -> RunIdentity | None:
    return _derive_thesis_identity(run_root) or _derive_offline_baseline_identity(
        run_root
    )


def _resolve_run_root_from_metric_path(metric_path: Path) -> Path:
    if "two_stage" in metric_path.parts:
        two_stage_index = metric_path.parts.index("two_stage")
        return Path(*metric_path.parts[:two_stage_index])
    return metric_path.parent


def _candidate_uq_paths(run_root: Path, metric_path: Path) -> list[Path]:
    return [
        run_root / "metrics" / "uq_summary.json",
        run_root / "two_stage" / "stage_b_fusion_finetuning" / "metrics" / "uq_summary.json",
        metric_path.parent / "metrics" / "uq_summary.json",
        metric_path.parent / "uq_summary.json",
    ]


def _normalize_metrics(metric_payload: Any) -> dict[str, float | None]:
    if not isinstance(metric_payload, dict):
        return {metric_key: None for metric_key in METRIC_KEYS}
    normalized = {}
    for metric_key in METRIC_KEYS:
        value = metric_payload.get(metric_key)
        if value is None and metric_key == "vus_pr":
            value = metric_payload.get("pr_auc")
        if value is None and metric_key == "affiliation_f1":
            value = metric_payload.get("f1")
        normalized[metric_key] = (
            float(value) if isinstance(value, (int, float)) else None
        )
    return normalized


def _collect_uq_payload(uq_payload: Any) -> dict[str, Any] | None:
    if not isinstance(uq_payload, dict):
        return None
    splits = uq_payload.get("splits")
    if not isinstance(splits, dict):
        return None
    compact_splits: dict[str, Any] = {}
    for split_name in UQ_SPLIT_NAMES:
        split_payload = splits.get(split_name)
        if not isinstance(split_payload, dict):
            continue
        compact_splits[split_name] = {
            "point_score_summary": split_payload.get("point_score_summary"),
            "window_score_summary": split_payload.get("window_score_summary"),
            "trace_audit": split_payload.get("trace_audit"),
            "uncertainty_summary": split_payload.get("uncertainty_summary"),
        }
    return {
        "run": uq_payload.get("run"),
        "run_scalar_logs": uq_payload.get("run_scalar_logs"),
        "splits": compact_splits,
    }


def _collect_run_record(metric_path: Path) -> dict[str, Any] | None:
    run_root = _resolve_run_root_from_metric_path(metric_path)
    identity = _derive_identity(run_root)
    if identity is None:
        return None
    metric_payload = _maybe_load_json(metric_path)
    if metric_payload is None:
        return None
    uq_payload = None
    uq_path = None
    for candidate in _candidate_uq_paths(run_root, metric_path):
        if candidate.exists():
            uq_path = candidate
            uq_payload = _maybe_load_json(candidate)
            break
    return {
        "run_root": str(run_root),
        "metric_path": str(metric_path),
        "uq_path": None if uq_path is None else str(uq_path),
        "identity": {
            "method_group": identity.method_group,
            "display_group": identity.display_group,
            "variant_name": identity.variant_name,
            "entity_id": identity.entity_id,
            "seed": identity.seed,
            "source_kind": identity.source_kind,
        },
        "metrics": _normalize_metrics(metric_payload),
        "uq_summary": _collect_uq_payload(uq_payload),
    }


def _iter_metric_paths(scan_root: Path) -> list[Path]:
    metric_paths: list[Path] = []
    if not scan_root.exists():
        return metric_paths
    for path in scan_root.rglob("evaluation_metrics.json"):
        if path.is_file():
            metric_paths.append(path)
    return sorted({path.resolve() for path in metric_paths})


def _aggregate_metric_rows(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        identity = record["identity"]
        grouped[
            (
                str(identity["method_group"]),
                identity.get("variant_name"),
                identity.get("entity_id"),
            )
        ].append(record)

    rows: list[dict[str, Any]] = []
    for (method_group, variant_name, entity_id), records in sorted(grouped.items()):
        seeds = sorted(
            [record["identity"]["seed"] for record in records if record["identity"]["seed"] is not None]
        )
        rows.append(
            {
                "method_group": method_group,
                "display_group": records[0]["identity"]["display_group"],
                "variant_name": variant_name,
                "entity_id": entity_id,
                "seed_count": len(records),
                "seeds": seeds,
                "metrics_mean": {
                    metric_key: _mean(
                        [record["metrics"].get(metric_key) for record in records]
                    )
                    for metric_key in METRIC_KEYS
                },
            }
        )
    return rows


def _aggregate_uq_rows(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None, str | None], list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        identity = record["identity"]
        if identity["method_group"] != "thesis_main":
            continue
        if record.get("uq_summary") is None:
            continue
        grouped[
            (
                str(identity["method_group"]),
                identity.get("variant_name"),
                identity.get("entity_id"),
            )
        ].append(record)

    rows: list[dict[str, Any]] = []
    for (method_group, variant_name, entity_id), records in sorted(grouped.items()):
        split_names = sorted(
            {
                split_name
                for record in records
                for split_name in record["uq_summary"]["splits"].keys()
            }
        )
        split_means: dict[str, Any] = {}
        for split_name in split_names:
            split_payloads = [
                record["uq_summary"]["splits"].get(split_name, {})
                for record in records
            ]
            uncertainty_keys = sorted(
                {
                    key
                    for split_payload in split_payloads
                    for key in (split_payload.get("uncertainty_summary") or {}).keys()
                }
            )
            split_means[split_name] = {
                "trace_audit": {
                    "any_uncertainty_history": any(
                        bool((split_payload.get("trace_audit") or {}).get("any_uncertainty_history"))
                        for split_payload in split_payloads
                    ),
                    "any_mc_sample_history": any(
                        bool((split_payload.get("trace_audit") or {}).get("any_mc_sample_history"))
                        for split_payload in split_payloads
                    ),
                },
                "uncertainty_summary_mean": {
                    key: _mean(
                        [
                            (
                                split_payload.get("uncertainty_summary") or {}
                            ).get(key)
                            for split_payload in split_payloads
                        ]
                    )
                    for key in uncertainty_keys
                },
            }
        rows.append(
            {
                "method_group": method_group,
                "display_group": records[0]["identity"]["display_group"],
                "variant_name": variant_name,
                "entity_id": entity_id,
                "seed_count": len(records),
                "seeds": sorted(
                    [
                        record["identity"]["seed"]
                        for record in records
                        if record["identity"]["seed"] is not None
                    ]
                ),
                "split_means": split_means,
            }
        )
    return rows


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Offline report collection data")
    lines.append("")
    lines.append(f"Created at UTC: `{payload['created_at_utc']}`")
    lines.append("")
    lines.append("## Run counts")
    lines.append("")
    lines.append(f"- Raw rows: `{payload['raw_run_count']}`")
    lines.append(f"- Metric table groups: `{len(payload['metric_table_rows'])}`")
    lines.append(f"- UQ table groups: `{len(payload['uq_table_rows'])}`")
    lines.append("")
    lines.append("## Source roots")
    lines.append("")
    for root in payload["scan_roots"]:
        lines.append(f"- `{root}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- JSON: `{payload['output_json']}`")
    lines.append("")
    return "\n".join(lines)


def build_report_data(scan_roots: list[Path]) -> dict[str, Any]:
    metric_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for scan_root in scan_roots:
        for metric_path in _iter_metric_paths(scan_root):
            if metric_path in seen_paths:
                continue
            seen_paths.add(metric_path)
            metric_paths.append(metric_path)
    run_records = []
    skipped: list[str] = []
    for metric_path in sorted(metric_paths):
        record = _collect_run_record(metric_path)
        if record is None:
            skipped.append(str(metric_path))
            continue
        run_records.append(record)
    payload = {
        "created_at_utc": _utc_now_iso(),
        "scan_roots": [str(path) for path in scan_roots],
        "raw_run_count": len(run_records),
        "raw_runs": run_records,
        "metric_table_rows": _aggregate_metric_rows(run_records),
        "uq_table_rows": _aggregate_uq_rows(run_records),
        "skipped_metric_paths": skipped,
    }
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect offline thesis benchmark metrics and UQ summaries into one JSON bundle."
        )
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="Root directory to scan recursively for evaluation_metrics.json.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to write the consolidated JSON bundle.",
    )
    parser.add_argument(
        "--output-md",
        default=str(DEFAULT_OUTPUT_MD),
        help="Path to write a small human-readable index markdown.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    scan_roots = [
        (repo_root / Path(root)).resolve()
        for root in (args.scan_roots or DEFAULT_SCAN_ROOTS)
    ]
    payload = build_report_data(scan_roots)
    output_json = (repo_root / Path(args.output_json)).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload["output_json"] = str(output_json)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md = (repo_root / Path(args.output_md)).resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "raw_run_count": payload["raw_run_count"],
                "metric_table_groups": len(payload["metric_table_rows"]),
                "uq_table_groups": len(payload["uq_table_rows"]),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

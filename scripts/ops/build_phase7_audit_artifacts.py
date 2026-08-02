from __future__ import annotations

"""Build the small local audit artifacts required by Phase 7.

This script reads the already-created local report bundle and prune manifest.
It does not access or modify the remote result tree.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DIRECTORY = Path("outputs/reporting/offline_phase_tables")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def coverage_status(row: dict[str, Any]) -> tuple[str, int | None]:
    coverage = row.get("coverage") or {}
    raw = coverage.get("raw_num_points")
    evaluated = coverage.get("evaluated_num_points")
    if not isinstance(raw, (int, float)) or not isinstance(evaluated, (int, float)):
        return "not_reported", None
    missing = int(raw - evaluated)
    return ("near_complete_tail_gap" if missing else "complete"), missing


def table_status(row: dict[str, Any], table_number: int) -> str:
    if row.get(f"table_{table_number}_eligible"):
        return "ready"
    if table_number == 2 and row["identity"]["method_name"] != "THESIS":
        uq_status = (row.get("uq_summary") or {}).get("status")
        return "not_applicable" if uq_status == "not_applicable" else "not_available"
    return "blocked"


def record_has_diagnostics(row: dict[str, Any]) -> bool:
    identity = row["identity"]
    coverage = row.get("coverage") or {}
    coverage_state, _ = coverage_status(row)
    return bool(
        identity.get("identity_conflict")
        or coverage_state == "near_complete_tail_gap"
        or coverage.get("protocol_status_raw") is not None
        or coverage.get("benchmark_comparability_raw") is not None
        or (row.get("provenance") or {}).get("diagnostics")
    )


def method_status(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    table_1_ready = all(row.get("table_1_eligible") for row in rows)
    table_2_expected = method == "THESIS"
    table_2_ready = all(row.get("table_2_eligible") for row in rows)
    conflicts = sum(bool(row["identity"].get("identity_conflict")) for row in rows)
    tail_gaps = []
    for row in rows:
        status, missing = coverage_status(row)
        if status == "near_complete_tail_gap":
            tail_gaps.append({"run_id": row["run_id"], "missing_points": missing})
    if table_2_expected:
        status = (
            "blocked"
            if not (table_1_ready and table_2_ready)
            else "ready_with_diagnostics"
            if any(record_has_diagnostics(row) for row in rows)
            else "ready"
        )
    else:
        status = (
            "blocked"
            if not table_1_ready
            else "ready_with_diagnostics"
            if any(record_has_diagnostics(row) for row in rows)
            else "ready"
        )
    return {
        "method": method,
        "run_count": len(rows),
        "status": status,
        "table_1_eligible_count": sum(
            row.get("table_1_eligible", False) for row in rows
        ),
        "table_2_expected": table_2_expected,
        "table_2_eligible_count": sum(
            row.get("table_2_eligible", False) for row in rows
        ),
        "identity_conflict_count": conflicts,
        "tail_gap_run_count": len(tail_gaps),
        "tail_gaps": tail_gaps,
    }


def canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    identity = row["identity"]
    provenance = row.get("provenance") or {}
    coverage, missing = coverage_status(row)
    report_ready = row.get("table_1_eligible") and (
        row.get("table_2_eligible") or identity["method_name"] != "THESIS"
    )
    return {
        "run_id": row["run_id"],
        "identity": identity,
        "status": {
            "overall": (
                "blocked"
                if not report_ready
                else "ready_with_diagnostics"
                if record_has_diagnostics(row)
                else "ready"
            ),
            "table_1": table_status(row, 1),
            "table_2": table_status(row, 2),
            "coverage": coverage,
        },
        "report_fields": {
            "metrics": row.get("metrics"),
            "uq_summary": row.get("uq_summary"),
            "validation_testing_comparison": row.get("validation_testing_comparison"),
        },
        "artifact_paths": provenance,
        "checkpoint_evidence": {
            "best_checkpoint_sha256_matches": provenance.get(
                "best_checkpoint_sha256_matches"
            ),
            "best_checkpoint_sha256_expected": provenance.get(
                "best_checkpoint_sha256_expected"
            ),
            "best_checkpoint_sha256_actual": provenance.get(
                "best_checkpoint_sha256_actual"
            ),
        },
        "diagnostics": {
            "missing_points": missing,
            "raw_protocol_status": (row.get("coverage") or {}).get(
                "protocol_status_raw"
            ),
            "raw_benchmark_comparability": (row.get("coverage") or {}).get(
                "benchmark_comparability_raw"
            ),
            "provenance_diagnostics": provenance.get("diagnostics", []),
        },
    }


def build_canonical_manifest(
    report: dict[str, Any],
    reconciliation_path: Path,
) -> dict[str, Any]:
    rows = report["records"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["identity"]["method_name"], []).append(row)
    method_reports = [
        method_status(method, grouped[method]) for method in sorted(grouped)
    ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Canonical local index for the two offline report tables and later audit. "
            "It records the chosen identity, report fields, provenance and diagnostics."
        ),
        "source_root": report["source_root"],
        "source_report": str(
            Path(
                "outputs/reporting/offline_phase_tables/offline_report_data.json"
            ).resolve()
        ),
        "identity_reconciliation": str(reconciliation_path.resolve()),
        "summary": {
            "record_count": len(rows),
            "method_count": len(grouped),
            "table_1_eligible_count": sum(
                row.get("table_1_eligible", False) for row in rows
            ),
            "table_2_eligible_count": sum(
                row.get("table_2_eligible", False) for row in rows
            ),
            "blocked_record_count": sum(
                canonical_record(row)["status"]["overall"] == "blocked" for row in rows
            ),
            "identity_conflict_count": sum(
                bool(row["identity"].get("identity_conflict")) for row in rows
            ),
        },
        "method_status": method_reports,
        "records": [canonical_record(row) for row in rows],
    }


def build_coverage_gap_report(
    report: dict[str, Any],
    prune_manifest: dict[str, Any],
    canonical_path: Path,
) -> dict[str, Any]:
    rows = report["records"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["identity"]["method_name"], []).append(row)
    method_reports = [
        method_status(method, grouped[method]) for method in sorted(grouped)
    ]
    tail_gap_rows = [
        {"run_id": row["run_id"], "missing_points": coverage_status(row)[1]}
        for row in rows
        if coverage_status(row)[0] == "near_complete_tail_gap"
    ]
    raw_status_counts = Counter(
        (row.get("coverage") or {}).get("protocol_status_raw")
        for row in rows
        if (row.get("coverage") or {}).get("protocol_status_raw") is not None
    )
    raw_comparability_counts = Counter(
        (row.get("coverage") or {}).get("benchmark_comparability_raw")
        for row in rows
        if (row.get("coverage") or {}).get("benchmark_comparability_raw") is not None
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (
            "Make accepted coverage gaps, raw protocol labels and identity diagnostics "
            "visible without reopening deleted raw traces."
        ),
        "source_report": str(
            Path(
                "outputs/reporting/offline_phase_tables/offline_report_data.json"
            ).resolve()
        ),
        "canonical_manifest": str(canonical_path.resolve()),
        "summary": {
            "record_count": len(rows),
            "table_1_eligible_count": sum(
                row.get("table_1_eligible", False) for row in rows
            ),
            "table_2_eligible_count": sum(
                row.get("table_2_eligible", False) for row in rows
            ),
            "table_1_complete": all(row.get("table_1_eligible", False) for row in rows),
            "table_2_complete_for_expected_methods": all(
                row.get("table_2_eligible", False)
                for row in rows
                if row["identity"]["method_name"] == "THESIS"
            ),
            "blocked_methods": [
                item["method"] for item in method_reports if item["status"] == "blocked"
            ],
            "identity_conflict_count": sum(
                bool(row["identity"].get("identity_conflict")) for row in rows
            ),
            "tail_gap_run_count": len(tail_gap_rows),
            "raw_protocol_status_counts": dict(sorted(raw_status_counts.items())),
            "raw_benchmark_comparability_counts": dict(
                sorted(raw_comparability_counts.items())
            ),
            "prune_delete_count": prune_manifest["summary"]["action_counts"].get(
                "delete", 0
            ),
            "prune_delete_bytes": prune_manifest["summary"]["delete_bytes"],
        },
        "method_status": method_reports,
        "accepted_gaps": {
            "policy": "near_complete_tail_gap",
            "runs": tail_gap_rows,
            "explanation": (
                "Một số run thiếu một phần đuôi rất nhỏ do chưa có end-aligned window; "
                "giữ nguyên diagnostic gốc và vẫn dùng report theo chính sách đã duyệt."
            ),
        },
        "identity_diagnostics": {
            "count": sum(
                bool(row["identity"].get("identity_conflict")) for row in rows
            ),
            "source": "canonical records preserve raw metadata and reconciliation diagnostics",
        },
        "limitations": [
            "offline_benchmark có metric chính nhưng không có UQ summary; UQ không áp dụng cho nhóm này.",
            "redlamp_baseline có coverage gần đầy đủ nhưng không có UQ summary.",
            "THESIS giữ nguyên nhãn protocol_status=truncated_smoke_evaluation và benchmark_comparability=non_comparable trong diagnostics; không đổi artifact gốc thành full-timeline comparable.",
            "Chín record THESIS có conflict metadata variant ở trường độ tin cậy thấp; variant canonical được resolve bằng path/config/manifest/checkpoint và raw conflict vẫn được giữ.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    directory = args.directory
    report_path = directory / "offline_report_data.json"
    reconciliation_path = directory / "identity_reconciliation.json"
    prune_path = directory / "prune_manifest.json"
    canonical_path = directory / "canonical_run_manifest.json"
    coverage_path = directory / "coverage_gap_report.json"
    report = load_json(report_path)
    reconciliation = load_json(reconciliation_path)
    prune_manifest = load_json(prune_path)
    canonical = build_canonical_manifest(report, reconciliation_path)
    coverage = build_coverage_gap_report(report, prune_manifest, canonical_path)
    canonical["resolution_policy"] = reconciliation.get("resolution_strategy", {})
    write_json(canonical_path, canonical)
    write_json(coverage_path, coverage)
    print(
        json.dumps(
            {
                "canonical": str(canonical_path),
                "coverage": str(coverage_path),
                "summary": coverage["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

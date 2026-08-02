from __future__ import annotations

"""Build a local prune manifest from a read-only remote inventory."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REMOTE_ROOT = "/root/bachelor-thesis-2026"
REPORT_PATH = Path("outputs/reporting/offline_phase_tables/offline_report_data.json")
DEFAULT_OUTPUT = Path("outputs/reporting/offline_phase_tables/prune_manifest.json")
RAW_TRACE_NAMES = {
    "clean_validation_traces.json",
    "synthetic_validation_traces.json",
    "test_traces.json",
}
REQUIRED_NAMES = {
    "evaluation_metrics.json",
    "offline_metrics.json",
    "uq_summary.json",
    "best.pt",
    "stage_b_init.pt",
    "resolved_protocol.json",
    "thresholds.json",
    "retention_bundle_manifest.json",
    "retention_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def read_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        size_text, path = line.split("\t", 1)
        rows.append({"path": path, "size_bytes": int(size_text)})
    return rows


def path_after(path: str, anchor: str) -> list[str]:
    parts = Path(path).parts
    return list(parts[parts.index(anchor) + 1 :])


def identity_for(path: str) -> dict[str, Any]:
    if "offline_benchmark" in Path(path).parts:
        tail = path_after(path, "offline_benchmark")
        return {
            "run_id": f"{tail[0]}/{tail[1]}/{tail[2]}",
            "method": tail[0],
            "variant": None,
            "entity": tail[1],
            "seed": tail[2],
            "phase": "offline",
            "stage": "metric_summary",
        }
    if "redlamp_baseline" in Path(path).parts:
        tail = path_after(path, "redlamp_baseline")
        return {
            "run_id": f"redlamp_baseline/{tail[0]}/{tail[1]}",
            "method": "redlamp_baseline",
            "variant": None,
            "entity": tail[0],
            "seed": tail[1],
            "phase": "offline",
            "stage": "evaluation",
        }
    tail = path_after(path, "thesis")
    stage = "stage_b_fusion_finetuning"
    if "stage_a_multitask_pretraining" in path:
        stage = "stage_a_multitask_pretraining"
    return {
        "run_id": f"thesis/{tail[0]}/{tail[1]}/{tail[2]}",
        "method": "THESIS",
        "variant": tail[0],
        "entity": tail[1],
        "seed": tail[2],
        "phase": "offline",
        "stage": stage,
    }


def report_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["run_id"]: row for row in report.get("records", [])}


def protected_role(path: str) -> str | None:
    name = Path(path).name
    if name == "evaluation_metrics.json":
        return "table_1_metric_summary"
    if name == "offline_metrics.json":
        return "fallback_metric_summary"
    if name == "uq_summary.json":
        return "table_2_uq_summary"
    if name == "best.pt":
        return "stage_best_checkpoint"
    if name == "stage_b_init.pt":
        return "stage_initialization_checkpoint"
    if name == "resolved_protocol.json":
        return "resolved_protocol"
    if name == "thresholds.json":
        return "thresholds"
    if name == "retention_bundle_manifest.json":
        return "retention_manifest"
    if name == "retention_summary.json":
        return "retention_summary"
    if name.endswith("_point_scores.npz"):
        return "compact_point_scores"
    return None


def is_raw_trace(path: str) -> bool:
    return Path(path).name in RAW_TRACE_NAMES


def action_for(
    path: str, identity: dict[str, Any], row: dict[str, Any] | None
) -> tuple[str, str, str]:
    name = Path(path).name
    method = identity["method"]
    if is_raw_trace(path):
        if row is None:
            return "review", "raw trace has no canonical report row", "raw_trace"
        if not row.get("table_1_eligible"):
            return "review", "table 1 summary is not eligible", "raw_trace"
        if method != "offline_benchmark" and not row.get("table_2_eligible"):
            return (
                "review",
                "required UQ summary is unavailable or invalid",
                "raw_trace",
            )
        row_identity = row["identity"]
        conflict_is_resolved = bool(
            row_identity.get("variant_name")
            and row_identity.get("resolution")
            and row_identity.get("resolution") != "path"
        )
        if row_identity.get("identity_conflict") and not conflict_is_resolved:
            return "review", "identity conflict is not resolved", "raw_trace"
        return (
            "delete",
            "report bundle contains required summaries; raw trace is not needed for current tables",
            "raw_trace",
        )
    role = protected_role(path)
    if role:
        if name == "offline_metrics.json":
            return (
                "review",
                "retain until fallback metric source discrepancy is formally closed",
                role,
            )
        return "keep", f"required or selected audit artifact: {role}", role
    return (
        "review",
        "artifact was inventoried but has no Phase 5 deletion rule",
        "unclassified",
    )


def build_entries(
    inventory: list[dict[str, Any]], report: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = report_index(report)
    entries: list[dict[str, Any]] = []
    for item in inventory:
        identity = identity_for(item["path"])
        row = rows.get(identity["run_id"])
        action, reason, role = action_for(item["path"], identity, row)
        entries.append(
            {
                "absolute_path": item["path"],
                "run_id": identity["run_id"],
                "method": identity["method"],
                "variant": identity["variant"],
                "entity": identity["entity"],
                "seed": identity["seed"],
                "phase": identity["phase"],
                "stage": identity["stage"],
                "artifact_role": role,
                "size_bytes": item["size_bytes"],
                "action": action,
                "reason": reason,
            }
        )
    return sorted(entries, key=lambda entry: entry["absolute_path"])


def summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_action = Counter(entry["action"] for entry in entries)
    by_role = Counter(entry["artifact_role"] for entry in entries)
    delete_bytes = sum(
        entry["size_bytes"] for entry in entries if entry["action"] == "delete"
    )
    return {
        "entry_count": len(entries),
        "action_counts": dict(sorted(by_action.items())),
        "role_counts": dict(sorted(by_role.items())),
        "delete_bytes": delete_bytes,
        "delete_gib": round(delete_bytes / (1024**3), 3),
        "protected_delete_count": sum(
            entry["action"] == "delete"
            and entry["artifact_role"]
            in {
                "table_1_metric_summary",
                "table_2_uq_summary",
                "stage_best_checkpoint",
                "stage_initialization_checkpoint",
                "resolved_protocol",
                "thresholds",
                "retention_manifest",
                "retention_summary",
            }
            for entry in entries
        ),
        "absolute_paths_only": all(
            Path(entry["absolute_path"]).is_absolute() for entry in entries
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary_data = payload["summary"]
    lines = [
        "# Prune manifest dry-run",
        "",
        "Đây là danh sách kiểm tra; chưa có file remote nào bị xóa.",
        "",
        f"- Tạo lúc (UTC): `{payload['created_at_utc']}`",
        f"- Remote root: `{REMOTE_ROOT}`",
        f"- Số entry: `{summary_data['entry_count']}`",
        f"- `keep`: `{summary_data['action_counts'].get('keep', 0)}`",
        f"- `review`: `{summary_data['action_counts'].get('review', 0)}`",
        f"- `delete` candidate: `{summary_data['action_counts'].get('delete', 0)}`",
        f"- Dung lượng `delete` candidate: `{summary_data['delete_gib']} GiB`",
        f"- Protected artifact bị đánh dấu delete: `{summary_data['protected_delete_count']}`",
        "",
        "## Quy tắc dry-run",
        "",
        "- Chỉ raw trace có canonical report row hợp lệ mới được đánh dấu `delete`.",
        "- Summary, protocol, threshold, retention manifest/summary và checkpoint đều phải `keep`.",
        "- `offline_metrics.json` giữ ở `review` cho tới khi chốt source discrepancy.",
        "- `review` không phải lệnh xóa; cần quyết định riêng ở phase sau.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    entries = build_entries(read_inventory(), report)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_root": REMOTE_ROOT,
        "source_report": str(args.report),
        "scope": "outputs/benchmark/smd",
        "summary": summary(entries),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown:
        write_markdown(args.markdown, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

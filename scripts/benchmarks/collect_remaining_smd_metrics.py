from __future__ import annotations

"""Collect only final requested metrics from a remaining-SMD manifest."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

FPR_BUDGET_LABELS = {"0.001": "0.1%", "0.005": "0.5%", "0.01": "1%"}
REQUESTED_METRICS = (
    "VUS-PR@FPR-budget",
    "VUS-PR",
    "Affiliation F1-score",
    "VUS-ROC",
    "raw-FPR",
)


def _metric_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "vus_pr" in value and "vus_roc" in value:
            found.append(value)
        for child in value.values():
            found.extend(_metric_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_metric_dicts(child))
    return found


def _number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_requested_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize offline and online runner metric keys into the report contract."""
    candidates = _metric_dicts(payload)
    if not candidates:
        return {}
    metrics = candidates[-1]
    budget_values = metrics.get("vus_pr_at_fpr_budget", {})
    normalized = {
        "VUS-PR@FPR-budget": {
            label: _number(budget_values.get(key))
            for key, label in FPR_BUDGET_LABELS.items()
        },
        "VUS-PR": _number(metrics.get("vus_pr")),
        "Affiliation F1-score": _number(metrics.get("affiliation_f1")),
        "VUS-ROC": _number(metrics.get("vus_roc")),
        "raw-FPR": _number(metrics.get("fpr")),
    }
    if any(value is None for value in normalized.values() if not isinstance(value, dict)):
        return {}
    if any(value is None for value in normalized["VUS-PR@FPR-budget"].values()):
        return {}
    return normalized


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _find_payload(run: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = Path(str(run["output_dir"]))
    candidates = [
        Path(str(run["report_path"])),
        output_dir / "evaluation_metrics.json",
        output_dir / "metrics/offline_metrics.json",
        output_dir / "online_metrics.json",
    ]
    for path in candidates:
        payload = _load_json(path)
        if payload is not None:
            return payload
    return None


def collect_manifest_metrics(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        payload = _find_payload(run)
        metrics = extract_requested_metrics(payload or {})
        rows.append(
            {
                "run_id": run["run_id"],
                "phase": run["phase"],
                "method": run["method"],
                "variant": run.get("variant"),
                "entity_id": run["entity_id"],
                "seed": run["seed"],
                "status": (
                    "completed"
                    if metrics
                    else "incomplete"
                    if payload
                    else "missing"
                ),
                "metrics": metrics,
            }
        )
    return {"mode": manifest["mode"], "metrics": list(REQUESTED_METRICS), "runs": rows}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Remaining-SMD metrics ({report['mode']})",
        "",
        "Main metric: `VUS-PR@FPR-budget`.",
        "",
        "| Run | Phase | Method | Variant | Entity | Seed | Status | VUS-PR@FPR-budget (0.1% / 0.5% / 1%) | VUS-PR | Affiliation F1-score | VUS-ROC | raw-FPR |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        metrics = row["metrics"]
        budget = metrics.get("VUS-PR@FPR-budget", {})
        budget_text = " / ".join(str(budget.get(label)) for label in ("0.1%", "0.5%", "1%"))
        values = [
            row["run_id"], row["phase"], row["method"], row.get("variant") or "-",
            row["entity_id"], str(row["seed"]), row["status"], budget_text,
            str(metrics.get("VUS-PR")), str(metrics.get("Affiliation F1-score")),
            str(metrics.get("VUS-ROC")), str(metrics.get("raw-FPR")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    report = collect_manifest_metrics(args.manifest)
    output_root = args.manifest.parent
    json_path = output_root / "remaining_smd_metrics.json"
    markdown_path = output_root / "remaining_smd_metrics.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(markdown_path)


if __name__ == "__main__":
    main()

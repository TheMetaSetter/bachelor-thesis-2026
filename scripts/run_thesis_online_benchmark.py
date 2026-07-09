from __future__ import annotations

"""THESIS online benchmark wrapper.

₍^. .^₎⟆ Online benchmark boundary

offline checkpoint + protocol thresholds
  -> A0/A1/A2 online runner
  -> normalized online records
  -> benchmark report

The first implementation slice exposes the report boundary for A0. The full
streaming scorer/adaptation policy can then grow behind this stable contract.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.engine.online_tta.online_engine import run_thesis_online_tta_experiment
from src.protocols.smd_benchmark_protocol import validate_protocol_config


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_yaml_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_experiment_config_with_compatibility(
    experiment_config_path: str,
) -> dict[str, Any]:
    raw_config = _load_yaml_config(experiment_config_path)
    has_config_references = all(
        key in raw_config
        for key in ("data_config_path", "model_config_path", "task_config_path")
    )
    if not has_config_references:
        return raw_config
    try:
        return load_experiment_config(experiment_config_path)
    except (FileNotFoundError, ValueError):
        return raw_config


def _normalize_online_records(
    records: list[dict[str, Any]],
    online_variant: str,
) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        normalized_record = dict(record)
        normalized_record.setdefault("online_variant", online_variant)
        if online_variant == "A0":
            normalized_record["did_update"] = False
        else:
            normalized_record.setdefault("did_update", True)
        normalized_records.append(normalized_record)
    return normalized_records


def _write_report(
    output_dir: Path,
    online_variant: str,
    report: dict[str, Any],
) -> Path:
    report_dir = output_dir / "benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"thesis_online_{online_variant}_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return report_path


def run_thesis_online_benchmark(
    *,
    experiment_config_path: str,
    protocol_config_path: str,
    online_variant: str,
    dry_run: bool,
) -> dict[str, Any]:
    if online_variant not in {"A0", "A1", "A2"}:
        raise ValueError("online_variant must be one of: A0, A1, A2")

    experiment_config = _load_experiment_config_with_compatibility(
        experiment_config_path
    )
    protocol_config = _load_yaml_config(protocol_config_path)
    validate_protocol_config(protocol_config)

    online_outputs = run_thesis_online_tta_experiment(
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        dry_run=dry_run,
    )
    online_outputs = dict(online_outputs)
    online_outputs["records"] = _normalize_online_records(
        list(online_outputs.get("records", [])),
        online_variant,
    )

    report = {
        "benchmark_status": "dry_run" if dry_run else "completed",
        "created_at_utc": _utc_now_iso(),
        "experiment_config_path": experiment_config_path,
        "online_variant": online_variant,
        "protocol_config_path": protocol_config_path,
        "protocol": protocol_config,
        "online_execution": online_outputs,
    }
    report_path = _write_report(
        Path(str(experiment_config["output_dir"])),
        online_variant,
        report,
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
    )
    parser.add_argument("--online-variant", choices=["A0", "A1", "A2"], default="A0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_thesis_online_benchmark(
        experiment_config_path=args.experiment_config,
        protocol_config_path=args.protocol_config,
        online_variant=args.online_variant,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

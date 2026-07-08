from __future__ import annotations

"""THESIS offline benchmark wrapper.

₍₍⚞(˶˃ ꒳ ˂˶)⚟⁾⁾ How this wrapper fits

experiment config + protocol config
  -> validate locked fairness rules
  -> materialize existing two-stage plan
  -> execute existing two-stage runner
  -> write one benchmark report

This file does not train a model by itself. It delegates training to
`scripts/run_two_stage_offline_pretraining.py` so there is only one owner for
Stage A and Stage B behavior.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_two_stage_offline_pretraining import (
    execute_two_stage_plan,
    materialize_two_stage_run_manifest,
    validate_two_stage_epoch_budget,
)
from src.core.config import load_experiment_config
from src.protocols.smd_benchmark_protocol import validate_protocol_config


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_yaml_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_dir = output_dir / "benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "thesis_offline_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return report_path


def run_thesis_offline_benchmark(
    *,
    experiment_config_path: str,
    protocol_config_path: str,
    dry_run: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    experiment_config = load_experiment_config(experiment_config_path)
    protocol_config = _load_yaml_config(protocol_config_path)

    validate_protocol_config(protocol_config)
    validate_two_stage_epoch_budget(experiment_config)

    manifest = materialize_two_stage_run_manifest(experiment_config)
    execution_report = execute_two_stage_plan(
        manifest,
        dry_run=dry_run,
        skip_completed=skip_completed,
    )
    report = {
        "benchmark_status": "dry_run" if dry_run else execution_report["status"],
        "created_at_utc": _utc_now_iso(),
        "experiment_config_path": experiment_config_path,
        "protocol_config_path": protocol_config_path,
        "protocol": protocol_config,
        "two_stage_manifest": manifest,
        "two_stage_execution": execution_report,
    }
    report_path = _write_report(Path(str(experiment_config["output_dir"])), report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    report = run_thesis_offline_benchmark(
        experiment_config_path=args.experiment_config,
        protocol_config_path=args.protocol_config,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verification_summary_path(output_dir: Path) -> Path:
    return output_dir / "three_stage" / "three_stage_run_verification.json"


def build_three_stage_run_verification_summary(output_dir: str) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir).resolve()
    three_stage_dir = resolved_output_dir / "three_stage"
    preflight_summary_path = three_stage_dir / "server_preflight_summary.json"
    manifest_path = three_stage_dir / "three_stage_manifest.json"
    execution_report_path = three_stage_dir / "three_stage_execution_report.json"
    metrics_path = resolved_output_dir / "evaluation_metrics.json"

    missing_artifacts = [
        str(path)
        for path in [
            preflight_summary_path,
            manifest_path,
            execution_report_path,
        ]
        if not path.exists()
    ]

    if missing_artifacts:
        return {
            "output_dir": str(resolved_output_dir),
            "status": "incomplete_artifacts",
            "missing_artifacts": missing_artifacts,
            "has_evaluation_metrics": metrics_path.exists(),
        }

    preflight_summary = _load_json(preflight_summary_path)
    manifest = _load_json(manifest_path)
    execution_report = _load_json(execution_report_path)

    evaluation_checkpoint_path = Path(
        str(
            execution_report.get(
                "evaluation_checkpoint_path",
                manifest.get("evaluation", {}).get("checkpoint_path", ""),
            )
        )
    )
    required_success_paths = [metrics_path, evaluation_checkpoint_path]
    missing_success_artifacts = [
        str(path) for path in required_success_paths if not path.exists()
    ]

    execution_status = str(execution_report.get("status", "unknown"))
    if execution_status == "unknown":
        executed_stage_names = list(execution_report.get("executed_stage_names", []))
        if executed_stage_names and executed_stage_names[-1] == "evaluation":
            execution_status = "completed_legacy"
    verification_status = "incomplete_artifacts"
    if execution_status == "failed":
        verification_status = "failed_run_detected"
    elif execution_status in {"completed", "completed_legacy"} and not missing_success_artifacts:
        verification_status = "verified_success"

    return {
        "output_dir": str(resolved_output_dir),
        "status": verification_status,
        "execution_status": execution_status,
        "launch_readiness_status": preflight_summary.get("launch_readiness", {}).get(
            "status"
        ),
        "missing_artifacts": missing_success_artifacts,
        "has_evaluation_metrics": metrics_path.exists(),
        "evaluation_metrics_path": str(metrics_path),
        "evaluation_checkpoint_path": str(evaluation_checkpoint_path),
        "completed_stage_names": execution_report.get("completed_stage_names", []),
        "failed_stage_name": execution_report.get("failed_stage_name"),
        "failed_returncode": execution_report.get("failed_returncode"),
        "preflight_summary_path": str(preflight_summary_path),
        "manifest_path": str(manifest_path),
        "execution_report_path": str(execution_report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a completed or failed three-stage offline pre-training run"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Experiment output directory that contains the three_stage artifacts",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Exit non-zero unless the run verifies as a successful completed run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_three_stage_run_verification_summary(args.output_dir)
    summary_path = _verification_summary_path(Path(args.output_dir).resolve())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.require_success and summary["status"] != "verified_success":
        raise SystemExit(f"Run verification failed: {json.dumps(summary)}")


if __name__ == "__main__":
    main()

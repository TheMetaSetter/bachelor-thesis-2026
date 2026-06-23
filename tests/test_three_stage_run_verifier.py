from __future__ import annotations

import json
from pathlib import Path
import subprocess

from scripts.verify_three_stage_run import build_three_stage_run_verification_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _materialize_completed_run(tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs" / "completed-run"
    three_stage_dir = output_dir / "three_stage"
    metrics_path = output_dir / "evaluation_metrics.json"
    checkpoint_path = output_dir / "three_stage" / "multitask_pretraining" / "checkpoints" / "best.pt"

    _write_json(
        three_stage_dir / "server_preflight_summary.json",
        {
            "experiment_name": "completed-run",
            "launch_readiness": {"status": "ready_for_rtx3090_tmux_launch"},
        },
    )
    _write_json(
        three_stage_dir / "three_stage_manifest.json",
        {
            "experiment_name": "completed-run",
            "evaluation": {"checkpoint_path": str(checkpoint_path)},
            "statistical_procedure_names": [
                "stage2_mtz_parameter_zipping",
                "stage3_memory_initialization",
            ],
        },
    )
    _write_json(
        three_stage_dir / "three_stage_execution_report.json",
        {
            "experiment_name": "completed-run",
            "status": "completed",
            "completed_stage_names": [
                "stage1_classification",
                "stage1_reconstruction",
                "stage2_recovery",
                "stage3_memory_initialization_and_fusion_warmup",
                "multitask_pretraining",
                "evaluation",
            ],
            "evaluation_checkpoint_path": str(checkpoint_path),
            "optimizer_training_phase_names": [
                "stage1_classification",
                "stage1_reconstruction",
                "stage2_recovery",
                "stage3_memory_initialization_and_fusion_warmup",
                "multitask_pretraining",
            ],
            "statistical_procedure_names": [
                "stage2_mtz_parameter_zipping",
                "stage3_memory_initialization",
            ],
            "started_at_utc": "2026-06-23T01:02:03Z",
            "finished_at_utc": "2026-06-23T03:04:05Z",
        },
    )
    _write_json(metrics_path, {"roc_auc": 0.75})
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    return output_dir


def _materialize_failed_run(tmp_path: Path) -> Path:
    output_dir = tmp_path / "outputs" / "failed-run"
    three_stage_dir = output_dir / "three_stage"

    _write_json(
        three_stage_dir / "server_preflight_summary.json",
        {
            "experiment_name": "failed-run",
            "launch_readiness": {"status": "ready_for_rtx3090_tmux_launch"},
        },
    )
    _write_json(
        three_stage_dir / "three_stage_manifest.json",
        {
            "experiment_name": "failed-run",
            "evaluation": {"checkpoint_path": str(output_dir / "missing-best.pt")},
        },
    )
    _write_json(
        three_stage_dir / "three_stage_execution_report.json",
        {
            "experiment_name": "failed-run",
            "status": "failed",
            "completed_stage_names": ["stage1_classification"],
            "failed_stage_name": "stage1_reconstruction",
            "failed_command": ["python", "scripts/train.py"],
            "failed_returncode": 17,
            "started_at_utc": "2026-06-23T01:02:03Z",
            "failed_at_utc": "2026-06-23T01:05:00Z",
        },
    )
    return output_dir


def test_three_stage_run_verifier_recognizes_completed_verified_run(tmp_path: Path) -> None:
    output_dir = _materialize_completed_run(tmp_path)

    summary = build_three_stage_run_verification_summary(str(output_dir))

    assert summary["status"] == "verified_success"
    assert summary["execution_status"] == "completed"
    assert summary["missing_artifacts"] == []
    assert summary["has_evaluation_metrics"] is True
    assert summary["launch_readiness_status"] == "ready_for_rtx3090_tmux_launch"
    assert summary["optimizer_training_phase_names"] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    assert summary["statistical_procedure_names"] == [
        "stage2_mtz_parameter_zipping",
        "stage3_memory_initialization",
    ]


def test_three_stage_run_verifier_can_infer_success_from_legacy_execution_report(
    tmp_path: Path,
) -> None:
    output_dir = _materialize_completed_run(tmp_path)
    legacy_report_path = output_dir / "three_stage" / "three_stage_execution_report.json"
    legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    legacy_report.pop("status")
    legacy_report.pop("completed_stage_names")
    legacy_report["executed_stage_names"] = [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
        "evaluation",
    ]
    legacy_report_path.write_text(json.dumps(legacy_report, indent=2), encoding="utf-8")

    summary = build_three_stage_run_verification_summary(str(output_dir))

    assert summary["status"] == "verified_success"
    assert summary["execution_status"] == "completed_legacy"


def test_three_stage_run_verifier_falls_back_to_manifest_metadata_for_legacy_report(
    tmp_path: Path,
) -> None:
    output_dir = _materialize_completed_run(tmp_path)
    legacy_report_path = output_dir / "three_stage" / "three_stage_execution_report.json"
    legacy_report = json.loads(legacy_report_path.read_text(encoding="utf-8"))
    legacy_report.pop("optimizer_training_phase_names")
    legacy_report.pop("statistical_procedure_names")
    legacy_report_path.write_text(json.dumps(legacy_report, indent=2), encoding="utf-8")
    manifest_path = output_dir / "three_stage" / "three_stage_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["optimizer_training_phase_names"] = [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    summary = build_three_stage_run_verification_summary(str(output_dir))

    assert summary["optimizer_training_phase_names"] == [
        "stage1_classification",
        "stage1_reconstruction",
        "stage2_recovery",
        "stage3_memory_initialization_and_fusion_warmup",
        "multitask_pretraining",
    ]
    assert summary["statistical_procedure_names"] == [
        "stage2_mtz_parameter_zipping",
        "stage3_memory_initialization",
    ]


def test_three_stage_run_verifier_recognizes_failed_run_and_preserves_failure_context(
    tmp_path: Path,
) -> None:
    output_dir = _materialize_failed_run(tmp_path)

    summary = build_three_stage_run_verification_summary(str(output_dir))

    assert summary["status"] == "failed_run_detected"
    assert summary["execution_status"] == "failed"
    assert summary["failed_stage_name"] == "stage1_reconstruction"
    assert summary["failed_returncode"] == 17
    assert summary["has_evaluation_metrics"] is False


def test_three_stage_run_verifier_cli_require_success_fails_for_incomplete_run(
    tmp_path: Path,
) -> None:
    output_dir = _materialize_completed_run(tmp_path)
    (output_dir / "evaluation_metrics.json").unlink()

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/verify_three_stage_run.py",
            "--output-dir",
            str(output_dir),
            "--require-success",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Run verification failed" in completed.stderr


def test_three_stage_run_verifier_cli_writes_summary_artifact(tmp_path: Path) -> None:
    output_dir = _materialize_completed_run(tmp_path)

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/verify_three_stage_run.py",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = output_dir / "three_stage" / "three_stage_run_verification.json"
    assert summary_path.exists()
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    printed_summary = json.loads(completed.stdout)
    assert saved_summary["status"] == "verified_success"
    assert saved_summary == printed_summary

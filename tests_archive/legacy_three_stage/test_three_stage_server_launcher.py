from __future__ import annotations

import subprocess


def test_tmux_launcher_dry_run_prints_exact_server_command() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/launch_tmux_three_stage_experiment.sh",
            "--dry-run",
            "--experiment-config",
            "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml",
            "--session-name",
            "unit-three-stage-smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tmux session: unit-three-stage-smoke" in completed.stdout
    assert "unit-three-stage-smoke.log" in completed.stdout
    assert "attach command: tmux attach -t unit-three-stage-smoke" in completed.stdout
    assert "preflight_three_stage_server.py" in completed.stdout
    assert "run_three_stage_offline_pretraining.py" in completed.stdout
    assert "verify_three_stage_run.py" in completed.stdout
    assert "server_preflight_summary.json" in completed.stdout
    assert "optimizer training phases:" in completed.stdout
    assert "stage3_memory_initialization_and_fusion_warmup" in completed.stdout
    assert "statistical procedures:" in completed.stdout
    assert "tmux inner command: (" in completed.stdout
    assert ") > " in completed.stdout


def test_tmux_launcher_reports_missing_tmux_binary_clearly() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/launch_tmux_three_stage_experiment.sh",
            "--experiment-config",
            "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml",
            "--session-name",
            "unit-three-stage-smoke",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 127
    assert "tmux is required but was not found in PATH." in completed.stderr

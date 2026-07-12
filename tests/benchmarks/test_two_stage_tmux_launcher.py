from __future__ import annotations

import subprocess


def test_two_stage_tmux_launcher_dry_run_prints_expected_benchmark_command() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/launch_tmux_two_stage_experiment.sh",
            "--dry-run",
            "--session-name",
            "unit-two-stage-benchmark",
            "--experiment-config",
            "configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_3_4__w20__seed8__smoke.yaml",
            "--protocol-config",
            "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
            "--gpu-index",
            "2",
            "--required-gpu-name-substring",
            "RTX 3090",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tmux session: unit-two-stage-benchmark" in completed.stdout
    assert "unit-two-stage-benchmark.log" in completed.stdout
    assert "thesis_offline_benchmark_report.json" in completed.stdout
    assert "attach command: tmux attach -t unit-two-stage-benchmark" in completed.stdout
    assert "run_thesis_offline_benchmark.py" in completed.stdout
    assert "--experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_3_4__w20__seed8__smoke.yaml" in completed.stdout
    assert "--protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml" in completed.stdout
    assert "CUDA_VISIBLE_DEVICES=2" in completed.stdout

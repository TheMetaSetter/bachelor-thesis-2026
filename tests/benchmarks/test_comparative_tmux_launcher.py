from __future__ import annotations

import subprocess


def test_comparative_tmux_launcher_dry_run_prints_expected_runner_commands() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/launch_tmux_comparative_smd_experiment.sh",
            "--dry-run",
            "--session-name",
            "unit-comparative-smd",
            "--report-dir",
            "outputs/comparative_smd_reports/unit-comparative-smd",
            "--gpu-index",
            "2",
            "--required-gpu-name-substring",
            "RTX 3090",
            "--data-num-workers-override",
            "4",
            "--smoke-profile",
            "functional",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "tmux session: unit-comparative-smd" in completed.stdout
    assert "unit-comparative-smd.log" in completed.stdout
    assert "comparative_manifest.json" in completed.stdout
    assert "comparative_execution_report.json" in completed.stdout
    assert "attach command: tmux attach -t unit-comparative-smd" in completed.stdout
    assert "run_comparative_smd_experiments.py" in completed.stdout
    assert "preflight_comparative_smd_server.py" in completed.stdout
    assert "--smoke-config-paths" in completed.stdout
    assert "--config-paths" in completed.stdout
    assert "--gpu-index 2" in completed.stdout
    assert "--required-gpu-name-substring RTX 3090" in completed.stdout
    assert "--data-num-workers-override 4" in completed.stdout
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in completed.stdout
    assert "CUDA_VISIBLE_DEVICES=2" in completed.stdout

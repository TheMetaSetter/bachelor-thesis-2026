from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ops.preflight_comparative_smd_server import (
    build_comparative_preflight_summary,
)


def _write_placeholder_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("placeholder: true\n", encoding="utf-8")


def _build_stub_config(
    *,
    experiment_name: str,
    dataset_root: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    entity_id: str,
    seed: int,
    model_name: str,
    device: str,
    num_workers: int,
    include_three_stage: bool,
) -> dict[str, object]:
    config: dict[str, object] = {
        "experiment_name": experiment_name,
        "seed": seed,
        "device": device,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "data": {
            "root_dir": str(dataset_root),
            "entity_ids": [entity_id],
            "num_workers": num_workers,
        },
        "model": {
            "model_name": model_name,
        },
    }
    if include_three_stage:
        config["three_stage"] = {
            "expected_total_training_epochs": 300,
        }
    return config


def test_comparative_preflight_summary_reports_launch_ready_for_valid_cuda_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    thesis_config_path = tmp_path / "configs" / "thesis.yaml"
    baseline_config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(thesis_config_path)
    _write_placeholder_config(baseline_config_path)

    stub_configs = {
        thesis_config_path.resolve(): _build_stub_config(
            experiment_name="thesis_run",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "thesis",
            checkpoint_dir=tmp_path / "outputs" / "thesis" / "checkpoints",
            entity_id="machine-3-9",
            seed=6,
            model_name="thesis_multitask",
            device="cuda",
            num_workers=4,
            include_three_stage=True,
        ),
        baseline_config_path.resolve(): _build_stub_config(
            experiment_name="baseline_run",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "baseline",
            checkpoint_dir=tmp_path / "outputs" / "baseline" / "checkpoints",
            entity_id="machine-1-6",
            seed=36,
            model_name="redlamp_baseline",
            device="cuda",
            num_workers=4,
            include_three_stage=False,
        ),
    }

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_configs[Path(config_path).resolve()],
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.shutil.which",
        lambda executable: "/usr/bin/fake" if executable == "tmux" else None,
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.torch.cuda.device_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.torch.cuda.get_device_name",
        lambda index: "NVIDIA GeForce RTX 3090",
    )

    summary = build_comparative_preflight_summary(
        config_paths=[thesis_config_path, baseline_config_path],
        report_dir=tmp_path / "reports",
        gpu_index=1,
        required_gpu_name_substring="RTX 3090",
    )

    assert summary["gpu_validation"]["status"] == "ok"
    assert summary["launch_readiness"]["status"] == "ready_for_comparative_tmux_launch"
    assert summary["launch_readiness"]["all_devices_are_cuda"] is True
    assert summary["launch_readiness"]["tmux_ready"] is True
    assert summary["launch_readiness"]["artifact_paths_unique"] is True
    assert summary["launch_readiness"]["data_roots_exist"] is True
    assert summary["requested_gpu_index"] == 1
    assert summary["report_dir"] == str((tmp_path / "reports").resolve())
    assert summary["preflight_summary_path"].endswith(
        "comparative_server_preflight_summary.json"
    )


def test_comparative_preflight_rejects_non_cuda_run_inside_gpu_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    config_path = tmp_path / "configs" / "baseline.yaml"
    _write_placeholder_config(config_path)

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: _build_stub_config(
            experiment_name="baseline_run",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "baseline",
            checkpoint_dir=tmp_path / "outputs" / "baseline" / "checkpoints",
            entity_id="machine-1-6",
            seed=36,
            model_name="redlamp_baseline",
            device="cpu",
            num_workers=0,
            include_three_stage=False,
        ),
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.shutil.which",
        lambda executable: "/usr/bin/fake" if executable == "tmux" else None,
    )

    summary = build_comparative_preflight_summary(
        config_paths=[config_path],
        report_dir=tmp_path / "reports",
        gpu_index=0,
        required_gpu_name_substring="RTX 3090",
    )

    assert summary["device_validation"]["status"] == "mixed_or_non_cuda_configs"
    assert summary["launch_readiness"]["status"] == "not_ready_for_comparative_launch"
    assert summary["launch_readiness"]["all_devices_are_cuda"] is False


def test_comparative_preflight_allows_cpu_functional_smoke_when_main_runs_are_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "data" / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    smoke_config_path = tmp_path / "configs" / "smoke.yaml"
    main_config_path = tmp_path / "configs" / "main.yaml"
    _write_placeholder_config(smoke_config_path)
    _write_placeholder_config(main_config_path)

    stub_configs = {
        smoke_config_path.resolve(): _build_stub_config(
            experiment_name="baseline_smoke",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "smoke",
            checkpoint_dir=tmp_path / "outputs" / "smoke" / "checkpoints",
            entity_id="machine-1-6",
            seed=6,
            model_name="redlamp_baseline",
            device="cpu",
            num_workers=0,
            include_three_stage=False,
        ),
        main_config_path.resolve(): _build_stub_config(
            experiment_name="baseline_main",
            dataset_root=dataset_root,
            output_dir=tmp_path / "outputs" / "main",
            checkpoint_dir=tmp_path / "outputs" / "main" / "checkpoints",
            entity_id="machine-1-6",
            seed=36,
            model_name="redlamp_baseline",
            device="cuda",
            num_workers=4,
            include_three_stage=False,
        ),
    }

    monkeypatch.setattr(
        "scripts.run_comparative_smd_experiments.load_experiment_config",
        lambda config_path: stub_configs[Path(config_path).resolve()],
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.shutil.which",
        lambda executable: "/usr/bin/fake" if executable == "tmux" else None,
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.torch.cuda.device_count",
        lambda: 1,
    )
    monkeypatch.setattr(
        "scripts.ops.preflight_comparative_smd_server.torch.cuda.get_device_name",
        lambda index: "NVIDIA GeForce RTX 3090",
    )

    summary = build_comparative_preflight_summary(
        config_paths=[main_config_path],
        smoke_config_paths=[smoke_config_path],
        report_dir=tmp_path / "reports",
        gpu_index=0,
        required_gpu_name_substring="RTX 3090",
    )

    assert summary["main_device_validation"]["status"] == "all_cuda"
    assert summary["smoke_device_validation"]["status"] == "all_cpu"
    assert summary["launch_readiness"]["status"] == "ready_for_comparative_tmux_launch"

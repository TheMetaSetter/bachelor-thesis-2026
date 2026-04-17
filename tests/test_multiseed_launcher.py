from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_multiseed_experiments import (
    build_train_command,
    load_resolved_experiment_configs,
    run_command_stage,
    validate_dataset_roots,
    validate_unique_artifact_paths,
)


def write_reconstruction_experiment_config(
    *,
    root_path: Path,
    experiment_name: str,
    seed: int,
    output_dir: str,
    checkpoint_dir: str,
    dataset_root: Path,
) -> Path:
    data_config_path = root_path / f"{experiment_name}_data.yaml"
    model_config_path = root_path / f"{experiment_name}_model.yaml"
    task_config_path = root_path / f"{experiment_name}_task.yaml"
    experiment_config_path = root_path / f"{experiment_name}_experiment.yaml"

    data_config_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                f"root_dir: {dataset_root}",
                "window_size: 100",
                "stride: 10",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: reconstruction_mlp_ae",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 16",
                "dropout: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: reconstruction",
                "loss_name: mse",
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                f"experiment_name: {experiment_name}",
                f"seed: {seed}",
                "device: cpu",
                f"output_dir: {output_dir}",
                f"checkpoint_dir: {checkpoint_dir}",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )
    return experiment_config_path


def test_build_train_command_uses_normalized_config_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data_root"
    dataset_root.mkdir()
    experiment_config_path = write_reconstruction_experiment_config(
        root_path=tmp_path,
        experiment_name="seed11",
        seed=11,
        output_dir="outputs/seed11",
        checkpoint_dir="outputs/seed11/checkpoints",
        dataset_root=dataset_root,
    )

    command = build_train_command(experiment_config_path)

    assert command[-2] == "--experiment-config"
    assert Path(command[-1]).is_absolute()
    assert Path(command[-1]) == experiment_config_path.resolve()


def test_validate_unique_artifact_paths_rejects_duplicate_output_dir(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data_root"
    dataset_root.mkdir()
    config_a = write_reconstruction_experiment_config(
        root_path=tmp_path,
        experiment_name="seed11",
        seed=11,
        output_dir="outputs/shared",
        checkpoint_dir="outputs/shared/checkpoints-a",
        dataset_root=dataset_root,
    )
    config_b = write_reconstruction_experiment_config(
        root_path=tmp_path,
        experiment_name="seed23",
        seed=23,
        output_dir="outputs/shared",
        checkpoint_dir="outputs/shared/checkpoints-b",
        dataset_root=dataset_root,
    )

    resolved_experiment_configs = load_resolved_experiment_configs([config_a, config_b])

    with pytest.raises(ValueError, match="Duplicate output_dir"):
        validate_unique_artifact_paths(resolved_experiment_configs)


def test_validate_dataset_roots_rejects_missing_root(tmp_path: Path) -> None:
    missing_dataset_root = tmp_path / "missing_data_root"
    experiment_config_path = write_reconstruction_experiment_config(
        root_path=tmp_path,
        experiment_name="seed11",
        seed=11,
        output_dir="outputs/seed11",
        checkpoint_dir="outputs/seed11/checkpoints",
        dataset_root=missing_dataset_root,
    )

    resolved_experiment_configs = load_resolved_experiment_configs([experiment_config_path])

    with pytest.raises(FileNotFoundError, match="Dataset root does not exist"):
        validate_dataset_roots(resolved_experiment_configs)


def test_run_command_stage_dry_run_accepts_three_distinct_configs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data_root"
    dataset_root.mkdir()
    config_paths = [
        write_reconstruction_experiment_config(
            root_path=tmp_path,
            experiment_name=f"seed{seed}",
            seed=seed,
            output_dir=f"outputs/seed{seed}",
            checkpoint_dir=f"outputs/seed{seed}/checkpoints",
            dataset_root=dataset_root,
        )
        for seed in (11, 23, 47)
    ]

    resolved_experiment_configs = load_resolved_experiment_configs(config_paths)
    validate_unique_artifact_paths(resolved_experiment_configs)
    validate_dataset_roots(resolved_experiment_configs)

    run_command_stage(
        config_paths=config_paths,
        execution_mode="parallel",
        max_concurrent_processes=3,
        dry_run=True,
    )

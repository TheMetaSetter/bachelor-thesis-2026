from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_experiment_config, load_yaml_config


def test_load_yaml_config_rejects_duplicate_root_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: duplicate-key-smoke",
                "model_overrides:",
                "  a: 1",
                "model_overrides:",
                "  b: 2",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate key in YAML mapping"):
        load_yaml_config(config_path)


def test_all_repo_config_files_have_no_duplicate_root_keys() -> None:
    config_paths = sorted(Path("configs").rglob("*.yaml"))
    assert config_paths, "Expected YAML config files under configs/"

    for config_path in config_paths:
        load_yaml_config(config_path)


@pytest.mark.parametrize(
    "experiment_config_path,mutator,expected_error",
    [
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["optimizer"]["scheduler"].update({"factor": 0.5}),
            "Unknown optimizer.scheduler keys for cosine",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["task"].update(
                {"classification_label_mode": "redlamp_multiclass"}
            )
            or cfg["model"].update({"num_classes": 2}),
            "requires num_classes == 12",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["data"].update({"num_workers": "many"}),
            "data.num_workers must be a non-negative integer or 'auto'",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["task"].update({"anomaly_probability": 1.5}),
            "anomaly_probability must be between 0 and 1",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["logging"].update(
                {"use_wandb": False, "wandb_mode": "online"}
            ),
            "logging.use_wandb is false",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["logging"].update(
                {"use_wandb": True, "wandb_mode": "disabled"}
            ),
            "logging.use_wandb is true",
        ),
        (
            "configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml",
            lambda cfg: cfg["optimizer"]["scheduler"].update(
                {"monitor_metric": "val_realistic_pr_auc"}
            ),
            "must match for reduce_on_plateau",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["task"].update({"unknown_task_flag": True}),
            "Unknown task config keys",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["optimizer"].update({"unexpected_optimizer_key": 1}),
            "Unknown optimizer config keys",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["logging"].update({"unexpected_logging_key": True}),
            "Unknown logging config keys",
        ),
        (
            "configs/experiment/smoke/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            lambda cfg: cfg["optimizer"]["scheduler"].update(
                {"mystery_scheduler_flag": 42}
            ),
            "Unknown optimizer.scheduler keys for cosine",
        ),
    ],
)
def test_validate_pipeline_rejects_semantic_config_contradictions(
    experiment_config_path: str,
    mutator,
    expected_error: str,
) -> None:
    config = load_experiment_config(experiment_config_path)
    mutator(config)

    with pytest.raises(ValueError, match=expected_error):
        from src.core.config import validate_experiment_config

        validate_experiment_config(config)

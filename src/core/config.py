from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded_config = yaml.safe_load(handle) or {}

    if not isinstance(loaded_config, dict):
        raise ValueError(f"Config file must contain a mapping at the root: {path}")

    return loaded_config


def validate_experiment_config(experiment_config: dict[str, Any]) -> None:
    required_sections = [
        "experiment_name",
        "seed",
        "device",
        "output_dir",
        "checkpoint_dir",
        "data",
        "model",
        "task",
        "optimizer",
        "epochs",
    ]
    for section_name in required_sections:
        if section_name not in experiment_config:
            raise ValueError(f"Experiment config is missing required section: {section_name}")

    data_config = experiment_config["data"]
    model_config = experiment_config["model"]
    task_config = experiment_config["task"]
    optimizer_config = experiment_config["optimizer"]

    if data_config.get("dataset_name") != "smd":
        raise ValueError("Phase 1 only supports dataset_name='smd'")
    if model_config.get("model_name") != "reconstruction_mlp_ae":
        raise ValueError("Phase 1 only supports model_name='reconstruction_mlp_ae'")
    if task_config.get("task_name") != "reconstruction":
        raise ValueError("Phase 1 only supports task_name='reconstruction'")

    integer_fields = {
        "seed": experiment_config["seed"],
        "epochs": experiment_config["epochs"],
        "window_size": data_config.get("window_size"),
        "stride": data_config.get("stride"),
        "batch_size": data_config.get("batch_size"),
        "encoder_dim": model_config.get("encoder_dim"),
        "hidden_dim": model_config.get("hidden_dim"),
        "input_dim": model_config.get("input_dim"),
    }
    for field_name, field_value in integer_fields.items():
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    float_fields = {
        "validation_split_ratio": data_config.get("validation_split_ratio"),
        "learning_rate": optimizer_config.get("learning_rate"),
        "weight_decay": optimizer_config.get("weight_decay"),
        "dropout": model_config.get("dropout"),
    }
    for field_name, field_value in float_fields.items():
        if not isinstance(field_value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")

    if not 0.0 < float(data_config["validation_split_ratio"]) < 1.0:
        raise ValueError("validation_split_ratio must be between 0 and 1")
    if data_config["stride"] > data_config["window_size"]:
        raise ValueError("stride must not exceed window_size")

    optional_window_limit_fields = [
        "max_train_windows",
        "max_val_windows",
        "max_test_windows",
    ]
    for field_name in optional_window_limit_fields:
        field_value = data_config.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer when provided")


def load_experiment_config(experiment_config_path: str | Path) -> dict[str, Any]:
    experiment_path = Path(experiment_config_path)
    root_config = load_yaml_config(experiment_path)

    required_reference_fields = [
        "data_config_path",
        "model_config_path",
        "task_config_path",
    ]
    for reference_field in required_reference_fields:
        if reference_field not in root_config:
            raise ValueError(f"Experiment config is missing file reference: {reference_field}")

    resolved_experiment_config = dict(root_config)
    for section_name, reference_field in [
        ("data", "data_config_path"),
        ("model", "model_config_path"),
        ("task", "task_config_path"),
    ]:
        config_reference = Path(root_config[reference_field])
        if not config_reference.is_absolute():
            config_reference = experiment_path.parent.parent / config_reference.relative_to("configs")
        resolved_experiment_config[section_name] = load_yaml_config(config_reference)

    validate_experiment_config(resolved_experiment_config)
    return resolved_experiment_config

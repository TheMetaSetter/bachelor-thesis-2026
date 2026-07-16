from __future__ import annotations

from typing import Any

from src.core.console import console_print
from src.core.config_model_validation import (
    _validate_data_runtime_config,
    _validate_model_and_task_config,
    _validate_model_and_task_semantics,
)


def validate_experiment_config(experiment_config: dict[str, Any]) -> None:
    from src.core.config import (
        _resolve_thesis_model_window_size,
        _validate_data_config,
        _validate_experiment_top_level_structure,
        _validate_logging_config,
        _validate_optimizer_config,
        _validate_three_stage_config,
        _validate_two_stage_config,
    )

    # Validation is intentionally centralized here so the rest of the runtime
    # can assume a decision-complete experiment config.
    required_sections = _validate_experiment_top_level_structure(experiment_config)
    console_print(
        "CONFIG",
        "Validated experiment config sections",
        experiment_name=experiment_config["experiment_name"],
        required_sections=required_sections,
    )

    data_config = experiment_config["data"]
    model_config = experiment_config["model"]
    task_config = experiment_config["task"]
    optimizer_config = experiment_config["optimizer"]
    three_stage_config = experiment_config.get("three_stage")
    two_stage_config = experiment_config.get("two_stage")
    if three_stage_config is not None and two_stage_config is not None:
        raise ValueError(
            "Experiment config cannot define both three_stage and two_stage"
        )
    if three_stage_config is not None:
        if not isinstance(three_stage_config, dict):
            raise ValueError("three_stage must be a mapping when provided")
        _validate_three_stage_config(three_stage_config)
    if two_stage_config is not None:
        if not isinstance(two_stage_config, dict):
            raise ValueError("two_stage must be a mapping when provided")
        _validate_two_stage_config(two_stage_config)
    initialization_checkpoint_path = experiment_config.get(
        "initialization_checkpoint_path"
    )
    if initialization_checkpoint_path is not None and (
        not isinstance(initialization_checkpoint_path, str)
        or not initialization_checkpoint_path
    ):
        raise ValueError(
            "initialization_checkpoint_path must be a non-empty string when provided"
        )
    if (
        task_config.get("task_name") == "online_adaptation"
        and task_config.get("seed") is not None
    ):
        if task_config["seed"] != experiment_config["seed"]:
            raise ValueError("online_adaptation task seed must match top-level seed")

    _resolve_thesis_model_window_size(experiment_config)
    _validate_data_config(data_config)
    _validate_data_runtime_config(data_config)

    _validate_model_and_task_config(
        data_config=data_config,
        model_config=model_config,
        task_config=task_config,
    )
    _validate_model_and_task_semantics(
        data_config=data_config,
        model_config=model_config,
        task_config=task_config,
    )

    checkpoint_monitor_metric = experiment_config.get(
        "checkpoint_monitor_metric",
        "val_loss",
    )
    _validate_optimizer_config(
        optimizer_config,
        checkpoint_monitor_metric=checkpoint_monitor_metric,
    )

    logging_config = experiment_config.get("logging")
    if logging_config is not None:
        if not isinstance(logging_config, dict):
            raise ValueError("logging must be a mapping when provided")
        _validate_logging_config(logging_config)

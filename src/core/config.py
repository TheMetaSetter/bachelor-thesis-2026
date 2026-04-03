from __future__ import annotations
"""Config loading and validation for the active experiment families.

This file owns two readability-critical jobs: loading the three referenced YAML
files that define an experiment, and validating the merged result before the
runtime is built. A new reader should look here to understand how the baseline,
multitask, ablation, and online experiments stay configuration-driven.
"""

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


def _merge_config_section(
    base_config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    # Ablation experiments stay readable by layering a small override mapping on
    # top of a shared base config instead of duplicating entire YAML files.
    if overrides is None:
        return dict(base_config)
    if not isinstance(overrides, dict):
        raise ValueError("Config overrides must be mappings")
    merged_config = dict(base_config)
    merged_config.update(overrides)
    return merged_config


def validate_experiment_config(experiment_config: dict[str, Any]) -> None:
    # Validation is intentionally centralized here so the rest of the runtime
    # can assume a decision-complete experiment config.
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

    supported_dataset_names = {"smd"}
    supported_model_names = {"reconstruction_mlp_ae", "thesis_multitask", "online_adaptation"}
    supported_task_names = {"reconstruction", "multitask_tsad", "online_adaptation"}

    if data_config.get("dataset_name") not in supported_dataset_names:
        raise ValueError(f"Unsupported dataset_name: {data_config.get('dataset_name')}")
    if model_config.get("model_name") not in supported_model_names:
        raise ValueError(f"Unsupported model_name: {model_config.get('model_name')}")
    if task_config.get("task_name") not in supported_task_names:
        raise ValueError(f"Unsupported task_name: {task_config.get('task_name')}")

    integer_fields = {
        "seed": experiment_config["seed"],
        "epochs": experiment_config["epochs"],
        "window_size": data_config.get("window_size"),
        "stride": data_config.get("stride"),
        "batch_size": data_config.get("batch_size"),
        "input_dim": model_config.get("input_dim"),
    }
    if model_config.get("model_name") in {"reconstruction_mlp_ae", "thesis_multitask", "online_adaptation"}:
        integer_fields["encoder_dim"] = model_config.get("encoder_dim")
        integer_fields["hidden_dim"] = model_config.get("hidden_dim")
    if model_config.get("model_name") == "thesis_multitask":
        integer_fields["num_classes"] = model_config.get("num_classes")
        integer_fields["continuous_num_prototypes"] = model_config.get("continuous_num_prototypes")
        integer_fields["discrete_codebook_size"] = model_config.get("discrete_codebook_size")
    if model_config.get("model_name") == "online_adaptation":
        integer_fields["projector_hidden_dim"] = model_config.get("projector_hidden_dim")
    for field_name, field_value in integer_fields.items():
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    float_fields = {
        "validation_split_ratio": data_config.get("validation_split_ratio"),
        "learning_rate": optimizer_config.get("learning_rate"),
        "weight_decay": optimizer_config.get("weight_decay"),
    }
    if model_config.get("model_name") in {"reconstruction_mlp_ae", "thesis_multitask"}:
        float_fields["dropout"] = model_config.get("dropout")
    if task_config.get("task_name") == "multitask_tsad":
        float_fields["gumbel_temperature"] = model_config.get("gumbel_temperature")
        float_fields["temperature_start"] = model_config.get("temperature_start")
        float_fields["temperature_end"] = model_config.get("temperature_end")
        float_fields["temperature_anneal_fraction"] = model_config.get("temperature_anneal_fraction")
        float_fields["alpha_logit_init"] = model_config.get("alpha_logit_init")
        float_fields["beta_logit_init"] = model_config.get("beta_logit_init")
        float_fields["lambda_cls"] = model_config.get("lambda_cls")
        float_fields["lambda_div"] = model_config.get("lambda_div")
        float_fields["lambda_var"] = model_config.get("lambda_var")
        float_fields["lambda_cov"] = model_config.get("lambda_cov")
        float_fields["lambda_use"] = model_config.get("lambda_use")
        float_fields["lambda_gate"] = model_config.get("lambda_gate")
        float_fields["variance_floor_gamma"] = model_config.get("variance_floor_gamma")
        float_fields["gate_barrier_margin"] = model_config.get("gate_barrier_margin")
        float_fields["warmup_alpha_value"] = task_config.get("warmup_alpha_value")
        float_fields["warmup_beta_value"] = task_config.get("warmup_beta_value")
        float_fields["anomaly_probability"] = task_config.get("anomaly_probability")
        float_fields["min_segment_fraction"] = task_config.get("min_segment_fraction")
        float_fields["max_segment_fraction"] = task_config.get("max_segment_fraction")
        float_fields["spike_scale"] = task_config.get("spike_scale")
    if task_config.get("task_name") == "online_adaptation":
        float_fields["projector_dropout"] = model_config.get("projector_dropout")
        float_fields["lambda_align"] = model_config.get("lambda_align")
        float_fields["lambda_proto"] = model_config.get("lambda_proto")
        float_fields["lambda_anchor"] = model_config.get("lambda_anchor")
        float_fields["view_noise_std"] = task_config.get("view_noise_std")
        float_fields["view_dropout_probability"] = task_config.get("view_dropout_probability")
        float_fields["reset_alignment_threshold"] = task_config.get("reset_alignment_threshold")
    for field_name, field_value in float_fields.items():
        if not isinstance(field_value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")

    if task_config.get("task_name") == "multitask_tsad":
        boolean_fields = {
            "enable_diversity_loss": model_config.get("enable_diversity_loss", False),
            "enable_variance_loss": model_config.get("enable_variance_loss", False),
            "enable_covariance_loss": model_config.get("enable_covariance_loss", False),
            "enable_usage_loss": model_config.get("enable_usage_loss", False),
            "enable_gate_loss": model_config.get("enable_gate_loss", False),
            "use_synthetic_augmentation": task_config.get("use_synthetic_augmentation"),
        }
        for field_name, field_value in boolean_fields.items():
            if not isinstance(field_value, bool):
                raise ValueError(f"{field_name} must be a boolean")
    if task_config.get("task_name") == "online_adaptation":
        boolean_fields = {
            "enable_prototype_alignment": model_config.get("enable_prototype_alignment"),
            "warm_start_projector": task_config.get("warm_start_projector"),
            "clean_stream_only": task_config.get("clean_stream_only"),
        }
        for field_name, field_value in boolean_fields.items():
            if not isinstance(field_value, bool):
                raise ValueError(f"{field_name} must be a boolean")

    if not 0.0 < float(data_config["validation_split_ratio"]) < 1.0:
        raise ValueError("validation_split_ratio must be between 0 and 1")
    if data_config["stride"] > data_config["window_size"]:
        raise ValueError("stride must not exceed window_size")
    if task_config.get("task_name") == "multitask_tsad":
        if float(model_config["gumbel_temperature"]) <= 0.0:
            raise ValueError("gumbel_temperature must be positive")
        if float(model_config["temperature_start"]) <= 0.0:
            raise ValueError("temperature_start must be positive")
        if float(model_config["temperature_end"]) <= 0.0:
            raise ValueError("temperature_end must be positive")
        if not 0.0 < float(model_config["temperature_anneal_fraction"]) <= 1.0:
            raise ValueError("temperature_anneal_fraction must be in (0, 1]")
        if not 0.0 <= float(model_config["gate_barrier_margin"]) < 0.5:
            raise ValueError("gate_barrier_margin must be in [0, 0.5)")
        freeze_fusion_for_epochs = task_config.get("freeze_fusion_for_epochs")
        if not isinstance(freeze_fusion_for_epochs, int) or freeze_fusion_for_epochs < 0:
            raise ValueError("freeze_fusion_for_epochs must be a non-negative integer")
        if not 0.0 <= float(task_config["warmup_alpha_value"]) <= 1.0:
            raise ValueError("warmup_alpha_value must be between 0 and 1")
        if not 0.0 <= float(task_config["warmup_beta_value"]) <= 1.0:
            raise ValueError("warmup_beta_value must be between 0 and 1")
        if not 0.0 <= float(task_config["anomaly_probability"]) <= 1.0:
            raise ValueError("anomaly_probability must be between 0 and 1")
        if not 0.0 < float(task_config["min_segment_fraction"]) <= 1.0:
            raise ValueError("min_segment_fraction must be between 0 and 1")
        if not 0.0 < float(task_config["max_segment_fraction"]) <= 1.0:
            raise ValueError("max_segment_fraction must be between 0 and 1")
        if float(task_config["min_segment_fraction"]) > float(task_config["max_segment_fraction"]):
            raise ValueError("min_segment_fraction must not exceed max_segment_fraction")
        anomaly_families = task_config.get("anomaly_families")
        if not isinstance(anomaly_families, list) or not anomaly_families:
            raise ValueError("anomaly_families must be a non-empty list")
        if not all(isinstance(family_name, str) and family_name for family_name in anomaly_families):
            raise ValueError("anomaly_families must contain non-empty strings")
    if task_config.get("task_name") == "online_adaptation":
        if float(model_config["projector_dropout"]) < 0.0:
            raise ValueError("projector_dropout must be non-negative")
        if float(model_config["lambda_align"]) < 0.0:
            raise ValueError("lambda_align must be non-negative")
        if float(model_config["lambda_proto"]) < 0.0:
            raise ValueError("lambda_proto must be non-negative")
        if float(model_config["lambda_anchor"]) < 0.0:
            raise ValueError("lambda_anchor must be non-negative")
        if float(task_config["view_noise_std"]) < 0.0:
            raise ValueError("view_noise_std must be non-negative")
        if not 0.0 <= float(task_config["view_dropout_probability"]) <= 1.0:
            raise ValueError("view_dropout_probability must be between 0 and 1")
        if task_config.get("target_param_group") not in {"projector_params", "online_encoder_params"}:
            raise ValueError("target_param_group must be one of: projector_params, online_encoder_params")
        if task_config.get("reset_policy") not in {"disabled", "threshold"}:
            raise ValueError("reset_policy must be one of: disabled, threshold")

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

    optional_online_integer_fields = [
        "max_online_steps",
        "log_every_n_steps",
        "checkpoint_every_n_steps",
    ]
    if task_config.get("task_name") == "online_adaptation":
        for field_name in optional_online_integer_fields:
            field_value = task_config.get(field_name)
            if not isinstance(field_value, int) or field_value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


def load_experiment_config(experiment_config_path: str | Path) -> dict[str, Any]:
    # The experiment file names the three source YAMLs, then optional override
    # sections can narrow that base into a specific ablation or online run.
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

    resolved_experiment_config["data"] = _merge_config_section(
        resolved_experiment_config["data"],
        root_config.get("data_overrides"),
    )
    resolved_experiment_config["model"] = _merge_config_section(
        resolved_experiment_config["model"],
        root_config.get("model_overrides"),
    )
    resolved_experiment_config["task"] = _merge_config_section(
        resolved_experiment_config["task"],
        root_config.get("task_overrides"),
    )

    validate_experiment_config(resolved_experiment_config)
    return resolved_experiment_config

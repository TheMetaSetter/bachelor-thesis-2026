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

from src.core.console import console_print


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    console_print("CONFIG", "Loading YAML config", path=path)
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
        console_print("CONFIG", "No overrides supplied for config section")
        return dict(base_config)
    if not isinstance(overrides, dict):
        raise ValueError("Config overrides must be mappings")
    merged_config = dict(base_config)
    merged_config.update(overrides)
    console_print(
        "CONFIG",
        "Merged config section overrides",
        override_keys=sorted(overrides.keys()),
    )
    return merged_config


def _resolve_thesis_model_window_size(experiment_config: dict[str, Any]) -> None:
    model_config = experiment_config["model"]
    data_config = experiment_config["data"]
    if model_config.get("model_name") != "thesis_multitask":
        return

    data_window_size = data_config.get("window_size")
    model_window_size = model_config.get("window_size")
    if model_window_size is None:
        model_config["window_size"] = data_window_size
    elif model_window_size != data_window_size:
        raise ValueError("model.window_size must match data.window_size")


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
            raise ValueError(
                f"Experiment config is missing required section: {section_name}"
            )
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

    _resolve_thesis_model_window_size(experiment_config)

    supported_dataset_names = {"smd"}
    supported_model_names = {
        "reconstruction_mlp_ae",
        "thesis_multitask",
        "redlamp_mlp_baseline",
        "online_adaptation",
    }
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
    if model_config.get("model_name") in {
        "reconstruction_mlp_ae",
        "thesis_multitask",
        "online_adaptation",
    }:
        integer_fields["encoder_dim"] = model_config.get("encoder_dim")
        integer_fields["hidden_dim"] = model_config.get("hidden_dim")
    if model_config.get("model_name") == "thesis_multitask":
        integer_fields["window_size"] = model_config.get("window_size")
        integer_fields["mlp_num_linear_layers"] = model_config.get(
            "mlp_num_linear_layers", 3
        )
        integer_fields["num_classes"] = model_config.get("num_classes")
        integer_fields["continuous_num_prototypes"] = model_config.get(
            "continuous_num_prototypes"
        )
        integer_fields["discrete_codebook_size"] = model_config.get(
            "discrete_codebook_size"
        )
        integer_fields["memory_initialization_batches"] = model_config.get(
            "memory_initialization_batches", 16
        )
    if model_config.get("model_name") == "redlamp_mlp_baseline":
        integer_fields["latent_dim"] = model_config.get("latent_dim")
        integer_fields["mlp_num_linear_layers"] = model_config.get(
            "mlp_num_linear_layers", 3
        )
        integer_fields["classifier_dim"] = model_config.get("classifier_dim")
        integer_fields["num_classes"] = model_config.get("num_classes")
    if model_config.get("model_name") == "online_adaptation":
        integer_fields["projector_hidden_dim"] = model_config.get(
            "projector_hidden_dim"
        )
    for field_name, field_value in integer_fields.items():
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    float_fields = {
        "validation_split_ratio": data_config.get("validation_split_ratio"),
        "learning_rate": optimizer_config.get("learning_rate"),
        "weight_decay": optimizer_config.get("weight_decay"),
    }
    if model_config.get("model_name") in {
        "reconstruction_mlp_ae",
        "thesis_multitask",
        "redlamp_mlp_baseline",
    }:
        float_fields["dropout"] = model_config.get("dropout")
    if task_config.get("task_name") == "multitask_tsad":
        if model_config.get("model_name") == "thesis_multitask":
            float_fields["gumbel_temperature"] = model_config.get("gumbel_temperature")
            float_fields["temperature_start"] = model_config.get("temperature_start")
            float_fields["temperature_end"] = model_config.get("temperature_end")
            float_fields["temperature_anneal_fraction"] = model_config.get(
                "temperature_anneal_fraction"
            )
            float_fields["temperature_hold_fraction"] = model_config.get(
                "temperature_hold_fraction", 0.0
            )
            float_fields["alpha_logit_init"] = model_config.get("alpha_logit_init")
            float_fields["beta_logit_init"] = model_config.get("beta_logit_init")
        float_fields["refurbishment_alpha"] = model_config.get(
            "refurbishment_alpha", 0.0
        )
        float_fields["refurbishment_beta"] = model_config.get("refurbishment_beta", 0.0)
        float_fields["lambda_cls"] = model_config.get("lambda_cls")
        if model_config.get("model_name") == "thesis_multitask":
            float_fields["lambda_div"] = model_config.get("lambda_div")
            float_fields["lambda_var"] = model_config.get("lambda_var")
            float_fields["lambda_cov"] = model_config.get("lambda_cov")
            float_fields["lambda_use"] = model_config.get("lambda_use")
            float_fields["lambda_gate"] = model_config.get("lambda_gate")
            float_fields["usage_lambda_start"] = model_config.get(
                "usage_lambda_start", model_config.get("lambda_use")
            )
            float_fields["usage_lambda_end"] = model_config.get(
                "usage_lambda_end", model_config.get("lambda_use")
            )
            float_fields["usage_lambda_schedule_fraction"] = model_config.get(
                "usage_lambda_schedule_fraction", 1.0
            )
            float_fields["variance_floor_gamma"] = model_config.get(
                "variance_floor_gamma"
            )
            float_fields["gate_barrier_margin"] = model_config.get(
                "gate_barrier_margin"
            )
            float_fields["discrete_ema_decay"] = model_config.get(
                "discrete_ema_decay", 0.99
            )
            float_fields["memory_norm_epsilon"] = model_config.get(
                "memory_norm_epsilon", 1.0e-6
            )
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
        float_fields["view_dropout_probability"] = task_config.get(
            "view_dropout_probability"
        )
        float_fields["reset_alignment_threshold"] = task_config.get(
            "reset_alignment_threshold"
        )
    for field_name, field_value in float_fields.items():
        if not isinstance(field_value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")

    optimizer_name = optimizer_config.get("optimizer_name", "adam")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("optimizer.optimizer_name must be one of: adam, adamw")

    gradient_clip_norm = optimizer_config.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        if (
            not isinstance(gradient_clip_norm, (int, float))
            or float(gradient_clip_norm) <= 0.0
        ):
            raise ValueError(
                "optimizer.gradient_clip_norm must be positive when provided"
            )

    checkpoint_monitor_metric = experiment_config.get(
        "checkpoint_monitor_metric",
        "val_loss",
    )
    if checkpoint_monitor_metric not in {
        "val_loss",
        "val_synth_loss",
        "val_synth_roc_auc",
        "val_synth_pr_auc",
        "val_synth_vus_pr",
        "val_vus_pr",
    }:
        raise ValueError(
            "checkpoint_monitor_metric must be one of: val_loss, val_synth_loss, "
            "val_synth_roc_auc, val_synth_pr_auc, val_synth_vus_pr, val_vus_pr"
        )

    scheduler_config = optimizer_config.get("scheduler")
    if scheduler_config is not None:
        if not isinstance(scheduler_config, dict):
            raise ValueError("optimizer.scheduler must be a mapping when provided")
        scheduler_name = scheduler_config.get("scheduler_name")
        if scheduler_name == "reduce_on_plateau":
            incompatible_cosine_fields = {
                "warmup_epochs",
                "warmup_start_lr",
                "cosine_end_lr",
                "cosine_after_warmup",
            }
            if incompatible_cosine_fields.intersection(scheduler_config):
                raise ValueError(
                    "Cosine-only scheduler fields are not valid for reduce_on_plateau"
                )
            monitor_metric = scheduler_config.get("monitor_metric")
            if monitor_metric not in {
                "val_loss",
                "val_synth_loss",
                "val_synth_roc_auc",
                "val_synth_pr_auc",
                "val_synth_vus_pr",
            }:
                raise ValueError(
                    "optimizer.scheduler.monitor_metric must be one of: val_loss, val_synth_loss, val_synth_roc_auc, val_synth_pr_auc, val_synth_vus_pr"
                )
            scheduler_factor = scheduler_config.get("factor")
            if (
                not isinstance(scheduler_factor, (int, float))
                or not 0.0 < float(scheduler_factor) < 1.0
            ):
                raise ValueError("optimizer.scheduler.factor must be in (0, 1)")
            scheduler_patience = scheduler_config.get("patience")
            if not isinstance(scheduler_patience, int) or scheduler_patience < 0:
                raise ValueError(
                    "optimizer.scheduler.patience must be a non-negative integer"
                )
            scheduler_threshold = scheduler_config.get("threshold")
            if (
                not isinstance(scheduler_threshold, (int, float))
                or float(scheduler_threshold) < 0.0
            ):
                raise ValueError("optimizer.scheduler.threshold must be non-negative")
            scheduler_threshold_mode = scheduler_config.get("threshold_mode")
            if scheduler_threshold_mode not in {"rel", "abs"}:
                raise ValueError(
                    "optimizer.scheduler.threshold_mode must be one of: rel, abs"
                )
            scheduler_cooldown = scheduler_config.get("cooldown")
            if not isinstance(scheduler_cooldown, int) or scheduler_cooldown < 0:
                raise ValueError(
                    "optimizer.scheduler.cooldown must be a non-negative integer"
                )
            scheduler_min_lr = scheduler_config.get("min_lr")
            if (
                not isinstance(scheduler_min_lr, (int, float))
                or float(scheduler_min_lr) <= 0.0
            ):
                raise ValueError("optimizer.scheduler.min_lr must be positive")
            if float(scheduler_min_lr) > float(optimizer_config["learning_rate"]):
                raise ValueError(
                    "optimizer.scheduler.min_lr must not exceed optimizer.learning_rate"
                )
        elif scheduler_name == "cosine":
            incompatible_plateau_fields = {
                "monitor_metric",
                "factor",
                "patience",
                "threshold",
                "threshold_mode",
                "cooldown",
                "min_lr",
            }
            if incompatible_plateau_fields.intersection(scheduler_config):
                raise ValueError(
                    "Plateau-only scheduler fields are not valid for cosine"
                )
            warmup_epochs = scheduler_config.get("warmup_epochs")
            if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
                raise ValueError(
                    "optimizer.scheduler.warmup_epochs must be a non-negative integer"
                )
            warmup_start_lr = scheduler_config.get("warmup_start_lr")
            if (
                not isinstance(warmup_start_lr, (int, float))
                or float(warmup_start_lr) <= 0.0
                or float(warmup_start_lr) > float(optimizer_config["learning_rate"])
            ):
                raise ValueError(
                    "optimizer.scheduler.warmup_start_lr must be positive and not exceed optimizer.learning_rate"
                )
            cosine_end_lr = scheduler_config.get("cosine_end_lr")
            if (
                not isinstance(cosine_end_lr, (int, float))
                or float(cosine_end_lr) < 0.0
                or float(cosine_end_lr) >= float(optimizer_config["learning_rate"])
            ):
                raise ValueError(
                    "optimizer.scheduler.cosine_end_lr must be non-negative and lower than optimizer.learning_rate"
                )
            if not isinstance(scheduler_config.get("cosine_after_warmup"), bool):
                raise ValueError(
                    "optimizer.scheduler.cosine_after_warmup must be a boolean"
                )
        else:
            raise ValueError(
                "optimizer.scheduler.scheduler_name must be one of: reduce_on_plateau, cosine"
            )

    if task_config.get("task_name") == "multitask_tsad":
        boolean_fields = {
            "enable_diversity_loss": model_config.get("enable_diversity_loss", False),
            "enable_variance_loss": model_config.get("enable_variance_loss", False),
            "enable_covariance_loss": model_config.get("enable_covariance_loss", False),
            "enable_usage_loss": model_config.get("enable_usage_loss", False),
            "enable_gate_loss": model_config.get("enable_gate_loss", False),
            "use_label_refurbishment": model_config.get(
                "use_label_refurbishment", False
            ),
            "reconstruction_normal_only": model_config.get(
                "reconstruction_normal_only", False
            ),
            "memory_initialization_with_synthetic_windows": model_config.get(
                "memory_initialization_with_synthetic_windows", True
            ),
            "use_synthetic_augmentation": task_config.get("use_synthetic_augmentation"),
            "use_synthetic_validation": task_config.get(
                "use_synthetic_validation", True
            ),
            "balance_binary_classes_within_batch": task_config.get(
                "balance_binary_classes_within_batch", False
            ),
        }
        if model_config.get("model_name") == "redlamp_mlp_baseline":
            for model_only_field in [
                "enable_diversity_loss",
                "enable_variance_loss",
                "enable_covariance_loss",
                "enable_usage_loss",
                "enable_gate_loss",
                "memory_initialization_with_synthetic_windows",
                "reconstruction_normal_only",
            ]:
                boolean_fields.pop(model_only_field, None)
        for field_name, field_value in boolean_fields.items():
            if not isinstance(field_value, bool):
                raise ValueError(f"{field_name} must be a boolean")
    if task_config.get("task_name") == "online_adaptation":
        boolean_fields = {
            "enable_prototype_alignment": model_config.get(
                "enable_prototype_alignment"
            ),
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
    optional_data_boolean_fields = {
        "download": data_config.get("download", False),
        "skip_existing_download": data_config.get("skip_existing_download", True),
        "annotate_cleaning_metadata": data_config.get(
            "annotate_cleaning_metadata", False
        ),
    }
    for field_name, field_value in optional_data_boolean_fields.items():
        if not isinstance(field_value, bool):
            raise ValueError(f"data.{field_name} must be a boolean when provided")
    num_workers_value = data_config.get("num_workers", 0)
    if isinstance(num_workers_value, str):
        if num_workers_value != "auto":
            raise ValueError(
                "data.num_workers must be a non-negative integer or 'auto'"
            )
    elif not isinstance(num_workers_value, int) or num_workers_value < 0:
        raise ValueError("data.num_workers must be a non-negative integer or 'auto'")
    min_num_workers_value = data_config.get("min_num_workers")
    if min_num_workers_value is not None:
        if not isinstance(min_num_workers_value, int) or min_num_workers_value <= 0:
            raise ValueError(
                "data.min_num_workers must be a positive integer when provided"
            )
    entity_ids = data_config.get("entity_ids")
    if entity_ids is not None:
        if not isinstance(entity_ids, list) or not entity_ids:
            raise ValueError(
                "data.entity_ids must be a non-empty list of strings when provided"
            )
        if not all(
            isinstance(entity_id, str) and entity_id for entity_id in entity_ids
        ):
            raise ValueError("data.entity_ids must contain non-empty strings")
    if task_config.get("task_name") == "multitask_tsad":
        if int(model_config.get("mlp_num_linear_layers", 3)) < 2:
            raise ValueError("mlp_num_linear_layers must be at least 2")
        if model_config.get("model_name") == "thesis_multitask":
            if float(model_config["gumbel_temperature"]) <= 0.0:
                raise ValueError("gumbel_temperature must be positive")
            if float(model_config["temperature_start"]) <= 0.0:
                raise ValueError("temperature_start must be positive")
            if float(model_config["temperature_end"]) <= 0.0:
                raise ValueError("temperature_end must be positive")
            if not 0.0 < float(model_config["temperature_anneal_fraction"]) <= 1.0:
                raise ValueError("temperature_anneal_fraction must be in (0, 1]")
            if (
                not 0.0
                <= float(model_config.get("temperature_hold_fraction", 0.0))
                < 1.0
            ):
                raise ValueError("temperature_hold_fraction must be in [0, 1)")
        if not 0.0 <= float(model_config.get("refurbishment_alpha", 0.0)) <= 1.0:
            raise ValueError("refurbishment_alpha must be in [0, 1]")
        if not 0.0 <= float(model_config.get("refurbishment_beta", 0.0)) <= 1.0:
            raise ValueError("refurbishment_beta must be in [0, 1]")
        classification_label_mode = task_config.get(
            "classification_label_mode", "binary"
        )
        if classification_label_mode not in {"binary", "redlamp_multiclass"}:
            raise ValueError(
                "classification_label_mode must be one of: binary, redlamp_multiclass"
            )
        if (
            classification_label_mode == "redlamp_multiclass"
            and int(model_config["num_classes"]) != 12
        ):
            raise ValueError(
                "classification_label_mode='redlamp_multiclass' requires num_classes == 12"
            )
        if model_config.get("model_name") == "thesis_multitask":
            bootstrap_encoder_epochs = model_config.get("bootstrap_encoder_epochs", 10)
            if (
                not isinstance(bootstrap_encoder_epochs, int)
                or isinstance(bootstrap_encoder_epochs, bool)
                or bootstrap_encoder_epochs < 0
            ):
                raise ValueError(
                    "bootstrap_encoder_epochs must be a non-negative integer"
                )
            if not 0.0 < float(model_config.get("discrete_ema_decay", 0.99)) < 1.0:
                raise ValueError("discrete_ema_decay must be in (0, 1)")
            if float(model_config.get("memory_norm_epsilon", 1.0e-6)) <= 0.0:
                raise ValueError("memory_norm_epsilon must be positive")
            if (
                float(
                    model_config.get("usage_lambda_start", model_config["lambda_use"])
                )
                < 0.0
            ):
                raise ValueError("usage_lambda_start must be non-negative")
            if (
                float(model_config.get("usage_lambda_end", model_config["lambda_use"]))
                < 0.0
            ):
                raise ValueError("usage_lambda_end must be non-negative")
            if (
                not 0.0
                < float(model_config.get("usage_lambda_schedule_fraction", 1.0))
                <= 1.0
            ):
                raise ValueError("usage_lambda_schedule_fraction must be in (0, 1]")
            if not 0.0 <= float(model_config["gate_barrier_margin"]) < 0.5:
                raise ValueError("gate_barrier_margin must be in [0, 0.5)")
        freeze_fusion_for_epochs = task_config.get("freeze_fusion_for_epochs")
        if (
            not isinstance(freeze_fusion_for_epochs, int)
            or freeze_fusion_for_epochs < 0
        ):
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
        if float(task_config["min_segment_fraction"]) > float(
            task_config["max_segment_fraction"]
        ):
            raise ValueError(
                "min_segment_fraction must not exceed max_segment_fraction"
            )
        anomaly_families = task_config.get("anomaly_families")
        if not isinstance(anomaly_families, list) or not anomaly_families:
            raise ValueError("anomaly_families must be a non-empty list")
        if not all(
            isinstance(family_name, str) and family_name
            for family_name in anomaly_families
        ):
            raise ValueError("anomaly_families must contain non-empty strings")
        synthetic_validation_seed = task_config.get("synthetic_validation_seed", 7)
        if (
            not isinstance(synthetic_validation_seed, int)
            or synthetic_validation_seed < 0
        ):
            raise ValueError("synthetic_validation_seed must be a non-negative integer")
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
        if task_config.get("target_param_group") not in {
            "projector_params",
            "online_encoder_params",
        }:
            raise ValueError(
                "target_param_group must be one of: projector_params, online_encoder_params"
            )
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

    logging_config = experiment_config.get("logging")
    if logging_config is not None:
        if not isinstance(logging_config, dict):
            raise ValueError("logging must be a mapping when provided")
        use_wandb = logging_config.get("use_wandb")
        if use_wandb is not None and not isinstance(use_wandb, bool):
            raise ValueError("logging.use_wandb must be a boolean when provided")
        if "wandb_project" in logging_config and not isinstance(
            logging_config["wandb_project"], str
        ):
            raise ValueError("logging.wandb_project must be a string when provided")
        if (
            "wandb_entity" in logging_config
            and logging_config["wandb_entity"] is not None
            and not isinstance(logging_config["wandb_entity"], str)
        ):
            raise ValueError("logging.wandb_entity must be a string or null")
        if "wandb_mode" in logging_config:
            if logging_config["wandb_mode"] not in {"online", "offline", "disabled"}:
                raise ValueError(
                    "logging.wandb_mode must be one of: online, offline, disabled"
                )
        if (
            "wandb_run_name" in logging_config
            and logging_config["wandb_run_name"] is not None
            and not isinstance(logging_config["wandb_run_name"], str)
        ):
            raise ValueError("logging.wandb_run_name must be a string or null")
        if (
            "wandb_job_type" in logging_config
            and logging_config["wandb_job_type"] is not None
            and not isinstance(logging_config["wandb_job_type"], str)
        ):
            raise ValueError("logging.wandb_job_type must be a string or null")
        if "wandb_tags" in logging_config:
            wandb_tags = logging_config["wandb_tags"]
            if wandb_tags is not None:
                if not isinstance(wandb_tags, list) or not all(
                    isinstance(tag, str) and tag for tag in wandb_tags
                ):
                    raise ValueError(
                        "logging.wandb_tags must be a list of non-empty strings or null"
                    )
        kaggle_boolean_fields = {
            "mirror_best_checkpoint_to_kaggle": logging_config.get(
                "mirror_best_checkpoint_to_kaggle", False
            ),
            "mirror_output_dir_to_kaggle": logging_config.get(
                "mirror_output_dir_to_kaggle", False
            ),
        }
        for field_name, field_value in kaggle_boolean_fields.items():
            if not isinstance(field_value, bool):
                raise ValueError(
                    f"logging.{field_name} must be a boolean when provided"
                )
        kaggle_dataset_handle = logging_config.get("kaggle_dataset_handle")
        if kaggle_dataset_handle is not None and not isinstance(
            kaggle_dataset_handle, str
        ):
            raise ValueError("logging.kaggle_dataset_handle must be a string or null")
        kaggle_version_notes = logging_config.get("kaggle_version_notes")
        if kaggle_version_notes is not None and not isinstance(
            kaggle_version_notes, str
        ):
            raise ValueError("logging.kaggle_version_notes must be a string or null")
        if (
            logging_config.get("mirror_best_checkpoint_to_kaggle", False)
            or logging_config.get("mirror_output_dir_to_kaggle", False)
        ) and not kaggle_dataset_handle:
            raise ValueError(
                "logging.kaggle_dataset_handle must be provided when Kaggle mirroring is enabled"
            )


def load_experiment_config(experiment_config_path: str | Path) -> dict[str, Any]:
    # The experiment file names the three source YAMLs, then optional override
    # sections can narrow that base into a specific ablation or online run.
    experiment_path = Path(experiment_config_path)
    console_print(
        "CONFIG", "Loading experiment config", experiment_config_path=experiment_path
    )
    root_config = load_yaml_config(experiment_path)

    required_reference_fields = [
        "data_config_path",
        "model_config_path",
        "task_config_path",
    ]
    for reference_field in required_reference_fields:
        if reference_field not in root_config:
            raise ValueError(
                f"Experiment config is missing file reference: {reference_field}"
            )

    resolved_experiment_config = dict(root_config)
    for section_name, reference_field in [
        ("data", "data_config_path"),
        ("model", "model_config_path"),
        ("task", "task_config_path"),
    ]:
        config_reference = Path(root_config[reference_field])
        if not config_reference.is_absolute():
            config_reference = (
                experiment_path.parent.parent / config_reference.relative_to("configs")
            )
        console_print(
            "CONFIG",
            "Resolving referenced config",
            section=section_name,
            reference_field=reference_field,
            resolved_path=config_reference,
        )
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

    _resolve_thesis_model_window_size(resolved_experiment_config)
    validate_experiment_config(resolved_experiment_config)
    console_print(
        "CONFIG",
        "Resolved experiment config",
        experiment_name=resolved_experiment_config["experiment_name"],
        dataset_name=resolved_experiment_config["data"]["dataset_name"],
        model_name=resolved_experiment_config["model"]["model_name"],
        task_name=resolved_experiment_config["task"]["task_name"],
        output_dir=resolved_experiment_config["output_dir"],
        checkpoint_dir=resolved_experiment_config["checkpoint_dir"],
    )
    return resolved_experiment_config

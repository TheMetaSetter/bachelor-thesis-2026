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
from src.core.config_experiment_validation import validate_experiment_config


STAGE3_WARMUP_EPOCHS_LEGACY_KEY = "stage3_prototype_warmup_epochs"
STAGE3_WARMUP_EPOCHS_CANONICAL_KEY = (
    "stage3_memory_initialization_and_fusion_warmup_epochs"
)
TWO_STAGE_A_EPOCHS_KEY = "stage_a_multitask_epochs"
TWO_STAGE_B_EPOCHS_KEY = "stage_b_fusion_finetuning_epochs"


class _UniqueKeyYamlLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_mapping_with_unique_keys(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in YAML mapping: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_with_unique_keys,
)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    console_print("CONFIG", "Loading YAML config", path=path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded_config = yaml.load(handle, Loader=_UniqueKeyYamlLoader) or {}

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


def _validate_non_negative_integer_fields(
    *,
    config_name: str,
    config: dict[str, Any],
    field_names: list[str],
) -> None:
    for field_name in field_names:
        field_value = config.get(field_name)
        if not isinstance(field_value, int) or field_value < 0:
            raise ValueError(
                f"{config_name}.{field_name} must be a non-negative integer"
            )


def _validate_boolean_fields(
    *,
    config_name: str,
    config: dict[str, Any],
    field_names: list[str],
) -> None:
    for field_name in field_names:
        field_value = config.get(field_name)
        if not isinstance(field_value, bool):
            raise ValueError(f"{config_name}.{field_name} must be a boolean")


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


def _normalize_alias_with_compatibility(
    config: dict[str, Any],
    *,
    new_key: str,
    legacy_keys: tuple[str, ...],
) -> None:
    legacy_values = [config[key] for key in legacy_keys if key in config]
    if not legacy_values:
        return
    resolved_value = legacy_values[0]
    if new_key in config and config[new_key] != resolved_value:
        raise ValueError(
            f"Config alias mismatch for {new_key}: {config[new_key]} != {resolved_value}"
        )
    config[new_key] = resolved_value
    for legacy_key in legacy_keys:
        if legacy_key in config and config[legacy_key] != resolved_value:
            raise ValueError(
                f"Config alias mismatch for {legacy_key}: {config[legacy_key]} != {resolved_value}"
            )


def _normalize_stage_metadata_aliases(experiment_config: dict[str, Any]) -> None:
    _normalize_alias_with_compatibility(
        experiment_config,
        new_key="stage_name",
        legacy_keys=("two_stage_phase", "three_stage_phase"),
    )
    _normalize_alias_with_compatibility(
        experiment_config,
        new_key="stage_global_epoch_start",
        legacy_keys=("two_stage_global_epoch_start", "three_stage_global_epoch_start"),
    )
    _normalize_alias_with_compatibility(
        experiment_config,
        new_key="stage_global_epoch_end",
        legacy_keys=("two_stage_global_epoch_end", "three_stage_global_epoch_end"),
    )
    model_config = experiment_config.get("model")
    if isinstance(model_config, dict):
        _normalize_alias_with_compatibility(
            model_config,
            new_key="stage_name",
            legacy_keys=("training_phase",),
        )


def _normalize_thesis_multitask_v3_aliases(experiment_config: dict[str, Any]) -> None:
    model_config = experiment_config.get("model")
    if not isinstance(model_config, dict):
        return

    def _normalize_variance_correction_value(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int) and value in {0, 1}:
            return value
        if isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value in {"unbiased", "sample", "sample_unbiased"}:
                return 1
            if normalized_value in {"population", "biased", "none"}:
                return 0
        raise ValueError(
            "variance_correction must be 0, 1, or one of: unbiased, sample, population"
        )

    if (
        "sample_variance_correction" in model_config
        and "variance_correction" in model_config
    ):
        if (
            model_config["sample_variance_correction"]
            != model_config["variance_correction"]
        ):
            raise ValueError(
                "Config alias mismatch for variance_correction and sample_variance_correction"
            )
    if (
        "variance_correction" not in model_config
        and "sample_variance_correction" in model_config
    ):
        model_config["variance_correction"] = model_config["sample_variance_correction"]
    if "variance_correction" in model_config:
        model_config["variance_correction"] = _normalize_variance_correction_value(
            model_config["variance_correction"]
        )
    model_config.pop("sample_variance_correction", None)


def _normalize_three_stage_config_keys(three_stage_config: dict[str, Any]) -> None:
    # Legacy three-stage compatibility remains read-supported only.
    # The active two-stage rerun should be interpreted from `two_stage`.
    has_legacy_key = STAGE3_WARMUP_EPOCHS_LEGACY_KEY in three_stage_config
    has_canonical_key = STAGE3_WARMUP_EPOCHS_CANONICAL_KEY in three_stage_config
    if has_legacy_key and has_canonical_key:
        if (
            three_stage_config[STAGE3_WARMUP_EPOCHS_LEGACY_KEY]
            != three_stage_config[STAGE3_WARMUP_EPOCHS_CANONICAL_KEY]
        ):
            raise ValueError(
                "three_stage stage3 warm-up epoch keys disagree: "
                f"{STAGE3_WARMUP_EPOCHS_LEGACY_KEY} vs "
                f"{STAGE3_WARMUP_EPOCHS_CANONICAL_KEY}"
            )
    if has_canonical_key:
        three_stage_config.pop(STAGE3_WARMUP_EPOCHS_LEGACY_KEY, None)
        return
    if has_legacy_key:
        three_stage_config[STAGE3_WARMUP_EPOCHS_CANONICAL_KEY] = three_stage_config[
            STAGE3_WARMUP_EPOCHS_LEGACY_KEY
        ]
        return
    raise ValueError(
        "three_stage config must define "
        f"{STAGE3_WARMUP_EPOCHS_CANONICAL_KEY} "
        f"(legacy alias: {STAGE3_WARMUP_EPOCHS_LEGACY_KEY})"
    )


def _validate_three_stage_config(three_stage_config: dict[str, Any]) -> None:
    _normalize_three_stage_config_keys(three_stage_config)
    # This validator is retained for the historical three-stage path.
    # The active rerun uses `_validate_two_stage_config`.
    allowed_three_stage_keys = {
        "expected_total_training_epochs",
        "stage1_classification_epochs",
        "stage1_reconstruction_epochs",
        "stage2_recovery_epochs",
        "stage3_prototype_warmup_epochs",
        "stage3_memory_initialization_and_fusion_warmup_epochs",
        "multitask_pretraining_epochs",
        "discrete_memory_label_source",
        "freeze_memories_after_initialization",
        "freeze_recovered_zipped_encoder_during_warmup",
    }
    unknown_three_stage_keys = sorted(
        set(three_stage_config) - allowed_three_stage_keys
    )
    if unknown_three_stage_keys:
        raise ValueError(
            "Unknown three_stage config keys: "
            f"{unknown_three_stage_keys}. Remove these keys from three_stage config."
        )

    _validate_non_negative_integer_fields(
        config_name="three_stage",
        config=three_stage_config,
        field_names=[
            "expected_total_training_epochs",
            "stage1_classification_epochs",
            "stage1_reconstruction_epochs",
            "stage2_recovery_epochs",
            STAGE3_WARMUP_EPOCHS_CANONICAL_KEY,
            "multitask_pretraining_epochs",
        ],
    )
    _validate_boolean_fields(
        config_name="three_stage",
        config=three_stage_config,
        field_names=[
            "freeze_memories_after_initialization",
            "freeze_recovered_zipped_encoder_during_warmup",
        ],
    )

    discrete_memory_label_source = three_stage_config.get(
        "discrete_memory_label_source"
    )
    if discrete_memory_label_source != "synthetic_train_labels":
        raise ValueError(
            "three_stage.discrete_memory_label_source must be 'synthetic_train_labels'"
        )

    computed_total_training_epochs = (
        three_stage_config["stage1_classification_epochs"]
        + three_stage_config["stage1_reconstruction_epochs"]
        + three_stage_config["stage2_recovery_epochs"]
        + three_stage_config[STAGE3_WARMUP_EPOCHS_CANONICAL_KEY]
        + three_stage_config["multitask_pretraining_epochs"]
    )
    if (
        computed_total_training_epochs
        != three_stage_config["expected_total_training_epochs"]
    ):
        raise ValueError(
            "three_stage training epochs must sum to expected_total_training_epochs. "
            f"Got total={computed_total_training_epochs}, "
            "expected_total_training_epochs="
            f"{three_stage_config['expected_total_training_epochs']}."
        )


def _validate_two_stage_config(two_stage_config: dict[str, Any]) -> None:
    # This validator defines the active rerun contract.
    allowed_two_stage_keys = {
        "expected_total_training_epochs",
        TWO_STAGE_A_EPOCHS_KEY,
        TWO_STAGE_B_EPOCHS_KEY,
        "discrete_memory_label_source",
        "freeze_encoder_and_memories_in_stage_b",
    }
    unknown_two_stage_keys = sorted(set(two_stage_config) - allowed_two_stage_keys)
    if unknown_two_stage_keys:
        raise ValueError(
            "Unknown two_stage config keys: "
            f"{unknown_two_stage_keys}. Remove these keys from two_stage config."
        )

    _validate_non_negative_integer_fields(
        config_name="two_stage",
        config=two_stage_config,
        field_names=[
            "expected_total_training_epochs",
            TWO_STAGE_A_EPOCHS_KEY,
            TWO_STAGE_B_EPOCHS_KEY,
        ],
    )
    _validate_boolean_fields(
        config_name="two_stage",
        config=two_stage_config,
        field_names=["freeze_encoder_and_memories_in_stage_b"],
    )

    if two_stage_config.get("discrete_memory_label_source") != "synthetic_train_labels":
        raise ValueError(
            "two_stage.discrete_memory_label_source must be 'synthetic_train_labels'"
        )

    computed_total_training_epochs = (
        two_stage_config[TWO_STAGE_A_EPOCHS_KEY]
        + two_stage_config[TWO_STAGE_B_EPOCHS_KEY]
    )
    if (
        computed_total_training_epochs
        != two_stage_config["expected_total_training_epochs"]
    ):
        raise ValueError(
            "two_stage training epochs must sum to expected_total_training_epochs. "
            f"Got total={computed_total_training_epochs}, "
            "expected_total_training_epochs="
            f"{two_stage_config['expected_total_training_epochs']}."
        )


def _validate_experiment_top_level_structure(
    experiment_config: dict[str, Any],
) -> list[str]:
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
    allowed_top_level_keys = {
        *required_sections,
        "data_config_path",
        "model_config_path",
        "task_config_path",
        "data_overrides",
        "model_overrides",
        "task_overrides",
        "evaluation",
        "logging",
        "checkpoint_monitor_metric",
        "experiment_variant",
        "three_stage",
        "two_stage",
        "three_stage_phase",
        "three_stage_global_epoch_start",
        "three_stage_global_epoch_end",
        "two_stage_phase",
        "two_stage_global_epoch_start",
        "two_stage_global_epoch_end",
        "stage_name",
        "stage_global_epoch_start",
        "stage_global_epoch_end",
        "initialization_checkpoint_path",
    }
    unknown_top_level_keys = sorted(set(experiment_config) - allowed_top_level_keys)
    if unknown_top_level_keys:
        raise ValueError(
            "Unknown top-level config keys: "
            f"{unknown_top_level_keys}. Remove these keys from the experiment YAML."
        )
    return required_sections


def _validate_data_config(data_config: dict[str, Any]) -> None:
    allowed_data_keys = {
        "dataset_name",
        "root_dir",
        "file_path",
        "window_size",
        "stride",
        "train_stride",
        "val_stride",
        "test_stride",
        "batch_size",
        "num_workers",
        "min_num_workers",
        "validation_split_ratio",
        "download",
        "skip_existing_download",
        "annotate_cleaning_metadata",
        "entity_ids",
        "shuffle_train",
        "max_train_windows",
        "max_val_windows",
        "max_test_windows",
    }
    unknown_data_keys = sorted(set(data_config) - allowed_data_keys)
    if unknown_data_keys:
        raise ValueError(
            "Unknown data config keys: "
            f"{unknown_data_keys}. Remove these keys from data config."
        )
    if data_config.get("dataset_name") not in {"smd", "anomaly_archive"}:
        raise ValueError(f"Unsupported dataset_name: {data_config.get('dataset_name')}")
    if data_config.get("dataset_name") == "anomaly_archive" and not data_config.get(
        "file_path"
    ):
        raise ValueError("anomaly_archive data config requires file_path")
    if not 0.0 < float(data_config["validation_split_ratio"]) < 1.0:
        raise ValueError("validation_split_ratio must be between 0 and 1")
    stride_field_names = ["stride", "train_stride", "val_stride", "test_stride"]
    for field_name in stride_field_names:
        field_value = data_config.get(field_name)
        if field_value is None:
            continue
        if field_value > data_config["window_size"]:
            raise ValueError(f"{field_name} must not exceed window_size")
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


def _validate_optimizer_config(
    optimizer_config: dict[str, Any],
    *,
    checkpoint_monitor_metric: str,
) -> None:
    optimizer_name = optimizer_config.get("optimizer_name", "adam")
    allowed_optimizer_keys = {
        "optimizer_name",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "scheduler",
    }
    unknown_optimizer_keys = sorted(set(optimizer_config) - allowed_optimizer_keys)
    if unknown_optimizer_keys:
        raise ValueError(
            "Unknown optimizer config keys: "
            f"{unknown_optimizer_keys}. Remove these keys from optimizer config."
        )
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

    checkpoint_monitor_metric_values = {
        "val_loss",
        "val_synth_loss",
        "val_synth_roc_auc",
        "val_synth_pr_auc",
        "val_synth_vus_pr",
        "val_vus_pr",
    }
    if checkpoint_monitor_metric not in checkpoint_monitor_metric_values:
        raise ValueError(
            "checkpoint_monitor_metric must be one of: val_loss, "
            "val_synth_loss, val_synth_roc_auc, val_synth_pr_auc, "
            "val_synth_vus_pr, val_vus_pr"
        )

    scheduler_config = optimizer_config.get("scheduler")
    if scheduler_config is None:
        return
    if not isinstance(scheduler_config, dict):
        raise ValueError("optimizer.scheduler must be a mapping when provided")

    scheduler_name = scheduler_config.get("scheduler_name")
    if scheduler_name == "reduce_on_plateau":
        allowed_scheduler_keys = {
            "scheduler_name",
            "monitor_metric",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "min_lr",
        }
        unknown_scheduler_keys = sorted(set(scheduler_config) - allowed_scheduler_keys)
        if unknown_scheduler_keys:
            raise ValueError(
                "Unknown optimizer.scheduler keys for reduce_on_plateau: "
                f"{unknown_scheduler_keys}. Remove these keys from optimizer.scheduler."
            )
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
        if checkpoint_monitor_metric != monitor_metric:
            raise ValueError(
                "Config contradiction: optimizer.scheduler.monitor_metric and "
                "checkpoint_monitor_metric must match for reduce_on_plateau. "
                f"Current values: monitor_metric='{monitor_metric}', "
                f"checkpoint_monitor_metric='{checkpoint_monitor_metric}'. "
                "Fix by setting checkpoint_monitor_metric to the same metric, "
                "or update optimizer.scheduler.monitor_metric accordingly."
            )
        if monitor_metric not in {
            "val_loss",
            "val_synth_loss",
            "val_synth_roc_auc",
            "val_synth_pr_auc",
            "val_synth_vus_pr",
        }:
            raise ValueError(
                "optimizer.scheduler.monitor_metric must be one of: val_loss, "
                "val_synth_loss, val_synth_roc_auc, val_synth_pr_auc, "
                "val_synth_vus_pr"
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
        return

    if scheduler_name == "cosine":
        allowed_scheduler_keys = {
            "scheduler_name",
            "warmup_epochs",
            "warmup_start_lr",
            "cosine_end_lr",
            "cosine_after_warmup",
        }
        unknown_scheduler_keys = sorted(set(scheduler_config) - allowed_scheduler_keys)
        if unknown_scheduler_keys:
            raise ValueError(
                "Unknown optimizer.scheduler keys for cosine: "
                f"{unknown_scheduler_keys}. Remove these keys from optimizer.scheduler."
            )
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
            raise ValueError("Plateau-only scheduler fields are not valid for cosine")
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
        return

    raise ValueError(
        "optimizer.scheduler.scheduler_name must be one of: reduce_on_plateau, cosine"
    )


def _validate_logging_config(logging_config: dict[str, Any]) -> None:
    allowed_logging_keys = {
        "use_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_mode",
        "wandb_run_name",
        "wandb_job_type",
        "wandb_tags",
        "mirror_best_checkpoint_to_kaggle",
        "mirror_output_dir_to_kaggle",
        "kaggle_dataset_handle",
        "kaggle_version_notes",
        "enable_reconstruction_diagnostics",
        "diagnostics_log_interval_steps",
        "diagnostics_include_grad_norm",
        "log_hard_prediction_ratio",
        "log_row_normalized_confusion_matrix",
        "log_focused_metrics_jsonl",
        "quiet_terminal",
        "diagnostics_stages_for_classification",
        "focus_metrics",
        "focused_metrics_filename",
    }
    unknown_logging_keys = sorted(set(logging_config) - allowed_logging_keys)
    if unknown_logging_keys:
        raise ValueError(
            "Unknown logging config keys: "
            f"{unknown_logging_keys}. Remove these keys from logging config."
        )
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
    resolved_use_wandb = bool(logging_config.get("use_wandb", False))
    resolved_wandb_mode = logging_config.get("wandb_mode", "disabled")
    if not resolved_use_wandb and resolved_wandb_mode != "disabled":
        raise ValueError(
            "Config contradiction: logging.use_wandb is false but "
            f"logging.wandb_mode is '{resolved_wandb_mode}'. "
            "Fix by setting logging.wandb_mode to 'disabled', or set "
            "logging.use_wandb to true if you want wandb logging."
        )
    if resolved_use_wandb and resolved_wandb_mode == "disabled":
        raise ValueError(
            "Config contradiction: logging.use_wandb is true but "
            "logging.wandb_mode is 'disabled'. "
            "Fix by setting logging.wandb_mode to 'online' or 'offline', "
            "or set logging.use_wandb to false."
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
            raise ValueError(f"logging.{field_name} must be a boolean when provided")
    kaggle_dataset_handle = logging_config.get("kaggle_dataset_handle")
    if kaggle_dataset_handle is not None and not isinstance(kaggle_dataset_handle, str):
        raise ValueError("logging.kaggle_dataset_handle must be a string or null")
    kaggle_version_notes = logging_config.get("kaggle_version_notes")
    if kaggle_version_notes is not None and not isinstance(kaggle_version_notes, str):
        raise ValueError("logging.kaggle_version_notes must be a string or null")
    if (
        logging_config.get("mirror_best_checkpoint_to_kaggle", False)
        or logging_config.get("mirror_output_dir_to_kaggle", False)
    ) and not kaggle_dataset_handle:
        raise ValueError(
            "logging.kaggle_dataset_handle must be provided when Kaggle mirroring is enabled"
        )
    reconstruction_diagnostics_enabled = logging_config.get(
        "enable_reconstruction_diagnostics"
    )
    if reconstruction_diagnostics_enabled is not None and not isinstance(
        reconstruction_diagnostics_enabled, bool
    ):
        raise ValueError(
            "logging.enable_reconstruction_diagnostics must be a boolean when provided"
        )
    diagnostics_log_interval_steps = logging_config.get(
        "diagnostics_log_interval_steps"
    )
    if diagnostics_log_interval_steps is not None:
        if (
            not isinstance(diagnostics_log_interval_steps, int)
            or diagnostics_log_interval_steps <= 0
        ):
            raise ValueError(
                "logging.diagnostics_log_interval_steps must be a positive integer when provided"
            )
    diagnostics_include_grad_norm = logging_config.get("diagnostics_include_grad_norm")
    if diagnostics_include_grad_norm is not None and not isinstance(
        diagnostics_include_grad_norm, bool
    ):
        raise ValueError(
            "logging.diagnostics_include_grad_norm must be a boolean when provided"
        )
    for field_name in [
        "log_hard_prediction_ratio",
        "log_row_normalized_confusion_matrix",
        "log_focused_metrics_jsonl",
        "quiet_terminal",
    ]:
        field_value = logging_config.get(field_name)
        if field_value is not None and not isinstance(field_value, bool):
            raise ValueError(f"logging.{field_name} must be a boolean when provided")
    diagnostics_stages_for_classification = logging_config.get(
        "diagnostics_stages_for_classification"
    )
    if diagnostics_stages_for_classification is not None:
        if not isinstance(diagnostics_stages_for_classification, list) or not all(
            isinstance(stage_name, str)
            for stage_name in diagnostics_stages_for_classification
        ):
            raise ValueError(
                "logging.diagnostics_stages_for_classification must be a list of strings when provided"
            )
        allowed_stages = {"train", "val", "val_synth", "test"}
        invalid_stage_names = sorted(
            set(diagnostics_stages_for_classification) - allowed_stages
        )
        if invalid_stage_names:
            raise ValueError(
                "logging.diagnostics_stages_for_classification contains unsupported stages: "
                f"{invalid_stage_names}"
            )
    focus_metrics = logging_config.get("focus_metrics")
    if focus_metrics is not None:
        if not isinstance(focus_metrics, list) or not all(
            isinstance(metric_name, str) and metric_name
            for metric_name in focus_metrics
        ):
            raise ValueError(
                "logging.focus_metrics must be a list of non-empty strings when provided"
            )
    focused_metrics_filename = logging_config.get("focused_metrics_filename")
    if focused_metrics_filename is not None and not isinstance(
        focused_metrics_filename, str
    ):
        raise ValueError(
            "logging.focused_metrics_filename must be a string when provided"
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
            candidate_reference = None
            if config_reference.parts and config_reference.parts[0] == "configs":
                repository_root = experiment_path.parent
                while (
                    repository_root != repository_root.parent
                    and not (repository_root / "configs").exists()
                ):
                    repository_root = repository_root.parent
                repository_candidate = repository_root / config_reference
                if repository_candidate.exists():
                    candidate_reference = repository_candidate
            if candidate_reference is None:
                sibling_candidate = experiment_path.parent / config_reference
                if sibling_candidate.exists():
                    candidate_reference = sibling_candidate
            if candidate_reference is None:
                cwd_candidate = Path.cwd() / config_reference
                if cwd_candidate.exists():
                    candidate_reference = cwd_candidate
            if candidate_reference is not None:
                config_reference = candidate_reference
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

    _normalize_stage_metadata_aliases(resolved_experiment_config)
    _normalize_thesis_multitask_v3_aliases(resolved_experiment_config)
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

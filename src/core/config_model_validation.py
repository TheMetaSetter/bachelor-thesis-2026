from __future__ import annotations

from typing import Any


def _validate_model_and_task_config(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    task_config: dict[str, Any],
) -> None:
    model_name = model_config.get("model_name")
    redlamp_baseline_model_keys = {
        "model_name",
        "input_dim",
        "window_size",
        "latent_dim",
        "encoder_family",
        "mlp_num_linear_layers",
        "cnn_num_layers",
        "cnn_kernel_size",
        "cnn_hidden_channels",
        "cnn_dropout",
        "classifier_dim",
        "num_classes",
        "dropout",
        "lambda_recon",
        "lambda_cls",
        "use_label_refurbishment",
        "refurbishment_alpha",
        "refurbishment_beta",
        "enable_gradient_conflict_profiling",
        "gradient_profiling_scope",
        "gradient_focus_layer_name",
        "gradient_log_every_n_steps",
        "gradient_ema_alpha",
        "gradient_sma_window",
        "gradient_profile_include_bias",
    }
    allowed_model_keys_by_model_name = {
        "reconstruction_mlp_ae": {
            "model_name",
            "input_dim",
            "encoder_dim",
            "hidden_dim",
            "dropout",
        },
        "redlamp_baseline": redlamp_baseline_model_keys,
        "thesis_multitask": {
            "model_name",
            "enable_classification_path",
            "input_dim",
            "window_size",
            "encoder_dim",
            "hidden_dim",
            "encoder_family",
            "mlp_num_linear_layers",
            "cnn_num_layers",
            "cnn_kernel_size",
            "cnn_hidden_channels",
            "cnn_dropout",
            "num_classes",
            "dropout",
            "continuous_enabled",
            "continuous_num_prototypes",
            "discrete_enabled",
            "discrete_codebook_size",
            "gumbel_temperature",
            "temperature_start",
            "temperature_end",
            "temperature_anneal_fraction",
            "temperature_hold_fraction",
            "alpha_logit_init",
            "beta_logit_init",
            "use_label_refurbishment",
            "refurbishment_alpha",
            "refurbishment_beta",
            "reconstruction_normal_only",
            "lambda_recon",
            "lambda_cls",
            "lambda_div",
            "lambda_var",
            "lambda_cov",
            "lambda_use",
            "lambda_gate",
            "usage_lambda_start",
            "usage_lambda_end",
            "usage_lambda_schedule_fraction",
            "variance_floor_gamma",
            "gate_barrier_margin",
            "enable_two_view_contrastive",
            "contrastive_temperature",
            "lambda_contrastive",
            "enable_cka_gated_fusion",
            "cka_eps",
            "bootstrap_encoder_epochs",
            "discrete_ema_decay",
            "memory_norm_epsilon",
            "memory_initialization_batches",
            "memory_initialization_with_synthetic_windows",
            "training_phase",
            "fusion_mode",
            "discrete_query_mode",
            "discrete_topk",
            "discrete_query_temperature",
            "freeze_memories_after_initialization",
            "freeze_recovered_zipped_encoder_during_warmup",
            "discrete_memory_label_source",
            "stage_name",
            "enable_gradient_conflict_profiling",
            "gradient_profiling_scope",
            "gradient_focus_layer_name",
            "gradient_log_every_n_steps",
            "gradient_ema_alpha",
            "gradient_sma_window",
            "gradient_profile_include_bias",
            "enable_diversity_loss",
            "enable_variance_loss",
            "enable_covariance_loss",
            "enable_usage_loss",
            "enable_gate_loss",
        },
        "online_adaptation": {
            "model_name",
            "input_dim",
            "encoder_dim",
            "hidden_dim",
            "projector_hidden_dim",
            "projector_dropout",
            "enable_prototype_alignment",
            "lambda_align",
            "lambda_proto",
            "lambda_anchor",
            "score_source",
        },
    }
    unknown_model_keys = sorted(
        set(model_config) - allowed_model_keys_by_model_name[model_name]
    )
    if unknown_model_keys:
        raise ValueError(
            f"Unknown model config keys for model_name='{model_name}': "
            f"{unknown_model_keys}. Remove these keys from model config."
        )

    task_name = task_config.get("task_name")
    allowed_task_keys_by_task_name = {
        "reconstruction": {"task_name", "loss_name"},
        "multitask_tsad": {
            "task_name",
            "use_synthetic_augmentation",
            "use_synthetic_validation",
            "synthetic_train_seed",
            "synthetic_validation_seed",
            "classification_label_mode",
            "freeze_fusion_for_epochs",
            "warmup_alpha_value",
            "warmup_beta_value",
            "anomaly_probability",
            "train_balance_classes",
            "min_segment_fraction",
            "max_segment_fraction",
            "spike_scale",
            "anomaly_visibility_boost",
            "anomaly_families",
        },
        "online_adaptation": {
            "task_name",
            "reference_checkpoint_path",
            "warm_start_projector",
            "target_param_group",
            "clean_stream_only",
            "max_online_steps",
            "log_every_n_steps",
            "checkpoint_every_n_steps",
            "view_noise_std",
            "view_dropout_probability",
            "reset_policy",
            "reset_alignment_threshold",
        },
    }
    unknown_task_keys = sorted(
        set(task_config) - allowed_task_keys_by_task_name[task_name]
    )
    if unknown_task_keys:
        raise ValueError(
            f"Unknown task config keys for task_name='{task_name}': "
            f"{unknown_task_keys}. Remove these keys from task config."
        )

    integer_fields = {
        "seed": None,
        "epochs": None,
        "window_size": data_config.get("window_size"),
        "stride": data_config.get("stride"),
        "train_stride": data_config.get("train_stride"),
        "val_stride": data_config.get("val_stride"),
        "test_stride": data_config.get("test_stride"),
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
        integer_fields["cnn_num_layers"] = model_config.get("cnn_num_layers", 3)
        integer_fields["cnn_kernel_size"] = model_config.get("cnn_kernel_size", 3)
        integer_fields["cnn_hidden_channels"] = model_config.get(
            "cnn_hidden_channels", 64
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
        integer_fields["gradient_log_every_n_steps"] = model_config.get(
            "gradient_log_every_n_steps", 1
        )
        integer_fields["gradient_sma_window"] = model_config.get(
            "gradient_sma_window", 50
        )
    if model_config.get("model_name") == "redlamp_baseline":
        integer_fields["latent_dim"] = model_config.get("latent_dim")
        integer_fields["mlp_num_linear_layers"] = model_config.get(
            "mlp_num_linear_layers", 3
        )
        integer_fields["cnn_num_layers"] = model_config.get("cnn_num_layers", 3)
        integer_fields["cnn_kernel_size"] = model_config.get("cnn_kernel_size", 3)
        integer_fields["cnn_hidden_channels"] = model_config.get(
            "cnn_hidden_channels", 64
        )
        integer_fields["classifier_dim"] = model_config.get("classifier_dim")
        integer_fields["num_classes"] = model_config.get("num_classes")
        integer_fields["gradient_log_every_n_steps"] = model_config.get(
            "gradient_log_every_n_steps", 1
        )
        integer_fields["gradient_sma_window"] = model_config.get(
            "gradient_sma_window", 50
        )
    if model_config.get("model_name") == "online_adaptation":
        integer_fields["projector_hidden_dim"] = model_config.get(
            "projector_hidden_dim"
        )
    for field_name, field_value in integer_fields.items():
        if field_value is None:
            continue
        if not isinstance(field_value, int) or field_value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    float_fields = {
        "validation_split_ratio": data_config.get("validation_split_ratio"),
    }
    if model_config.get("model_name") in {
        "reconstruction_mlp_ae",
        "thesis_multitask",
        "redlamp_baseline",
    }:
        float_fields["dropout"] = model_config.get("dropout")
    if model_config.get("model_name") == "redlamp_baseline":
        float_fields["cnn_dropout"] = model_config.get(
            "cnn_dropout", model_config.get("dropout")
        )
        float_fields["gradient_ema_alpha"] = model_config.get("gradient_ema_alpha", 0.1)
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
        float_fields["lambda_recon"] = model_config.get("lambda_recon", 0.9)
        float_fields["lambda_cls"] = model_config.get("lambda_cls", 0.1)
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
            float_fields["gradient_ema_alpha"] = model_config.get(
                "gradient_ema_alpha", 0.1
            )
            float_fields["cnn_dropout"] = model_config.get(
                "cnn_dropout", model_config.get("dropout")
            )
        float_fields["warmup_alpha_value"] = task_config.get("warmup_alpha_value")
        float_fields["warmup_beta_value"] = task_config.get("warmup_beta_value")
        float_fields["anomaly_probability"] = task_config.get("anomaly_probability")
        float_fields["min_segment_fraction"] = task_config.get("min_segment_fraction")
        float_fields["max_segment_fraction"] = task_config.get("max_segment_fraction")
        float_fields["spike_scale"] = task_config.get("spike_scale")
        float_fields["anomaly_visibility_boost"] = task_config.get(
            "anomaly_visibility_boost", 1.5
        )
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

    if task_config.get("task_name") == "multitask_tsad":
        # The active multitask path is RedLamp-aligned by default unless a
        # config explicitly opts back into the older binary semantics.
        task_config.setdefault("train_balance_classes", True)
        if "classification_label_mode" not in task_config:
            if int(model_config.get("num_classes", 12)) == 2:
                task_config["classification_label_mode"] = "binary"
            else:
                task_config["classification_label_mode"] = "redlamp_multiclass"
        boolean_fields = {
            "enable_classification_path": model_config.get(
                "enable_classification_path", True
            ),
            "enable_diversity_loss": model_config.get("enable_diversity_loss", False),
            "enable_variance_loss": model_config.get("enable_variance_loss", False),
            "enable_covariance_loss": model_config.get("enable_covariance_loss", False),
            "enable_usage_loss": model_config.get("enable_usage_loss", False),
            "enable_gate_loss": model_config.get("enable_gate_loss", False),
            "use_label_refurbishment": model_config.get(
                "use_label_refurbishment", True
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
            "train_balance_classes": task_config.get("train_balance_classes", True),
        }
        if model_config.get("model_name") == "redlamp_baseline":
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


def _validate_data_runtime_config(data_config: dict[str, Any]) -> None:
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


def _validate_model_and_task_semantics(
    *,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    task_config: dict[str, Any],
) -> None:
    task_name = task_config.get("task_name")
    model_name = model_config.get("model_name")

    if task_name == "multitask_tsad":
        if int(model_config.get("mlp_num_linear_layers", 3)) < 2:
            raise ValueError("mlp_num_linear_layers must be at least 2")
        encoder_family = model_config.get("encoder_family", "mlp")
        if encoder_family not in {"mlp", "cnn_simple"}:
            raise ValueError("encoder_family must be one of: mlp, cnn_simple")
        cnn_num_layers = model_config.get("cnn_num_layers", 3)
        if not isinstance(cnn_num_layers, int) or cnn_num_layers < 2:
            raise ValueError("cnn_num_layers must be a positive integer >= 2")
        cnn_kernel_size = model_config.get("cnn_kernel_size", 3)
        if not isinstance(cnn_kernel_size, int) or cnn_kernel_size <= 0:
            raise ValueError("cnn_kernel_size must be a positive integer")
        cnn_hidden_channels = model_config.get("cnn_hidden_channels", 64)
        if not isinstance(cnn_hidden_channels, int) or cnn_hidden_channels <= 0:
            raise ValueError("cnn_hidden_channels must be a positive integer")
        if model_name == "thesis_multitask":
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
        if float(model_config.get("lambda_recon", 0.9)) < 0.0:
            raise ValueError("lambda_recon must be non-negative")
        if float(model_config.get("lambda_cls", 0.1)) < 0.0:
            raise ValueError("lambda_cls must be non-negative")
        anomaly_families = task_config.get("anomaly_families")
        if not isinstance(anomaly_families, list) or not anomaly_families:
            raise ValueError("anomaly_families must be a non-empty list")
        if not all(
            isinstance(family_name, str) and family_name
            for family_name in anomaly_families
        ):
            raise ValueError("anomaly_families must contain non-empty strings")
        classification_label_mode = task_config.get("classification_label_mode")
        if classification_label_mode is None:
            if int(model_config.get("num_classes", 12)) == 2:
                classification_label_mode = "binary"
            else:
                classification_label_mode = "redlamp_multiclass"
        if classification_label_mode not in {"binary", "redlamp_multiclass"}:
            raise ValueError(
                "classification_label_mode must be one of: binary, redlamp_multiclass"
            )
        if (
            classification_label_mode == "redlamp_multiclass"
            and int(model_config["num_classes"]) != 12
        ):
            raise ValueError(
                "classification_label_mode='redlamp_multiclass' requires num_classes == 12 "
                "for the active RedLamp-aligned multitask taxonomy"
            )
        if model_name == "thesis_multitask":
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
            if model_config.get("gradient_profiling_scope", "encoder_all") not in {
                "encoder_all"
            }:
                raise ValueError(
                    "gradient_profiling_scope must be one of {'encoder_all'}"
                )
            if model_config.get(
                "gradient_focus_layer_name", "encoder_last_affine"
            ) not in {"encoder_last_linear", "encoder_last_affine"}:
                raise ValueError(
                    "gradient_focus_layer_name must be one of: "
                    "encoder_last_linear, encoder_last_affine"
                )
            if int(model_config.get("gradient_log_every_n_steps", 1)) < 1:
                raise ValueError("gradient_log_every_n_steps must be >= 1")
            if int(model_config.get("gradient_sma_window", 50)) < 1:
                raise ValueError("gradient_sma_window must be >= 1")
            if not (0.0 < float(model_config.get("gradient_ema_alpha", 0.1)) <= 1.0):
                raise ValueError("gradient_ema_alpha must satisfy 0 < alpha <= 1")
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
        if float(task_config.get("anomaly_visibility_boost", 1.5)) <= 0.0:
            raise ValueError("anomaly_visibility_boost must be positive")
        synthetic_validation_seed = task_config.get("synthetic_validation_seed", 7)
        if (
            not isinstance(synthetic_validation_seed, int)
            or synthetic_validation_seed < 0
        ):
            raise ValueError("synthetic_validation_seed must be a non-negative integer")
        synthetic_train_seed = task_config.get("synthetic_train_seed")
        if synthetic_train_seed is not None and (
            not isinstance(synthetic_train_seed, int) or synthetic_train_seed < 0
        ):
            raise ValueError(
                "synthetic_train_seed must be a non-negative integer or null"
            )
        if synthetic_train_seed is not None and bool(
            data_config.get("shuffle_train", True)
        ):
            raise ValueError(
                "synthetic_train_seed requires data.shuffle_train=false so each "
                "window keeps a stable augmentation assignment across epochs"
            )
    if task_name == "online_adaptation":
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

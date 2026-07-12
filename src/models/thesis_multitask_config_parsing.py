from __future__ import annotations

"""Flat-kwargs parsing helpers for the thesis multitask model.

The public config dataclasses stay in `thesis_multitask_components.py`, while
this file keeps the long alias-and-group parsing logic short and easy to read.
"""

from typing import Any


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


def _take_group(
    remaining_kwargs: dict[str, Any],
    group_keys: set[str],
) -> dict[str, Any]:
    group_values: dict[str, Any] = {}
    for key in group_keys:
        if key in remaining_kwargs:
            group_values[key] = remaining_kwargs.pop(key)
    return group_values


def split_thesis_multitask_flat_kwargs(
    flat_kwargs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    remaining_kwargs = dict(flat_kwargs)

    if (
        "sample_variance_correction" in remaining_kwargs
        and "variance_correction" in remaining_kwargs
        and remaining_kwargs["sample_variance_correction"]
        != remaining_kwargs["variance_correction"]
    ):
        raise ValueError(
            "sample_variance_correction and variance_correction must match when both are provided"
        )
    if (
        "variance_correction" not in remaining_kwargs
        and "sample_variance_correction" in remaining_kwargs
    ):
        remaining_kwargs["variance_correction"] = remaining_kwargs.pop(
            "sample_variance_correction"
        )
    else:
        remaining_kwargs.pop("sample_variance_correction", None)
    if "variance_correction" in remaining_kwargs:
        remaining_kwargs["variance_correction"] = _normalize_variance_correction_value(
            remaining_kwargs["variance_correction"]
        )

    if "stage_name" in remaining_kwargs:
        stage_name = remaining_kwargs.pop("stage_name")
        if (
            "training_phase" in remaining_kwargs
            and remaining_kwargs["training_phase"] != stage_name
        ):
            raise ValueError(
                "training_phase and stage_name must match when both are provided"
            )
        remaining_kwargs["training_phase"] = stage_name

    architecture_keys = {
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
    }
    prototype_keys = {
        "continuous_enabled",
        "continuous_num_prototypes",
        "discrete_enabled",
        "discrete_codebook_size",
        "gumbel_temperature",
        "discrete_ema_decay",
        "stochastic_inference",
        "monte_carlo_samples",
        "continuous_temperature",
        "discrete_temperature",
        "variance_correction",
        "return_mc_samples",
        "sample_retention_policy",
    }
    schedule_keys = {
        "temperature_start",
        "temperature_end",
        "temperature_anneal_fraction",
        "temperature_hold_fraction",
        "usage_lambda_start",
        "usage_lambda_end",
        "usage_lambda_schedule_fraction",
        "freeze_fusion_for_epochs",
        "warmup_alpha_value",
        "warmup_beta_value",
    }
    objective_keys = {
        "enable_classification_path",
        "alpha_logit_init",
        "beta_logit_init",
        "use_label_refurbishment",
        "refurbishment_alpha",
        "refurbishment_beta",
        "reconstruction_normal_only",
        "lambda_recon",
        "lambda_cls",
        "enable_diversity_loss",
        "enable_variance_loss",
        "enable_covariance_loss",
        "enable_usage_loss",
        "enable_gate_loss",
        "lambda_div",
        "lambda_var",
        "lambda_cov",
        "lambda_use",
        "lambda_gate",
        "enable_score_loss",
        "score_loss_granularity",
        "score_loss_type",
        "score_loss_target",
        "score_loss_normalization",
        "score_loss_reduction",
        "variance_floor_gamma",
        "gate_barrier_margin",
        "enable_two_view_contrastive",
        "contrastive_temperature",
        "lambda_contrastive",
        "enable_cka_gated_fusion",
        "cka_eps",
    }
    memory_keys = {
        "bootstrap_encoder_epochs",
        "memory_norm_epsilon",
        "memory_initialization_batches",
        "memory_initialization_with_synthetic_windows",
    }
    runtime_keys = {
        "training_phase",
        "fusion_mode",
        "discrete_query_mode",
        "discrete_topk",
        "discrete_query_temperature",
        "freeze_memories_after_initialization",
        "freeze_recovered_zipped_encoder_during_warmup",
        "discrete_memory_label_source",
    }
    profiling_keys = {
        "enable_gradient_conflict_profiling",
        "gradient_profiling_scope",
        "gradient_focus_layer_name",
        "gradient_log_every_n_steps",
        "gradient_ema_alpha",
        "gradient_sma_window",
        "gradient_profile_include_bias",
    }
    synthetic_keys = {
        "use_synthetic_augmentation",
        "use_synthetic_validation",
        "synthetic_train_seed",
        "synthetic_validation_seed",
        "anomaly_probability",
        "min_segment_fraction",
        "max_segment_fraction",
        "spike_scale",
        "anomaly_visibility_boost",
        "train_balance_classes",
        "anomaly_families",
        "classification_label_mode",
    }

    architecture_values = _take_group(remaining_kwargs, architecture_keys)
    missing_required_architecture_keys = sorted(
        {"input_dim", "window_size", "encoder_dim", "hidden_dim"}
        - set(architecture_values)
    )
    if missing_required_architecture_keys:
        raise ValueError(
            "Missing required ThesisMultitaskModel architecture kwargs: "
            f"{missing_required_architecture_keys}"
        )

    prototype_values = _take_group(remaining_kwargs, prototype_keys)
    schedule_values = _take_group(remaining_kwargs, schedule_keys)
    objective_values = _take_group(remaining_kwargs, objective_keys)
    memory_values = _take_group(remaining_kwargs, memory_keys)
    runtime_values = _take_group(remaining_kwargs, runtime_keys)
    profiling_values = _take_group(remaining_kwargs, profiling_keys)
    synthetic_values = _take_group(remaining_kwargs, synthetic_keys)

    if (
        "classification_label_mode" not in synthetic_values
        and architecture_values.get("num_classes") == 2
    ):
        synthetic_values["classification_label_mode"] = "binary"
    if (
        "classification_label_mode" not in synthetic_values
        and architecture_values.get("num_classes") == 12
    ):
        # Keep flat kwargs aligned with the 12-class taxonomy instead of
        # silently defaulting to a binary assumption.
        synthetic_values["classification_label_mode"] = "redlamp_multiclass"
    if "anomaly_families" in synthetic_values:
        synthetic_values["anomaly_families"] = tuple(synthetic_values["anomaly_families"])

    if remaining_kwargs:
        raise ValueError(
            f"Unknown ThesisMultitaskModel flat kwargs: {sorted(remaining_kwargs)}"
        )

    return {
        "architecture_values": architecture_values,
        "prototype_values": prototype_values,
        "schedule_values": schedule_values,
        "objective_values": objective_values,
        "memory_values": memory_values,
        "runtime_values": runtime_values,
        "profiling_values": profiling_values,
        "synthetic_values": synthetic_values,
    }

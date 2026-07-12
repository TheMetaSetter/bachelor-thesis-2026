from __future__ import annotations

"""Setup helpers for the thesis multitask model.

The mixin keeps the public method names, while this file holds the longer
attribute-plumbing logic so the mixin stays readable and small.
"""

from collections import deque
from typing import Any

from src.core.console import print_parameter_summary
from src.models.thesis_multitask_components import (
    STAGE3_PHASE_CANONICAL_NAME,
    STAGE3_PHASE_LEGACY_NAME,
    TWO_STAGE_A_PHASE_NAME,
    TWO_STAGE_B_PHASE_NAME,
    TWO_STAGE_PHASE_NAMES,
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    ThesisMultitaskModelConfig,
)


def store_thesis_multitask_config_values(
    model: Any,
    config: ThesisMultitaskModelConfig,
) -> None:
    # The constructor copies each config field into an explicit attribute so
    # later code can read the runtime contract directly from the model object.
    architecture = config.architecture
    prototypes = config.prototypes
    schedule = config.schedule
    objective = config.objective
    memory = config.memory
    runtime = config.runtime
    profiling = config.profiling
    synthetic = config.synthetic

    model.model_config = config
    model.window_size = architecture.window_size
    model.hidden_dim = architecture.hidden_dim
    model.encoder_family = architecture.encoder_family
    model.mlp_num_linear_layers = architecture.mlp_num_linear_layers
    model.cnn_num_layers = architecture.cnn_num_layers
    model.cnn_kernel_size = architecture.cnn_kernel_size
    model.cnn_hidden_channels = architecture.cnn_hidden_channels
    model.cnn_dropout = architecture.cnn_dropout
    model.num_classes = architecture.num_classes
    model.continuous_num_prototypes = prototypes.continuous_num_prototypes
    model.discrete_codebook_size = prototypes.discrete_codebook_size
    model.default_gumbel_temperature = prototypes.gumbel_temperature
    model.gumbel_temperature = prototypes.gumbel_temperature
    model.stochastic_inference = prototypes.stochastic_inference
    model.monte_carlo_samples = prototypes.monte_carlo_samples
    model.continuous_temperature = prototypes.continuous_temperature
    model.discrete_temperature = prototypes.discrete_temperature
    model.variance_correction = prototypes.variance_correction
    model.return_mc_samples = prototypes.return_mc_samples
    model.sample_retention_policy = prototypes.sample_retention_policy
    model.temperature_start = schedule.temperature_start
    model.temperature_end = schedule.temperature_end
    model.temperature_anneal_fraction = schedule.temperature_anneal_fraction
    model.temperature_hold_fraction = schedule.temperature_hold_fraction
    model.use_label_refurbishment = objective.use_label_refurbishment
    model.refurbishment_alpha = objective.refurbishment_alpha
    model.refurbishment_beta = objective.refurbishment_beta
    model.reconstruction_normal_only = objective.reconstruction_normal_only
    model.lambda_recon = objective.lambda_recon
    model.lambda_cls = objective.lambda_cls
    model.enable_classification_path = objective.enable_classification_path
    model.lambda_div = objective.lambda_div
    model.lambda_var = objective.lambda_var
    model.lambda_cov = objective.lambda_cov
    model.lambda_use = objective.lambda_use
    model.lambda_gate = objective.lambda_gate
    model.usage_lambda_start = (
        objective.lambda_use
        if schedule.usage_lambda_start is None
        else schedule.usage_lambda_start
    )
    model.usage_lambda_end = (
        objective.lambda_use
        if schedule.usage_lambda_end is None
        else schedule.usage_lambda_end
    )
    model.usage_lambda_schedule_fraction = schedule.usage_lambda_schedule_fraction
    model.current_usage_lambda = model.usage_lambda_start
    model.enable_diversity_loss = objective.enable_diversity_loss
    model.enable_variance_loss = objective.enable_variance_loss
    model.enable_covariance_loss = objective.enable_covariance_loss
    model.enable_usage_loss = objective.enable_usage_loss
    model.enable_gate_loss = objective.enable_gate_loss
    model.enable_score_loss = objective.enable_score_loss
    model.score_loss_granularity = objective.score_loss_granularity
    model.score_loss_type = objective.score_loss_type
    model.score_loss_target = objective.score_loss_target
    model.score_loss_normalization = objective.score_loss_normalization
    model.score_loss_reduction = objective.score_loss_reduction
    model.variance_floor_gamma = objective.variance_floor_gamma
    model.gate_barrier_margin = objective.gate_barrier_margin
    model.enable_two_view_contrastive = objective.enable_two_view_contrastive
    model.contrastive_temperature = objective.contrastive_temperature
    model.lambda_contrastive = objective.lambda_contrastive
    model.enable_cka_gated_fusion = objective.enable_cka_gated_fusion
    model.cka_eps = objective.cka_eps
    model.bootstrap_encoder_epochs = memory.bootstrap_encoder_epochs
    model.enable_gradient_conflict_profiling = (
        profiling.enable_gradient_conflict_profiling
    )
    model.gradient_profiling_scope = profiling.gradient_profiling_scope
    model.gradient_focus_layer_name = profiling.gradient_focus_layer_name
    model.gradient_log_every_n_steps = profiling.gradient_log_every_n_steps
    model.gradient_ema_alpha = profiling.gradient_ema_alpha
    model.gradient_sma_window = profiling.gradient_sma_window
    model.gradient_profile_include_bias = profiling.gradient_profile_include_bias
    model._gradient_profile_train_step_count = 0
    model._gradient_profile_ema_state = {}
    model._gradient_profile_sma_buffers = {}
    model.discrete_ema_decay = prototypes.discrete_ema_decay
    model.memory_norm_epsilon = memory.memory_norm_epsilon
    model.memory_initialization_batches = memory.memory_initialization_batches
    model.memory_initialization_with_synthetic_windows = (
        memory.memory_initialization_with_synthetic_windows
    )
    model.training_phase = runtime.training_phase
    model.fusion_mode = runtime.fusion_mode
    model.discrete_query_mode = runtime.discrete_query_mode
    model.discrete_topk = runtime.discrete_topk
    model.discrete_query_temperature = runtime.discrete_query_temperature
    model.freeze_memories_after_initialization = (
        runtime.freeze_memories_after_initialization
    )
    model.freeze_recovered_zipped_encoder_during_warmup = (
        runtime.freeze_recovered_zipped_encoder_during_warmup
    )
    model.discrete_memory_label_source = runtime.discrete_memory_label_source
    model.use_synthetic_augmentation = synthetic.use_synthetic_augmentation
    model.use_synthetic_validation = synthetic.use_synthetic_validation
    model.synthetic_train_seed = synthetic.synthetic_train_seed
    model.synthetic_validation_seed = synthetic.synthetic_validation_seed
    model.classification_label_mode = synthetic.classification_label_mode
    model.freeze_fusion_for_epochs = schedule.freeze_fusion_for_epochs
    model.warmup_alpha_value = schedule.warmup_alpha_value
    model.warmup_beta_value = schedule.warmup_beta_value
    model.epsilon = 1e-6
    model.current_epoch_index = 0
    model.current_total_epochs = 1
    model.active_alpha_override = None
    model.active_beta_override = None
    model._score_loss_skipped_batches = 0
    model.continuous_memory_enabled = (
        prototypes.continuous_enabled and prototypes.continuous_num_prototypes > 0
    )
    model.discrete_memory_enabled = (
        prototypes.discrete_enabled and prototypes.discrete_codebook_size > 0
    )
    model.memory_initialized = memory.bootstrap_encoder_epochs <= 0
    model.memory_training_enabled = model.memory_initialized
    if model.training_phase in TWO_STAGE_PHASE_NAMES:
        model.memory_initialized = False
        model.memory_training_enabled = False
    model.memory_ready_for_initialization = False
    model.memory_initialization_epoch = None
    model.schedule_state = {
        "epoch": 1,
        "warmup_active": False,
        "freeze_fusion_for_epochs": model.freeze_fusion_for_epochs,
        "temperature": model.gumbel_temperature,
        "usage_lambda": model.current_usage_lambda,
    }
    if (
        model.use_label_refurbishment
        and model.classification_label_mode == "binary"
        and model.num_classes != 2
    ):
        raise ValueError(
            "label refurbishment currently supports only binary classification"
        )


def print_thesis_multitask_model_summary(
    model: Any,
    config: ThesisMultitaskModelConfig,
) -> None:
    architecture = config.architecture
    prototypes = config.prototypes
    schedule = config.schedule
    objective = config.objective
    memory = config.memory
    profiling = config.profiling
    synthetic = config.synthetic
    print_parameter_summary(
        "MODEL",
        "ThesisMultitaskModel",
        model,
        {
            "encoder": model.encoder,
            "continuous_prototype_bank": model.continuous_prototype_bank,
            "discrete_assignment": model.discrete_assignment,
            "discrete_codebook": model.discrete_codebook,
            "continuous_update_gate": model.continuous_update_gate,
            "reconstruction_head": model.reconstruction_head,
            "classification_head": model.classification_head,
            "alpha_logit": model.alpha_logit,
            "beta_logit": model.beta_logit,
        },
        input_dim=architecture.input_dim,
        window_size=architecture.window_size,
        encoder_dim=architecture.encoder_dim,
        hidden_dim=architecture.hidden_dim,
        encoder_family=architecture.encoder_family,
        mlp_num_linear_layers=architecture.mlp_num_linear_layers,
        cnn_num_layers=architecture.cnn_num_layers,
        cnn_kernel_size=architecture.cnn_kernel_size,
        cnn_hidden_channels=architecture.cnn_hidden_channels,
        num_classes=architecture.num_classes,
        use_label_refurbishment=objective.use_label_refurbishment,
        refurbishment_alpha=objective.refurbishment_alpha,
        refurbishment_beta=objective.refurbishment_beta,
        reconstruction_normal_only=objective.reconstruction_normal_only,
        lambda_recon=objective.lambda_recon,
        lambda_cls=objective.lambda_cls,
        lambda_div=objective.lambda_div,
        lambda_var=objective.lambda_var,
        lambda_cov=objective.lambda_cov,
        lambda_use=objective.lambda_use,
        lambda_gate=objective.lambda_gate,
        enable_score_loss=objective.enable_score_loss,
        score_loss_granularity=objective.score_loss_granularity,
        score_loss_type=objective.score_loss_type,
        score_loss_target=objective.score_loss_target,
        score_loss_normalization=objective.score_loss_normalization,
        score_loss_reduction=objective.score_loss_reduction,
        temperature_start=schedule.temperature_start,
        temperature_end=schedule.temperature_end,
        temperature_hold_fraction=schedule.temperature_hold_fraction,
        usage_lambda_start=model.usage_lambda_start,
        usage_lambda_end=model.usage_lambda_end,
        usage_lambda_schedule_fraction=schedule.usage_lambda_schedule_fraction,
        bootstrap_encoder_epochs=memory.bootstrap_encoder_epochs,
        enable_gradient_conflict_profiling=(
            profiling.enable_gradient_conflict_profiling
        ),
        gradient_profiling_scope=profiling.gradient_profiling_scope,
        gradient_focus_layer_name=profiling.gradient_focus_layer_name,
        gradient_log_every_n_steps=profiling.gradient_log_every_n_steps,
        gradient_ema_alpha=profiling.gradient_ema_alpha,
        gradient_sma_window=profiling.gradient_sma_window,
        gradient_profile_include_bias=profiling.gradient_profile_include_bias,
        discrete_ema_decay=prototypes.discrete_ema_decay,
        memory_norm_epsilon=memory.memory_norm_epsilon,
        memory_initialization_batches=memory.memory_initialization_batches,
        memory_initialization_with_synthetic_windows=(
            memory.memory_initialization_with_synthetic_windows
        ),
        use_synthetic_validation=synthetic.use_synthetic_validation,
        synthetic_validation_seed=synthetic.synthetic_validation_seed,
    )

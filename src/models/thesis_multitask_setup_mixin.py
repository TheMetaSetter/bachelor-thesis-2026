from __future__ import annotations

"""Mixin extracted from the thesis multitask model.

This file keeps constructor and configuration plumbing together so the main
model file can stay below the code-size limit without changing runtime
behavior.
"""

import math
import time
from collections import OrderedDict, deque
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.console import (
    console_print,
    print_parameter_summary,
    summarize_batch,
    summarize_label_distribution,
    summarize_tensor,
)
from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import SyntheticAnomalyInjector
from src.models.thesis_multitask_components import (
    STAGE3_PHASE_CANONICAL_NAME,
    STAGE3_PHASE_LEGACY_NAME,
    TWO_STAGE_A_PHASE_NAME,
    TWO_STAGE_B_PHASE_NAME,
    TWO_STAGE_PHASE_NAMES,
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    MultitaskArchitectureConfig,
    MultitaskWindowEncoder,
    ObjectiveConfig,
    MemoryInitializationConfig,
    PrototypeBranchConfig,
    ScheduleAndWarmupConfig,
    SyntheticAnomalyConfig,
    ThesisMultitaskModelConfig,
    build_multilayer_perceptron,
)
from src.models.neural_blocks import _initialize_mlp_linear_layers


class ThesisMultitaskSetupMixin:
    def _store_config_values(self, config: ThesisMultitaskModelConfig) -> None:
        # This constructor stores both the architecture and the experiment
        # switches because the repository follows the one-model-one-file rule.
        architecture = config.architecture
        prototypes = config.prototypes
        schedule = config.schedule
        objective = config.objective
        memory = config.memory
        runtime = config.runtime
        profiling = config.profiling
        synthetic = config.synthetic

        self.model_config = config
        self.window_size = architecture.window_size
        self.hidden_dim = architecture.hidden_dim
        self.encoder_family = architecture.encoder_family
        self.mlp_num_linear_layers = architecture.mlp_num_linear_layers
        self.cnn_num_layers = architecture.cnn_num_layers
        self.cnn_kernel_size = architecture.cnn_kernel_size
        self.cnn_hidden_channels = architecture.cnn_hidden_channels
        self.cnn_dropout = architecture.cnn_dropout
        self.num_classes = architecture.num_classes
        self.continuous_num_prototypes = prototypes.continuous_num_prototypes
        self.discrete_codebook_size = prototypes.discrete_codebook_size
        self.default_gumbel_temperature = prototypes.gumbel_temperature
        self.gumbel_temperature = prototypes.gumbel_temperature
        self.temperature_start = schedule.temperature_start
        self.temperature_end = schedule.temperature_end
        self.temperature_anneal_fraction = schedule.temperature_anneal_fraction
        self.temperature_hold_fraction = schedule.temperature_hold_fraction
        self.use_label_refurbishment = objective.use_label_refurbishment
        self.refurbishment_alpha = objective.refurbishment_alpha
        self.refurbishment_beta = objective.refurbishment_beta
        self.reconstruction_normal_only = objective.reconstruction_normal_only
        self.lambda_recon = objective.lambda_recon
        self.lambda_cls = objective.lambda_cls
        self.enable_classification_path = objective.enable_classification_path
        self.lambda_div = objective.lambda_div
        self.lambda_var = objective.lambda_var
        self.lambda_cov = objective.lambda_cov
        self.lambda_use = objective.lambda_use
        self.lambda_gate = objective.lambda_gate
        self.usage_lambda_start = (
            objective.lambda_use
            if schedule.usage_lambda_start is None
            else schedule.usage_lambda_start
        )
        self.usage_lambda_end = (
            objective.lambda_use
            if schedule.usage_lambda_end is None
            else schedule.usage_lambda_end
        )
        self.usage_lambda_schedule_fraction = schedule.usage_lambda_schedule_fraction
        self.current_usage_lambda = self.usage_lambda_start
        self.enable_diversity_loss = objective.enable_diversity_loss
        self.enable_variance_loss = objective.enable_variance_loss
        self.enable_covariance_loss = objective.enable_covariance_loss
        self.enable_usage_loss = objective.enable_usage_loss
        self.enable_gate_loss = objective.enable_gate_loss
        self.enable_score_loss = objective.enable_score_loss
        self.score_loss_granularity = objective.score_loss_granularity
        self.score_loss_type = objective.score_loss_type
        self.score_loss_target = objective.score_loss_target
        self.score_loss_normalization = objective.score_loss_normalization
        self.score_loss_reduction = objective.score_loss_reduction
        self.variance_floor_gamma = objective.variance_floor_gamma
        self.gate_barrier_margin = objective.gate_barrier_margin
        self.enable_two_view_contrastive = objective.enable_two_view_contrastive
        self.contrastive_temperature = objective.contrastive_temperature
        self.lambda_contrastive = objective.lambda_contrastive
        self.enable_cka_gated_fusion = objective.enable_cka_gated_fusion
        self.cka_eps = objective.cka_eps
        self.bootstrap_encoder_epochs = memory.bootstrap_encoder_epochs
        self.enable_gradient_conflict_profiling = (
            profiling.enable_gradient_conflict_profiling
        )
        self.gradient_profiling_scope = profiling.gradient_profiling_scope
        self.gradient_focus_layer_name = profiling.gradient_focus_layer_name
        self.gradient_log_every_n_steps = profiling.gradient_log_every_n_steps
        self.gradient_ema_alpha = profiling.gradient_ema_alpha
        self.gradient_sma_window = profiling.gradient_sma_window
        self.gradient_profile_include_bias = profiling.gradient_profile_include_bias
        self._gradient_profile_train_step_count = 0
        self._gradient_profile_ema_state: dict[str, float] = {}
        self._gradient_profile_sma_buffers: dict[str, deque[float]] = {}
        self.discrete_ema_decay = prototypes.discrete_ema_decay
        self.memory_norm_epsilon = memory.memory_norm_epsilon
        self.memory_initialization_batches = memory.memory_initialization_batches
        self.memory_initialization_with_synthetic_windows = (
            memory.memory_initialization_with_synthetic_windows
        )
        self.training_phase = runtime.training_phase
        self.fusion_mode = runtime.fusion_mode
        self.discrete_query_mode = runtime.discrete_query_mode
        self.discrete_topk = runtime.discrete_topk
        self.discrete_query_temperature = runtime.discrete_query_temperature
        self.freeze_memories_after_initialization = (
            runtime.freeze_memories_after_initialization
        )
        self.freeze_recovered_zipped_encoder_during_warmup = (
            runtime.freeze_recovered_zipped_encoder_during_warmup
        )
        self.discrete_memory_label_source = runtime.discrete_memory_label_source
        self.use_synthetic_augmentation = synthetic.use_synthetic_augmentation
        self.use_synthetic_validation = synthetic.use_synthetic_validation
        self.synthetic_train_seed = synthetic.synthetic_train_seed
        self.synthetic_validation_seed = synthetic.synthetic_validation_seed
        self.classification_label_mode = synthetic.classification_label_mode
        self.freeze_fusion_for_epochs = schedule.freeze_fusion_for_epochs
        self.warmup_alpha_value = schedule.warmup_alpha_value
        self.warmup_beta_value = schedule.warmup_beta_value
        self.epsilon = 1e-6
        self.current_epoch_index = 0
        self.current_total_epochs = 1
        self.active_alpha_override: float | None = None
        self.active_beta_override: float | None = None
        self._score_loss_skipped_batches = 0
        self.continuous_memory_enabled = (
            prototypes.continuous_enabled and prototypes.continuous_num_prototypes > 0
        )
        self.discrete_memory_enabled = (
            prototypes.discrete_enabled and prototypes.discrete_codebook_size > 0
        )
        self.memory_initialized = memory.bootstrap_encoder_epochs <= 0
        self.memory_training_enabled = self.memory_initialized
        if self.training_phase in TWO_STAGE_PHASE_NAMES:
            self.memory_initialized = False
            self.memory_training_enabled = False
        self.memory_ready_for_initialization = False
        self.memory_initialization_epoch: int | None = None
        self.schedule_state = {
            "epoch": 1,
            "warmup_active": False,
            "freeze_fusion_for_epochs": self.freeze_fusion_for_epochs,
            "temperature": self.gumbel_temperature,
            "usage_lambda": self.current_usage_lambda,
        }
        if (
            self.use_label_refurbishment
            and self.classification_label_mode == "binary"
            and self.num_classes != 2
        ):
            raise ValueError(
                "label refurbishment currently supports only binary classification"
            )

    def _phase_uses_prototype_path(self) -> bool:
        # Active two-stage runs only use the prototype path in Stage B.
        return self.training_phase in {
            STAGE3_PHASE_CANONICAL_NAME,
            "multitask_pretraining",
            TWO_STAGE_B_PHASE_NAME,
        }

    def _phase_uses_contrastive_objective(self) -> bool:
        return self.training_phase in {
            "stage1_classification",
            "stage1_reconstruction",
            "multitask_pretraining",
            TWO_STAGE_A_PHASE_NAME,
        }

    def _phase_reconstruction_weight(self) -> float:
        if self.training_phase == "stage1_classification":
            return 0.0
        if self.training_phase == "stage1_reconstruction":
            return 1.0
        return float(self.lambda_recon)

    def _phase_classification_weight(self) -> float:
        if not self.enable_classification_path:
            return 0.0
        if self.training_phase == "stage1_classification":
            return 1.0
        if self.training_phase == "stage1_reconstruction":
            return 0.0
        return float(self.lambda_cls)

    def _phase_contrastive_weight(self) -> float:
        if self.training_phase in {"stage1_classification", "stage1_reconstruction"}:
            return 0.1
        if self.training_phase == TWO_STAGE_B_PHASE_NAME:
            return 0.0
        return float(self.lambda_contrastive)

    def _phase_freezes_encoder(self) -> bool:
        # Stage B is the active freeze point for the two-stage rerun.
        return self.training_phase == TWO_STAGE_B_PHASE_NAME or (
            self.training_phase == STAGE3_PHASE_CANONICAL_NAME
            and self.freeze_recovered_zipped_encoder_during_warmup
        )

    def _set_module_requires_grad(
        self,
        module: nn.Module | None,
        *,
        requires_grad: bool,
    ) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _set_parameter_requires_grad(
        self,
        parameter: nn.Parameter | None,
        *,
        requires_grad: bool,
    ) -> None:
        if parameter is None:
            return
        parameter.requires_grad = requires_grad

    def _configure_trainable_parameters_for_phase(self) -> None:
        self._set_module_requires_grad(self.encoder, requires_grad=True)
        self._set_module_requires_grad(
            self.reconstruction_head,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.classification_head,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.reconstruction_concat_projection,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.classification_concat_projection,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.classification_fusion_gate,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.reconstruction_fusion_gate,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.continuous_update_gate,
            requires_grad=True,
        )
        self._set_module_requires_grad(
            self.discrete_assignment,
            requires_grad=True,
        )
        self._set_parameter_requires_grad(self.alpha_logit, requires_grad=True)
        self._set_parameter_requires_grad(self.beta_logit, requires_grad=True)

        if self.training_phase == "stage1_classification":
            self._set_module_requires_grad(
                self.reconstruction_head,
                requires_grad=False,
            )
        if self.training_phase == "stage1_reconstruction":
            self._set_module_requires_grad(
                self.classification_head,
                requires_grad=False,
            )
        if not self._phase_uses_prototype_path():
            self._set_module_requires_grad(
                self.reconstruction_concat_projection,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.classification_concat_projection,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.classification_fusion_gate,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.reconstruction_fusion_gate,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.continuous_update_gate,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.discrete_assignment,
                requires_grad=False,
            )
        if self.training_phase in {STAGE3_PHASE_CANONICAL_NAME, TWO_STAGE_B_PHASE_NAME}:
            self._set_module_requires_grad(
                self.classification_fusion_gate,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.reconstruction_fusion_gate,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.continuous_update_gate,
                requires_grad=False,
            )
            self._set_module_requires_grad(
                self.discrete_assignment,
                requires_grad=False,
            )
            self._set_parameter_requires_grad(self.alpha_logit, requires_grad=False)
            self._set_parameter_requires_grad(self.beta_logit, requires_grad=False)
        if self._phase_freezes_encoder():
            self._set_module_requires_grad(self.encoder, requires_grad=False)

    def _classification_class_names(self) -> tuple[str, ...]:
        if self.classification_label_mode == "binary":
            return ("normal", "anomaly")
        return REDLAMP_MULTICLASS_CLASS_NAMES

    def _build_encoder(self, config: ThesisMultitaskModelConfig) -> None:
        architecture = config.architecture
        # Encoder block.
        # This produces the common hidden state that both prototype branches see.
        self.encoder = MultitaskWindowEncoder(architecture)

    def _build_prototype_memory(self, config: ThesisMultitaskModelConfig) -> None:
        architecture = config.architecture
        prototypes = config.prototypes
        # Continuous branch.
        # This branch retrieves a soft prototype context from a learned bank.
        if self.continuous_memory_enabled:
            self.register_buffer(
                "continuous_prototype_bank",
                torch.randn(
                    prototypes.continuous_num_prototypes,
                    architecture.hidden_dim,
                ),
            )
        else:
            self.register_buffer("continuous_prototype_bank", None)

        # Discrete branch.
        # The active two-stage rerun uses cosine-topk, so the Gumbel-only
        # assignment head stays optional in Stage B while legacy phases keep it.
        if self.discrete_memory_enabled:
            if (
                self.training_phase == TWO_STAGE_B_PHASE_NAME
                and self.discrete_query_mode == "cosine_topk"
            ):
                self.discrete_assignment = None
            else:
                self.discrete_assignment = nn.Linear(
                    architecture.hidden_dim,
                    prototypes.discrete_codebook_size,
                )
            self.register_buffer(
                "discrete_codebook",
                torch.randn(
                    prototypes.discrete_codebook_size,
                    architecture.hidden_dim,
                ),
            )
            self.register_buffer(
                "discrete_ema_counts",
                torch.zeros(prototypes.discrete_codebook_size),
            )
            self.register_buffer(
                "discrete_ema_sums",
                torch.zeros(
                    prototypes.discrete_codebook_size,
                    architecture.hidden_dim,
                ),
            )
        else:
            self.discrete_assignment = None
            self.register_buffer("discrete_codebook", None)
            self.register_buffer("discrete_ema_counts", None)
            self.register_buffer("discrete_ema_sums", None)

        self.continuous_update_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        _initialize_mlp_linear_layers(self.continuous_update_gate)

    def _build_fusion_parameters(self, config: ThesisMultitaskModelConfig) -> None:
        objective = config.objective
        # Fusion scalars.
        # `alpha` controls the classification mix and `beta` controls the
        # reconstruction mix so the two tasks can prefer different geometry.
        self.alpha_logit = nn.Parameter(torch.tensor(float(objective.alpha_logit_init)))
        self.beta_logit = nn.Parameter(torch.tensor(float(objective.beta_logit_init)))

    def _build_task_heads(self, config: ThesisMultitaskModelConfig) -> None:
        architecture = config.architecture
        # Task heads.
        # Supervision lives on the fused task-specific hidden states, not on the
        # branch-local states. That keeps the branches observable but not separate predictors.
        self.reconstruction_head = build_multilayer_perceptron(
            input_dim=architecture.hidden_dim,
            intermediate_dim=architecture.encoder_dim,
            output_dim=architecture.input_dim,
            num_linear_layers=architecture.mlp_num_linear_layers,
            dropout=architecture.dropout,
            apply_output_activation=False,
        )

        self.classification_head = build_multilayer_perceptron(
            input_dim=architecture.window_size * architecture.hidden_dim,
            intermediate_dim=architecture.hidden_dim,
            output_dim=architecture.num_classes,
            num_linear_layers=architecture.mlp_num_linear_layers,
            dropout=architecture.dropout,
            apply_output_activation=False,
        )
        self.classification_fusion_gate = nn.Sequential(
            nn.Linear(2, architecture.hidden_dim),
            nn.ReLU(),
            nn.Linear(architecture.hidden_dim, 1),
        )
        _initialize_mlp_linear_layers(self.classification_fusion_gate)
        self.reconstruction_fusion_gate = nn.Sequential(
            nn.Linear(2, architecture.hidden_dim),
            nn.ReLU(),
            nn.Linear(architecture.hidden_dim, 1),
        )
        _initialize_mlp_linear_layers(self.reconstruction_fusion_gate)
        self.classification_concat_projection = build_multilayer_perceptron(
            input_dim=2 * architecture.hidden_dim,
            intermediate_dim=architecture.hidden_dim,
            output_dim=architecture.hidden_dim,
            num_linear_layers=2,
            dropout=architecture.dropout,
            apply_output_activation=False,
        )
        self.reconstruction_concat_projection = build_multilayer_perceptron(
            input_dim=2 * architecture.hidden_dim,
            intermediate_dim=architecture.hidden_dim,
            output_dim=architecture.hidden_dim,
            num_linear_layers=2,
            dropout=architecture.dropout,
            apply_output_activation=False,
        )

    def _build_synthetic_injectors(self, config: ThesisMultitaskModelConfig) -> None:
        architecture = config.architecture
        synthetic = config.synthetic
        # Offline objective helpers.
        # Optional losses are activated by `lambda_*` so ablations can stay on
        # one codepath instead of branching into separate model variants. The
        # intended starting point is still only reconstruction plus
        # classification loss until observed failure modes justify more terms.
        self.branch_layer_norm = nn.LayerNorm(architecture.hidden_dim)
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=synthetic.anomaly_probability,
            min_segment_fraction=synthetic.min_segment_fraction,
            max_segment_fraction=synthetic.max_segment_fraction,
            spike_scale=synthetic.spike_scale,
            anomaly_visibility_boost=synthetic.anomaly_visibility_boost,
            anomaly_families=synthetic.anomaly_families,
            train_balance_classes=synthetic.train_balance_classes,
            deterministic_seed=synthetic.synthetic_train_seed,
            classification_label_mode=synthetic.classification_label_mode,
        )
        self.synthetic_validation_injector = SyntheticAnomalyInjector(
            anomaly_probability=synthetic.anomaly_probability,
            min_segment_fraction=synthetic.min_segment_fraction,
            max_segment_fraction=synthetic.max_segment_fraction,
            spike_scale=synthetic.spike_scale,
            anomaly_visibility_boost=synthetic.anomaly_visibility_boost,
            anomaly_families=synthetic.anomaly_families,
            train_balance_classes=synthetic.train_balance_classes,
            deterministic_seed=synthetic.synthetic_validation_seed,
            classification_label_mode=synthetic.classification_label_mode,
        )

    def _build_optional_loss_configs(self) -> None:
        self.optional_loss_configs: dict[str, dict[str, Any]] = {
            "diversity_loss": {
                "enabled": self.lambda_div > 0.0,
                "weight": self.lambda_div,
                "compute_fn": self._compute_cross_branch_diversity_loss,
            },
            "variance_loss": {
                "enabled": self.lambda_var > 0.0,
                "weight": self.lambda_var,
                "compute_fn": self._compute_variance_floor_loss,
            },
            "covariance_loss": {
                "enabled": self.lambda_cov > 0.0,
                "weight": self.lambda_cov,
                "compute_fn": self._compute_covariance_reduction_loss,
            },
            "usage_loss": {
                "enabled": max(self.usage_lambda_start, self.usage_lambda_end) > 0.0,
                "weight": self.lambda_use,
                "compute_fn": self._compute_prototype_usage_loss,
            },
            "gate_loss": {
                "enabled": self.lambda_gate > 0.0,
                "weight": self.lambda_gate,
                "compute_fn": self._compute_gate_regularization_loss,
            },
        }

    def _print_model_summary(self, config: ThesisMultitaskModelConfig) -> None:
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
            self,
            {
                "encoder": self.encoder,
                "continuous_prototype_bank": self.continuous_prototype_bank,
                "discrete_assignment": self.discrete_assignment,
                "discrete_codebook": self.discrete_codebook,
                "continuous_update_gate": self.continuous_update_gate,
                "reconstruction_head": self.reconstruction_head,
                "classification_head": self.classification_head,
                "alpha_logit": self.alpha_logit,
                "beta_logit": self.beta_logit,
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
            usage_lambda_start=self.usage_lambda_start,
            usage_lambda_end=self.usage_lambda_end,
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

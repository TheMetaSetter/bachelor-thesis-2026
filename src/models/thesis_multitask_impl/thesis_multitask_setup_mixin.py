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
    summarize_batch,
    summarize_label_distribution,
    summarize_tensor,
)
from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import SyntheticAnomalyInjector
from src.models.thesis_multitask_impl.thesis_multitask_components import (
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
from src.models.thesis_multitask_impl.thesis_multitask_setup_helpers import (
    print_thesis_multitask_model_summary,
    store_thesis_multitask_config_values,
)


class ThesisMultitaskSetupMixin:
    def _store_config_values(self, config: ThesisMultitaskModelConfig) -> None:
        store_thesis_multitask_config_values(self, config)

    def _phase_uses_prototype_path(self) -> bool:
        return self.training_phase in {
            TWO_STAGE_A_PHASE_NAME,
            TWO_STAGE_B_PHASE_NAME,
        }

    def _phase_uses_contrastive_objective(self) -> bool:
        return self.training_phase == TWO_STAGE_A_PHASE_NAME

    def _phase_reconstruction_weight(self) -> float:
        return float(self.lambda_recon)

    def _phase_classification_weight(self) -> float:
        if not self.enable_classification_path:
            return 0.0
        return float(self.lambda_cls)

    def _phase_contrastive_weight(self) -> float:
        return float(self.lambda_contrastive)

    def _phase_freezes_encoder(self) -> bool:
        return self.training_phase == TWO_STAGE_B_PHASE_NAME

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
        if self.training_phase == TWO_STAGE_B_PHASE_NAME:
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

            # khởi tạo ...
            self.anomalous_codeword_mask = torch.zeros(
                prototypes.discrete_codebook_size, dtype=torch.bool
            )

            # khởi tạo bán kính của các cụm abstract anomalous pattern
            # hoặc abstract normal pattern
            # mà có centroid nằm trong codebook
            self.anomaly_radii = torch.zeros(prototypes.discrete_codebook_size)

            self.verification_metadata_source = "uninitialized"
            self.verification_metadata_schema_version = 1
            self.verification_metadata_split = "synthetic_train"
            self.verification_metadata_initialization_seed = (
                self.synthetic_train_seed
                if self.synthetic_train_seed is not None
                else self.synthetic_validation_seed
            )
            self.verification_codeword_class_ids = torch.zeros(
                prototypes.discrete_codebook_size, dtype=torch.long
            )
            self.verification_contributing_token_counts = torch.zeros(
                prototypes.discrete_codebook_size, dtype=torch.float32
            )
        else:
            self.discrete_assignment = None
            self.register_buffer("discrete_codebook", None)
            self.register_buffer("discrete_ema_counts", None)
            self.register_buffer("discrete_ema_sums", None)
            self.anomalous_codeword_mask = None
            self.anomaly_radii = None
            self.verification_metadata_source = "disabled"
            self.verification_metadata_schema_version = 1
            self.verification_metadata_split = "synthetic_train"
            self.verification_metadata_initialization_seed = 0
            self.verification_codeword_class_ids = None
            self.verification_contributing_token_counts = None

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
        injector_kwargs = {
            "anomaly_probability": synthetic.anomaly_probability,
            "min_segment_fraction": synthetic.min_segment_fraction,
            "max_segment_fraction": synthetic.max_segment_fraction,
            "spike_scale": synthetic.spike_scale,
            "anomaly_visibility_boost": synthetic.anomaly_visibility_boost,
            "anomaly_families": synthetic.anomaly_families,
            "train_balance_classes": synthetic.train_balance_classes,
            "classification_label_mode": synthetic.classification_label_mode,
        }
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            **injector_kwargs,
            deterministic_seed=synthetic.synthetic_train_seed,
        )
        self.synthetic_validation_injector = SyntheticAnomalyInjector(
            **injector_kwargs,
            deterministic_seed=synthetic.synthetic_validation_seed,
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
        print_thesis_multitask_model_summary(self, config)

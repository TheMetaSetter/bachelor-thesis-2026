from __future__ import annotations

"""Self-contained multitask prototype-fusion model.

This is the main offline thesis model, so the file is intentionally long and
intentionally self-contained. A fresher should read it in this order: encoder,
continuous branch, discrete branch, fusion, optional losses, then the shared
stage step that assembles the training objective.
"""

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
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
from src.data.augment import (
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    SyntheticAnomalyInjector,
)
from src.models.base_model import BaseModel
from src.models.neural_blocks import SimpleWindowCnnEncoder, build_multilayer_perceptron

# Legacy Stage 3 label kept only so older configs and checkpoints still load.
STAGE3_PHASE_LEGACY_NAME = "stage3_prototype_warmup"
STAGE3_PHASE_CANONICAL_NAME = "stage3_memory_initialization_and_fusion_warmup"
TWO_STAGE_A_PHASE_NAME = "stage_a_multitask_pretraining"
TWO_STAGE_B_PHASE_NAME = "stage_b_fusion_finetuning"
TWO_STAGE_PHASE_NAMES = {TWO_STAGE_A_PHASE_NAME, TWO_STAGE_B_PHASE_NAME}


class MultitaskWindowEncoder(nn.Module):
    def __init__(
        self,
        architecture: "MultitaskArchitectureConfig",
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder_family = architecture.encoder_family
        if architecture.encoder_family == "mlp":
            # The encoder depth is shared with both task heads so the offline model
            # can form a symmetric MLP contract from YAML instead of hard-coding
            # different depths in different submodules.
            self.network = build_multilayer_perceptron(
                input_dim=architecture.input_dim,
                intermediate_dim=architecture.encoder_dim,
                output_dim=architecture.hidden_dim,
                num_linear_layers=architecture.mlp_num_linear_layers,
                dropout=architecture.dropout,
                apply_output_activation=True,
            )
        elif architecture.encoder_family == "cnn_simple":
            self.network = SimpleWindowCnnEncoder(
                input_dim=architecture.input_dim,
                output_dim=architecture.hidden_dim,
                hidden_channels=architecture.cnn_hidden_channels,
                kernel_size=architecture.cnn_kernel_size,
                num_layers=architecture.cnn_num_layers,
                dropout=architecture.cnn_dropout,
            )
        else:
            raise ValueError(
                f"Unsupported encoder_family: {architecture.encoder_family}"
            )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        hidden = self.network(batch["x"])
        if hidden.shape[1] != self.architecture.window_size:
            raise ValueError(
                "encoder must preserve window_size="
                f"{self.architecture.window_size}, but received hidden.shape[1]="
                f"{hidden.shape[1]}"
            )
        return {
            "hidden": hidden,
            "pooled": hidden.mean(dim=1),
            "aux": {"encoder_name": "multitask_window_encoder"},
        }


@dataclass(frozen=True)
class MultitaskArchitectureConfig:
    input_dim: int
    window_size: int
    encoder_dim: int
    hidden_dim: int
    encoder_family: str = "mlp"
    mlp_num_linear_layers: int = 3
    cnn_num_layers: int = 3
    cnn_kernel_size: int = 3
    cnn_hidden_channels: int = 64
    cnn_dropout: float = 0.0
    num_classes: int = len(REDLAMP_MULTICLASS_CLASS_NAMES)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.encoder_family not in {"mlp", "cnn_simple"}:
            raise ValueError("encoder_family must be one of: mlp, cnn_simple")
        if self.mlp_num_linear_layers < 2:
            raise ValueError("mlp_num_linear_layers must be at least 2")
        if self.cnn_num_layers < 2:
            raise ValueError("cnn_num_layers must be at least 2")
        if self.cnn_kernel_size <= 0:
            raise ValueError("cnn_kernel_size must be positive")
        if self.cnn_hidden_channels <= 0:
            raise ValueError("cnn_hidden_channels must be positive")
        if not 0.0 <= self.cnn_dropout <= 1.0:
            raise ValueError("cnn_dropout must be between 0 and 1")


@dataclass(frozen=True)
class PrototypeBranchConfig:
    continuous_enabled: bool = True
    continuous_num_prototypes: int = 8
    discrete_enabled: bool = True
    discrete_codebook_size: int = 16
    gumbel_temperature: float = 1.0
    discrete_ema_decay: float = 0.99


@dataclass(frozen=True)
class ScheduleAndWarmupConfig:
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    temperature_anneal_fraction: float = 1.0
    temperature_hold_fraction: float = 0.0
    usage_lambda_start: float | None = None
    usage_lambda_end: float | None = None
    usage_lambda_schedule_fraction: float = 1.0
    freeze_fusion_for_epochs: int = 0
    warmup_alpha_value: float = 0.5
    warmup_beta_value: float = 0.5


@dataclass(frozen=True)
class ObjectiveConfig:
    enable_classification_path: bool = True
    alpha_logit_init: float = 0.0
    beta_logit_init: float = 0.0
    use_label_refurbishment: bool = True
    refurbishment_alpha: float = 0.0
    refurbishment_beta: float = 0.0
    reconstruction_normal_only: bool = False
    lambda_recon: float = 0.9
    lambda_cls: float = 0.1
    enable_diversity_loss: bool = False
    enable_variance_loss: bool = False
    enable_covariance_loss: bool = False
    enable_usage_loss: bool = False
    enable_gate_loss: bool = False
    lambda_div: float = 0.0
    lambda_var: float = 0.0
    lambda_cov: float = 0.0
    lambda_use: float = 0.0
    lambda_gate: float = 0.0
    enable_score_loss: bool = False
    score_loss_granularity: str = "point"
    score_loss_type: str = "pointwise_balanced_bce_logits"
    score_loss_target: str = "synthetic_anomaly_mask"
    score_loss_normalization: str = "train_batch_normal_tokens_detached_mean_std"
    score_loss_reduction: str = "pointwise_binary_balanced_mean"
    variance_floor_gamma: float = 1.0
    gate_barrier_margin: float = 0.25
    enable_two_view_contrastive: bool = False
    contrastive_temperature: float = 0.1
    lambda_contrastive: float = 1.0
    enable_cka_gated_fusion: bool = False
    cka_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.lambda_recon < 0.0:
            raise ValueError("lambda_recon must be non-negative")
        if self.lambda_cls < 0.0:
            raise ValueError("lambda_cls must be non-negative")
        if self.score_loss_granularity not in {"point"}:
            raise ValueError("score_loss_granularity must be one of: point")
        if self.score_loss_type not in {
            "pointwise_balanced_bce_logits",
            "pointwise_balanced_reconstruction_score",
        }:
            raise ValueError(
                "score_loss_type must be one of: "
                "pointwise_balanced_bce_logits, pointwise_balanced_reconstruction_score"
            )
        if self.score_loss_target != "synthetic_anomaly_mask":
            raise ValueError("score_loss_target must be 'synthetic_anomaly_mask'")
        if self.score_loss_normalization not in {
            "train_batch_normal_tokens_detached_mean_std",
            "batch_normal_tokens_detached_mean_std",
        }:
            raise ValueError(
                "score_loss_normalization must be one of: "
                "train_batch_normal_tokens_detached_mean_std, "
                "batch_normal_tokens_detached_mean_std"
            )
        if self.score_loss_reduction != "pointwise_binary_balanced_mean":
            raise ValueError(
                "score_loss_reduction must be 'pointwise_binary_balanced_mean'"
            )


@dataclass(frozen=True)
class MemoryInitializationConfig:
    bootstrap_encoder_epochs: int = 0
    memory_norm_epsilon: float = 1.0e-6
    memory_initialization_batches: int = 16
    memory_initialization_with_synthetic_windows: bool = True


@dataclass(frozen=True)
class ThreeStageRuntimeConfig:
    training_phase: str = "multitask_pretraining"
    fusion_mode: str = "learnable_sigmoid_scalars"
    discrete_query_mode: str = "gumbel_softmax"
    discrete_topk: int = 1
    discrete_query_temperature: float = 0.1
    freeze_memories_after_initialization: bool = False
    freeze_recovered_zipped_encoder_during_warmup: bool = False
    discrete_memory_label_source: str = "synthetic_train_labels"

    def __post_init__(self) -> None:
        normalized_training_phase = (
            STAGE3_PHASE_CANONICAL_NAME
            if self.training_phase == STAGE3_PHASE_LEGACY_NAME
            else self.training_phase
        )
        object.__setattr__(self, "training_phase", normalized_training_phase)
        if self.training_phase not in {
            "stage1_classification",
            "stage1_reconstruction",
            "stage2_recovery",
            STAGE3_PHASE_CANONICAL_NAME,
            "multitask_pretraining",
            TWO_STAGE_A_PHASE_NAME,
            TWO_STAGE_B_PHASE_NAME,
        }:
            raise ValueError(
                "training_phase must be one of: stage1_classification, "
                "stage1_reconstruction, stage2_recovery, "
                f"{STAGE3_PHASE_CANONICAL_NAME}, multitask_pretraining, "
                f"{TWO_STAGE_A_PHASE_NAME}, {TWO_STAGE_B_PHASE_NAME}"
            )
        if self.fusion_mode not in {
            "task_specific_concat_projection",
            "learnable_sigmoid_scalars",
        }:
            raise ValueError(
                "fusion_mode must be one of: "
                "task_specific_concat_projection, learnable_sigmoid_scalars"
            )
        if self.discrete_query_mode not in {"cosine_topk", "gumbel_softmax"}:
            raise ValueError(
                "discrete_query_mode must be one of: cosine_topk, gumbel_softmax"
            )
        if self.discrete_topk < 1:
            raise ValueError("discrete_topk must be >= 1")
        if self.discrete_query_temperature <= 0.0:
            raise ValueError("discrete_query_temperature must be positive")
        if self.discrete_memory_label_source != "synthetic_train_labels":
            raise ValueError(
                "discrete_memory_label_source must be 'synthetic_train_labels'"
            )


@dataclass(frozen=True)
class GradientProfilingConfig:
    enable_gradient_conflict_profiling: bool = False
    gradient_profiling_scope: str = "encoder_all"
    gradient_focus_layer_name: str = "encoder_last_affine"
    gradient_log_every_n_steps: int = 1
    gradient_ema_alpha: float = 0.1
    gradient_sma_window: int = 50
    gradient_profile_include_bias: bool = False

    def __post_init__(self) -> None:
        if self.gradient_profiling_scope not in {"encoder_all"}:
            raise ValueError("gradient_profiling_scope must be one of {'encoder_all'}")
        if self.gradient_focus_layer_name not in {
            "encoder_last_linear",
            "encoder_last_affine",
        }:
            raise ValueError(
                "gradient_focus_layer_name must be one of: "
                "encoder_last_linear, encoder_last_affine"
            )
        if self.gradient_log_every_n_steps < 1:
            raise ValueError("gradient_log_every_n_steps must be >= 1")
        if self.gradient_sma_window < 1:
            raise ValueError("gradient_sma_window must be >= 1")
        if not (0.0 < self.gradient_ema_alpha <= 1.0):
            raise ValueError("gradient_ema_alpha must satisfy 0 < alpha <= 1")


@dataclass(frozen=True)
class SyntheticAnomalyConfig:
    use_synthetic_augmentation: bool = True
    use_synthetic_validation: bool = True
    synthetic_train_seed: int | None = None
    synthetic_validation_seed: int = 7
    anomaly_probability: float = 0.5
    min_segment_fraction: float = 0.2
    max_segment_fraction: float = 0.3
    spike_scale: float = 3.0
    anomaly_visibility_boost: float = 1.5
    train_balance_classes: bool = True
    anomaly_families: tuple[str, ...] = REDLAMP_ANOMALY_FAMILIES
    classification_label_mode: str = "redlamp_multiclass"

    def __post_init__(self) -> None:
        if self.synthetic_train_seed is not None and self.synthetic_train_seed < 0:
            raise ValueError(
                "synthetic_train_seed must be a non-negative integer or None"
            )
        if self.classification_label_mode not in {"binary", "redlamp_multiclass"}:
            raise ValueError(
                "classification_label_mode must be one of: binary, redlamp_multiclass"
            )
        object.__setattr__(self, "anomaly_families", tuple(self.anomaly_families))


@dataclass(frozen=True)
class ThesisMultitaskModelConfig:
    architecture: MultitaskArchitectureConfig
    prototypes: PrototypeBranchConfig = field(default_factory=PrototypeBranchConfig)
    schedule: ScheduleAndWarmupConfig = field(default_factory=ScheduleAndWarmupConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    memory: MemoryInitializationConfig = field(
        default_factory=MemoryInitializationConfig
    )
    runtime: ThreeStageRuntimeConfig = field(default_factory=ThreeStageRuntimeConfig)
    profiling: GradientProfilingConfig = field(default_factory=GradientProfilingConfig)
    synthetic: SyntheticAnomalyConfig = field(default_factory=SyntheticAnomalyConfig)

    @classmethod
    def from_flat_kwargs(
        cls, flat_kwargs: dict[str, Any]
    ) -> "ThesisMultitaskModelConfig":
        remaining_kwargs = dict(flat_kwargs)

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

        def take_group(group_keys: set[str]) -> dict[str, Any]:
            group_values: dict[str, Any] = {}
            for key in group_keys:
                if key in remaining_kwargs:
                    group_values[key] = remaining_kwargs.pop(key)
            return group_values

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

        architecture_values = take_group(architecture_keys)
        missing_required_architecture_keys = sorted(
            {"input_dim", "window_size", "encoder_dim", "hidden_dim"}
            - set(architecture_values)
        )
        if missing_required_architecture_keys:
            raise ValueError(
                "Missing required ThesisMultitaskModel architecture kwargs: "
                f"{missing_required_architecture_keys}"
            )

        prototype_values = take_group(prototype_keys)
        schedule_values = take_group(schedule_keys)
        objective_values = take_group(objective_keys)
        memory_values = take_group(memory_keys)
        runtime_values = take_group(runtime_keys)
        profiling_values = take_group(profiling_keys)
        synthetic_values = take_group(synthetic_keys)
        if (
            "classification_label_mode" not in synthetic_values
            and architecture_values.get("num_classes") == 2
        ):
            synthetic_values["classification_label_mode"] = "binary"
        if (
            "classification_label_mode" not in synthetic_values
            and architecture_values.get("num_classes") == 12
        ):
            # Keep the flat-kwargs path aligned with the active 12-class
            # taxonomy instead of remaining implicitly binary-first.
            synthetic_values["classification_label_mode"] = "redlamp_multiclass"
        if "anomaly_families" in synthetic_values:
            synthetic_values["anomaly_families"] = tuple(
                synthetic_values["anomaly_families"]
            )

        if remaining_kwargs:
            raise ValueError(
                f"Unknown ThesisMultitaskModel flat kwargs: {sorted(remaining_kwargs)}"
            )

        return cls(
            architecture=MultitaskArchitectureConfig(**architecture_values),
            prototypes=PrototypeBranchConfig(**prototype_values),
            schedule=ScheduleAndWarmupConfig(**schedule_values),
            objective=ObjectiveConfig(**objective_values),
            memory=MemoryInitializationConfig(**memory_values),
            runtime=ThreeStageRuntimeConfig(**runtime_values),
            profiling=GradientProfilingConfig(**profiling_values),
            synthetic=SyntheticAnomalyConfig(**synthetic_values),
        )

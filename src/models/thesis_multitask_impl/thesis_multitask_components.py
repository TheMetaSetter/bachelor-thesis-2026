from __future__ import annotations

"""Self-contained multitask prototype-fusion model.

This is the main offline thesis model, so the file is intentionally long and
intentionally self-contained. A fresher should read it in this order: encoder,
continuous branch, discrete branch, fusion, optional losses, then the shared
stage step that assembles the training objective.
"""

from dataclasses import dataclass, field
from typing import Any

from src.data.augment import (
    REDLAMP_ANOMALY_FAMILIES,  # not contain "normal"
    REDLAMP_MULTICLASS_CLASS_NAMES,  # contain "normal"
)
from src.models.neural_blocks import SimpleWindowCnnEncoder, build_multilayer_perceptron
from src.models.thesis_multitask_impl.thesis_multitask_config_parsing import (
    split_thesis_multitask_flat_kwargs,
)

TWO_STAGE_A_PHASE_NAME = "stage_a_multitask_pretraining"
TWO_STAGE_B_PHASE_NAME = "stage_b_fusion_finetuning"
TWO_STAGE_PHASE_NAMES = {TWO_STAGE_A_PHASE_NAME, TWO_STAGE_B_PHASE_NAME}


@dataclass(frozen=True)
class MultitaskArchitectureConfig:
    input_dim: int
    window_size: int
    encoder_dim: int
    hidden_dim: int
    encoder_family: str = "cnn_simple"
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
    stochastic_inference: bool = True
    monte_carlo_samples: int = 10
    continuous_temperature: float = 0.9
    discrete_temperature: float = 0.9
    variance_correction: int = 1
    return_mc_samples: bool = False
    sample_retention_policy: str = "none"

    def __post_init__(self) -> None:
        if self.continuous_num_prototypes < 0:
            raise ValueError("continuous_num_prototypes must be non-negative")
        if self.discrete_codebook_size < 0:
            raise ValueError("discrete_codebook_size must be non-negative")
        if self.gumbel_temperature <= 0.0:
            raise ValueError("gumbel_temperature must be positive")
        if not isinstance(self.stochastic_inference, bool):
            raise TypeError("stochastic_inference must be boolean")
        if self.monte_carlo_samples < 1:
            raise ValueError("monte_carlo_samples must be at least 1")
        if self.continuous_temperature <= 0.0:
            raise ValueError("continuous_temperature must be positive")
        if self.discrete_temperature <= 0.0:
            raise ValueError("discrete_temperature must be positive")
        if self.variance_correction not in {0, 1}:
            raise ValueError("variance_correction must be 0 or 1")
        if not isinstance(self.return_mc_samples, bool):
            raise TypeError("return_mc_samples must be boolean")
        if self.sample_retention_policy not in {
            "none",
            "retain_all",
            "retain_for_eda",
        }:
            raise ValueError(
                "sample_retention_policy must be one of: none, retain_all, retain_for_eda"
            )


@dataclass(frozen=True)
class QueryBundle:
    hidden: Any
    normalized_hidden: Any
    continuous_memory_bank: Any = None
    discrete_codebook: Any = None
    continuous_logits: Any = None
    discrete_logits: Any = None
    memory_bypass_active: bool = False
    discrete_query_mode: str = "cosine_topk"
    continuous_temperature: float = 1.0
    discrete_temperature: float = 1.0
    discrete_topk: int = 3


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
        if self.variance_floor_gamma <= 0.0:
            raise ValueError("variance_floor_gamma must be positive")


@dataclass(frozen=True)
class MemoryInitializationConfig:
    bootstrap_encoder_epochs: int = 0
    memory_norm_epsilon: float = 1.0e-6
    memory_initialization_batches: int = 16
    memory_initialization_with_synthetic_windows: bool = True


@dataclass(frozen=True)
class ActiveRuntimeConfig:
    training_phase: str = TWO_STAGE_A_PHASE_NAME
    fusion_mode: str = "learnable_sigmoid_scalars"
    discrete_query_mode: str = "gumbel_softmax"
    discrete_topk: int = 1
    discrete_query_temperature: float = 0.1
    freeze_memories_after_initialization: bool = False
    discrete_memory_label_source: str = "synthetic_train_labels"

    def __post_init__(self) -> None:
        if self.training_phase not in {
            TWO_STAGE_A_PHASE_NAME,
            TWO_STAGE_B_PHASE_NAME,
        }:
            raise ValueError(
                "training_phase must be one of: "
                f"{TWO_STAGE_A_PHASE_NAME}, {TWO_STAGE_B_PHASE_NAME}"
            )
        if self.fusion_mode not in {
            "task_specific_concat_projection",
            "learnable_sigmoid_scalars",
            "direct_branch_routing",
        }:
            raise ValueError(
                "fusion_mode must be one of: "
                "task_specific_concat_projection, learnable_sigmoid_scalars, "
                "direct_branch_routing"
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
    runtime: ActiveRuntimeConfig = field(default_factory=ActiveRuntimeConfig)
    profiling: GradientProfilingConfig = field(default_factory=GradientProfilingConfig)
    synthetic: SyntheticAnomalyConfig = field(default_factory=SyntheticAnomalyConfig)

    @classmethod
    def from_flat_kwargs(
        cls, flat_kwargs: dict[str, Any]
    ) -> "ThesisMultitaskModelConfig":
        sections = split_thesis_multitask_flat_kwargs(flat_kwargs)
        return cls(
            architecture=MultitaskArchitectureConfig(**sections["architecture_values"]),
            prototypes=PrototypeBranchConfig(**sections["prototype_values"]),
            schedule=ScheduleAndWarmupConfig(**sections["schedule_values"]),
            objective=ObjectiveConfig(**sections["objective_values"]),
            memory=MemoryInitializationConfig(**sections["memory_values"]),
            runtime=ActiveRuntimeConfig(**sections["runtime_values"]),
            profiling=GradientProfilingConfig(**sections["profiling_values"]),
            synthetic=SyntheticAnomalyConfig(**sections["synthetic_values"]),
        )


# Compatibility re-export: existing mixins and public imports keep resolving.
from src.models.thesis_multitask_impl.thesis_multitask_encoder import (
    MultitaskWindowEncoder,
)

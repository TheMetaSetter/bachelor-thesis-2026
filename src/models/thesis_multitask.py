from __future__ import annotations

"""Self-contained multitask prototype-fusion model.

This is the main offline thesis model, so the file is intentionally long and
intentionally self-contained. A fresher should read it in this order: encoder,
continuous branch, discrete branch, fusion, optional losses, then the shared
stage step that assembles the training objective.
"""

import math
import time
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


def build_multilayer_perceptron(
    *,
    input_dim: int,
    intermediate_dim: int,
    output_dim: int,
    num_linear_layers: int,
    dropout: float,
    apply_output_activation: bool,
) -> nn.Sequential:
    """Build a readable MLP with one explicit shared depth contract.

    The repository keeps one thesis model in one file, so this helper exists
    only to avoid repeating the same layer-construction pattern three times for
    the encoder, reconstruction head, and classification head.
    """
    if num_linear_layers < 2:
        raise ValueError("num_linear_layers must be at least 2")

    layer_dims = (
        [input_dim] + [intermediate_dim] * (num_linear_layers - 1) + [output_dim]
    )
    network_layers: list[nn.Module] = []
    for layer_index, (layer_input_dim, layer_output_dim) in enumerate(
        zip(layer_dims[:-1], layer_dims[1:])
    ):
        is_last_linear_layer = layer_index == num_linear_layers - 1
        network_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
        if not is_last_linear_layer:
            network_layers.append(nn.ReLU())
            network_layers.append(nn.Dropout(dropout))
        elif apply_output_activation:
            network_layers.append(nn.ReLU())

    mlp = nn.Sequential(*network_layers)
    _initialize_mlp_linear_layers(mlp)
    return mlp


def _initialize_mlp_linear_layers(mlp: nn.Sequential) -> None:
    """Apply the repository-wide MLP initialization policy.

    Policy:
    - Linear immediately followed by ReLU: Kaiming-uniform initialization.
    - Final/output linear (not followed by ReLU): Xavier-uniform initialization.
    - All linear biases: zero.
    """
    for layer_index, layer in enumerate(mlp):
        if not isinstance(layer, nn.Linear):
            continue
        next_layer = mlp[layer_index + 1] if layer_index + 1 < len(mlp) else None
        followed_by_relu = isinstance(next_layer, nn.ReLU)
        if followed_by_relu:
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SimpleWindowCnnEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        layer_dims = [input_dim] + [hidden_channels] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for layer_index, (layer_input_dim, layer_output_dim) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:])
        ):
            is_last_layer = layer_index == num_layers - 1
            padding_total = kernel_size - 1
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            layers.append(nn.ConstantPad1d((padding_left, padding_right), 0.0))
            layers.append(nn.Conv1d(layer_input_dim, layer_output_dim, kernel_size))
            if not is_last_layer:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self._initialize_conv_layers()

    def _initialize_conv_layers(self) -> None:
        for layer in self.network:
            if not isinstance(layer, nn.Conv1d):
                continue
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, D]")
        x_channel_first = x.transpose(1, 2)
        hidden_channel_first = self.network(x_channel_first)
        return hidden_channel_first.transpose(1, 2)


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
    use_label_refurbishment: bool = False
    refurbishment_alpha: float = 0.0
    refurbishment_beta: float = 0.0
    reconstruction_normal_only: bool = False
    lambda_cls: float = 1.0
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
    variance_floor_gamma: float = 1.0
    gate_barrier_margin: float = 0.25
    enable_two_view_contrastive: bool = False
    contrastive_temperature: float = 0.1
    lambda_contrastive: float = 1.0
    enable_cka_gated_fusion: bool = False
    cka_eps: float = 1.0e-6


@dataclass(frozen=True)
class MemoryInitializationConfig:
    bootstrap_encoder_epochs: int = 0
    memory_norm_epsilon: float = 1.0e-6
    memory_initialization_batches: int = 16
    memory_initialization_with_synthetic_windows: bool = True


@dataclass(frozen=True)
class SyntheticAnomalyConfig:
    use_synthetic_augmentation: bool = True
    use_synthetic_validation: bool = True
    synthetic_validation_seed: int = 7
    val_realistic: bool = True
    val_realistic_source: str = "test_same_scope"
    val_anomaly_rate_override: float | None = None
    anomaly_probability: float = 0.5
    min_segment_fraction: float = 0.2
    max_segment_fraction: float = 0.3
    spike_scale: float = 3.0
    anomaly_visibility_boost: float = 1.5
    train_balance_classes: bool = False
    anomaly_families: tuple[str, ...] = REDLAMP_ANOMALY_FAMILIES
    classification_label_mode: str = "redlamp_multiclass"

    def __post_init__(self) -> None:
        if self.classification_label_mode not in {"binary", "redlamp_multiclass"}:
            raise ValueError(
                "classification_label_mode must be one of: binary, redlamp_multiclass"
            )
        if self.val_realistic_source not in {"test_same_scope", "test_smd_all"}:
            raise ValueError(
                "val_realistic_source must be one of: test_same_scope, test_smd_all"
            )
        if self.val_anomaly_rate_override is not None and not 0.0 <= float(
            self.val_anomaly_rate_override
        ) <= 1.0:
            raise ValueError("val_anomaly_rate_override must be between 0 and 1")
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
    synthetic: SyntheticAnomalyConfig = field(default_factory=SyntheticAnomalyConfig)

    @classmethod
    def from_flat_kwargs(
        cls, flat_kwargs: dict[str, Any]
    ) -> "ThesisMultitaskModelConfig":
        remaining_kwargs = dict(flat_kwargs)

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
        synthetic_keys = {
            "use_synthetic_augmentation",
            "use_synthetic_validation",
            "synthetic_validation_seed",
            "val_realistic",
            "val_realistic_source",
            "val_anomaly_rate_override",
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
        synthetic_values = take_group(synthetic_keys)
        if (
            "classification_label_mode" not in synthetic_values
            and architecture_values.get("num_classes") == 2
        ):
            synthetic_values["classification_label_mode"] = "binary"
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
            synthetic=SyntheticAnomalyConfig(**synthetic_values),
        )


class ThesisMultitaskModel(BaseModel):
    def __init__(
        self,
        config: ThesisMultitaskModelConfig | None = None,
        **flat_kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None and flat_kwargs:
            raise ValueError("Pass either config or flat keyword arguments, not both")
        if config is None:
            config = ThesisMultitaskModelConfig.from_flat_kwargs(flat_kwargs)
        if not isinstance(config, ThesisMultitaskModelConfig):
            raise TypeError("config must be a ThesisMultitaskModelConfig instance")

        self._store_config_values(config)
        self._build_encoder(config)
        self._build_prototype_memory(config)
        self._build_fusion_parameters(config)
        self._build_task_heads(config)
        self._build_synthetic_injectors(config)
        self._build_optional_loss_configs()
        self.set_epoch_context(epoch_index=0, total_epochs=1)
        self._print_model_summary(config)

    def _store_config_values(self, config: ThesisMultitaskModelConfig) -> None:
        # This constructor stores both the architecture and the experiment
        # switches because the repository follows the one-model-one-file rule.
        architecture = config.architecture
        prototypes = config.prototypes
        schedule = config.schedule
        objective = config.objective
        memory = config.memory
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
        self.variance_floor_gamma = objective.variance_floor_gamma
        self.gate_barrier_margin = objective.gate_barrier_margin
        self.enable_two_view_contrastive = objective.enable_two_view_contrastive
        self.contrastive_temperature = objective.contrastive_temperature
        self.lambda_contrastive = objective.lambda_contrastive
        self.enable_cka_gated_fusion = objective.enable_cka_gated_fusion
        self.cka_eps = objective.cka_eps
        self.bootstrap_encoder_epochs = memory.bootstrap_encoder_epochs
        self.discrete_ema_decay = prototypes.discrete_ema_decay
        self.memory_norm_epsilon = memory.memory_norm_epsilon
        self.memory_initialization_batches = memory.memory_initialization_batches
        self.memory_initialization_with_synthetic_windows = (
            memory.memory_initialization_with_synthetic_windows
        )
        self.use_synthetic_augmentation = synthetic.use_synthetic_augmentation
        self.use_synthetic_validation = synthetic.use_synthetic_validation
        self.synthetic_validation_seed = synthetic.synthetic_validation_seed
        self.val_realistic = synthetic.val_realistic
        self.val_realistic_source = synthetic.val_realistic_source
        self.val_anomaly_rate_override = synthetic.val_anomaly_rate_override
        self.classification_label_mode = synthetic.classification_label_mode
        self.freeze_fusion_for_epochs = schedule.freeze_fusion_for_epochs
        self.warmup_alpha_value = schedule.warmup_alpha_value
        self.warmup_beta_value = schedule.warmup_beta_value
        self.epsilon = 1e-6
        self.current_epoch_index = 0
        self.current_total_epochs = 1
        self.active_alpha_override: float | None = None
        self.active_beta_override: float | None = None
        self.continuous_memory_enabled = (
            prototypes.continuous_enabled and prototypes.continuous_num_prototypes > 0
        )
        self.discrete_memory_enabled = (
            prototypes.discrete_enabled and prototypes.discrete_codebook_size > 0
        )
        self.memory_initialized = memory.bootstrap_encoder_epochs <= 0
        self.memory_training_enabled = self.memory_initialized
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
        # This branch assigns tokens to a codebook through Gumbel-Softmax.
        if self.discrete_memory_enabled:
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
            lambda_cls=objective.lambda_cls,
            lambda_div=objective.lambda_div,
            lambda_var=objective.lambda_var,
            lambda_cov=objective.lambda_cov,
            lambda_use=objective.lambda_use,
            lambda_gate=objective.lambda_gate,
            temperature_start=schedule.temperature_start,
            temperature_end=schedule.temperature_end,
            temperature_hold_fraction=schedule.temperature_hold_fraction,
            usage_lambda_start=self.usage_lambda_start,
            usage_lambda_end=self.usage_lambda_end,
            usage_lambda_schedule_fraction=schedule.usage_lambda_schedule_fraction,
            bootstrap_encoder_epochs=memory.bootstrap_encoder_epochs,
            discrete_ema_decay=prototypes.discrete_ema_decay,
            memory_norm_epsilon=memory.memory_norm_epsilon,
            memory_initialization_batches=memory.memory_initialization_batches,
            memory_initialization_with_synthetic_windows=(
                memory.memory_initialization_with_synthetic_windows
            ),
            use_synthetic_validation=synthetic.use_synthetic_validation,
            synthetic_validation_seed=synthetic.synthetic_validation_seed,
        )

    def _zero_loss(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        return reference_tensor.new_zeros(())

    def _compute_temperature_for_epoch(
        self, epoch_index: int, total_epochs: int
    ) -> float:
        # The temperature schedule is kept inside the model because it changes
        # the discrete branch behavior, not the generic trainer behavior.
        hold_epochs = math.ceil(total_epochs * self.temperature_hold_fraction)
        if epoch_index < hold_epochs:
            return float(self.temperature_start)

        anneal_epoch_index = max(epoch_index - hold_epochs, 0)
        anneal_epochs = max(
            1, math.ceil(total_epochs * self.temperature_anneal_fraction)
        )
        if anneal_epochs == 1:
            progress = 0.0
        else:
            progress = min(anneal_epoch_index / float(anneal_epochs - 1), 1.0)
        if progress <= 0.0:
            return float(self.temperature_start)
        if progress >= 1.0:
            return float(self.temperature_end)
        return float(
            self.temperature_start
            + progress * (self.temperature_end - self.temperature_start)
        )

    def _compute_usage_lambda_for_epoch(
        self, epoch_index: int, total_epochs: int
    ) -> float:
        usage_schedule_epochs = max(
            1, math.ceil(total_epochs * self.usage_lambda_schedule_fraction)
        )
        if usage_schedule_epochs == 1:
            progress = 1.0
        else:
            progress = min(epoch_index / float(usage_schedule_epochs - 1), 1.0)
        if progress <= 0.0:
            return float(self.usage_lambda_start)
        if progress >= 1.0:
            return float(self.usage_lambda_end)
        return float(
            self.usage_lambda_start
            + progress * (self.usage_lambda_end - self.usage_lambda_start)
        )

    def set_epoch_context(self, epoch_index: int, total_epochs: int) -> None:
        # Warm-up can temporarily pin fusion to a known regime so ablations can
        # compare continuous-only, discrete-only, and fused behavior cleanly.
        self.current_epoch_index = epoch_index
        self.current_total_epochs = total_epochs
        self.gumbel_temperature = self._compute_temperature_for_epoch(
            epoch_index, total_epochs
        )
        self.current_usage_lambda = self._compute_usage_lambda_for_epoch(
            epoch_index, total_epochs
        )
        warmup_active = epoch_index < self.freeze_fusion_for_epochs
        self.active_alpha_override = self.warmup_alpha_value if warmup_active else None
        self.active_beta_override = self.warmup_beta_value if warmup_active else None
        self.schedule_state = {
            "epoch": epoch_index + 1,
            "warmup_active": warmup_active,
            "freeze_fusion_for_epochs": self.freeze_fusion_for_epochs,
            "temperature": self.gumbel_temperature,
            "usage_lambda": self.current_usage_lambda,
            "bootstrap_active": self._is_bootstrap_active(),
            "train_memory_mode": float(
                not self._should_bypass_memory_for_stage("train")
            ),
        }
        console_print(
            "MODEL",
            "Updated multitask epoch context",
            epoch=epoch_index + 1,
            total_epochs=total_epochs,
            warmup_active=warmup_active,
            temperature=self.gumbel_temperature,
            usage_lambda=self.current_usage_lambda,
            alpha_override=self.active_alpha_override,
            beta_override=self.active_beta_override,
            bootstrap_active=self.schedule_state["bootstrap_active"],
            train_memory_mode=self.schedule_state["train_memory_mode"],
        )

    def get_schedule_state(self) -> dict[str, Any]:
        return dict(self.schedule_state)

    def _is_bootstrap_active(self) -> bool:
        return (
            self.bootstrap_encoder_epochs > 0
            and self.current_epoch_index < self.bootstrap_encoder_epochs
            and not self.memory_initialized
        )

    def _should_bypass_memory_for_stage(self, stage_name: str) -> bool:
        del stage_name
        return self._is_bootstrap_active() or not self.memory_initialized

    def _should_update_memory(self, stage_name: str) -> bool:
        return (
            stage_name == "train"
            and self.memory_training_enabled
            and self.memory_initialized
        )

    def get_memory_lifecycle_state(self) -> dict[str, Any]:
        return {
            "bootstrap_encoder_epochs": self.bootstrap_encoder_epochs,
            "current_epoch": self.current_epoch_index + 1,
            "memory_initialized": self.memory_initialized,
            "memory_training_enabled": self.memory_training_enabled,
            "memory_ready_for_initialization": self.memory_ready_for_initialization,
            "memory_initialization_epoch": self.memory_initialization_epoch,
            "memory_mode": float(not self._should_bypass_memory_for_stage("train")),
            "train_memory_mode": float(
                not self._should_bypass_memory_for_stage("train")
            ),
        }

    def get_checkpoint_extra_state(self) -> dict[str, Any]:
        return self.get_memory_lifecycle_state()

    def get_memory_tensor_state(self) -> dict[str, torch.Tensor | None]:
        return {
            "continuous_prototype_bank": (
                None
                if self.continuous_prototype_bank is None
                else self.continuous_prototype_bank.detach().clone()
            ),
            "discrete_codebook": (
                None
                if self.discrete_codebook is None
                else self.discrete_codebook.detach().clone()
            ),
            "discrete_ema_counts": (
                None
                if self.discrete_ema_counts is None
                else self.discrete_ema_counts.detach().clone()
            ),
            "discrete_ema_sums": (
                None
                if self.discrete_ema_sums is None
                else self.discrete_ema_sums.detach().clone()
            ),
        }

    def mark_memories_initialized(
        self, initialization_epoch: int | None = None
    ) -> None:
        self.memory_initialized = True
        self.memory_training_enabled = True
        self.memory_ready_for_initialization = False
        self.memory_initialization_epoch = initialization_epoch
        console_print(
            "MODEL",
            "Marked prototype memories as initialized",
            initialization_epoch=initialization_epoch,
            memory_state=self.get_memory_lifecycle_state(),
        )

    def maybe_initialize_memories_from_loader(
        self,
        train_loader: Any,
        device: str,
    ) -> bool:
        if self.memory_initialized:
            return False
        if self.current_epoch_index < self.bootstrap_encoder_epochs:
            return False
        self.memory_ready_for_initialization = True
        token_pool = self._collect_memory_initialization_token_pool_from_loader(
            train_loader,
            device,
        )
        hidden_tokens = token_pool["hidden_tokens"]
        if hidden_tokens.shape[0] == 0:
            console_print(
                "MODEL",
                "No normal hidden tokens were available for memory initialization",
                epoch=self.current_epoch_index + 1,
                num_batches_used=token_pool["num_batches_used"],
            )
            return False
        self._initialize_memory_buffers_from_token_pool(hidden_tokens)
        self.mark_memories_initialized(
            initialization_epoch=self.current_epoch_index + 1
        )
        console_print(
            "MODEL",
            "Initialized prototype memories from normal hidden tokens",
            epoch=self.current_epoch_index + 1,
            bootstrap_encoder_epochs=self.bootstrap_encoder_epochs,
            num_batches_used=token_pool["num_batches_used"],
            num_clean_tokens=token_pool["num_clean_tokens"],
            num_synthetic_normal_tokens=token_pool["num_synthetic_normal_tokens"],
        )
        return True

    def load_checkpoint_extra_state(self, extra_state: dict[str, Any] | None) -> None:
        if not extra_state:
            return
        self.memory_initialized = bool(
            extra_state.get("memory_initialized", self.memory_initialized)
        )
        self.memory_training_enabled = bool(
            extra_state.get("memory_training_enabled", self.memory_training_enabled)
        )
        self.memory_ready_for_initialization = bool(
            extra_state.get(
                "memory_ready_for_initialization",
                self.memory_ready_for_initialization,
            )
        )
        self.memory_initialization_epoch = extra_state.get(
            "memory_initialization_epoch",
            self.memory_initialization_epoch,
        )

    def _move_initialization_batch_to_device(
        self,
        batch: dict[str, Any],
        device: str,
    ) -> dict[str, Any]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _normalize_memory_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(vectors, dim=-1, eps=self.memory_norm_epsilon)

    def _normalize_hidden_for_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(hidden, dim=-1, eps=self.memory_norm_epsilon)

    def _select_covering_vectors(
        self,
        candidate_vectors: torch.Tensor,
        num_vectors: int,
    ) -> torch.Tensor:
        if candidate_vectors.shape[0] == 0:
            raise ValueError("candidate_vectors must contain at least one token")

        normalized_vectors = self._normalize_memory_vectors(candidate_vectors)
        if normalized_vectors.shape[0] <= num_vectors:
            repeated_indices = (
                torch.arange(
                    num_vectors,
                    device=normalized_vectors.device,
                )
                % normalized_vectors.shape[0]
            )
            return normalized_vectors.index_select(0, repeated_indices)

        mean_vector = normalized_vectors.mean(dim=0, keepdim=True)
        squared_distances_to_mean = torch.sum(
            (normalized_vectors - mean_vector) ** 2,
            dim=1,
        )
        first_index = int(torch.argmin(squared_distances_to_mean).item())
        selected_indices = [first_index]
        minimum_squared_distances = torch.sum(
            (normalized_vectors - normalized_vectors[first_index]) ** 2,
            dim=1,
        )

        while len(selected_indices) < num_vectors:
            next_index = int(torch.argmax(minimum_squared_distances).item())
            selected_indices.append(next_index)
            next_squared_distances = torch.sum(
                (normalized_vectors - normalized_vectors[next_index]) ** 2,
                dim=1,
            )
            minimum_squared_distances = torch.minimum(
                minimum_squared_distances,
                next_squared_distances,
            )

        selected_index_tensor = torch.tensor(
            selected_indices,
            device=normalized_vectors.device,
        )
        return normalized_vectors.index_select(0, selected_index_tensor)

    def _collect_memory_initialization_token_pool_from_loader(
        self,
        train_loader: Any,
        device: str,
    ) -> dict[str, Any]:
        clean_hidden_tokens: list[torch.Tensor] = []
        synthetic_normal_hidden_tokens: list[torch.Tensor] = []
        num_batches_used = 0
        previous_training_mode = self.training

        self.eval()
        with torch.no_grad():
            for batch_index, raw_batch in enumerate(train_loader):
                if batch_index >= self.memory_initialization_batches:
                    break
                num_batches_used += 1
                batch_on_device = self._move_initialization_batch_to_device(
                    raw_batch,
                    device,
                )
                clean_batch = self._prepare_clean_batch(
                    batch_on_device,
                    stage_name="memory_init",
                )
                clean_hidden = self.encoder(clean_batch)["hidden"].reshape(
                    -1,
                    self.hidden_dim,
                )
                clean_hidden_tokens.append(clean_hidden)

                if (
                    self.memory_initialization_with_synthetic_windows
                    and self.use_synthetic_augmentation
                ):
                    synthetic_batch = self.synthetic_anomaly_injector.augment_batch(
                        self._clone_batch(batch_on_device)
                    )
                    synthetic_hidden = self.encoder(synthetic_batch)["hidden"]
                    normal_time_step_mask = (
                        synthetic_batch["synthetic_anomaly_mask"] == 0
                    )
                    synthetic_normal_hidden = synthetic_hidden[normal_time_step_mask]
                    if synthetic_normal_hidden.numel() > 0:
                        synthetic_normal_hidden_tokens.append(synthetic_normal_hidden)

        self.train(previous_training_mode)

        hidden_token_groups = clean_hidden_tokens + synthetic_normal_hidden_tokens
        if hidden_token_groups:
            hidden_tokens = torch.cat(hidden_token_groups, dim=0)
        else:
            hidden_tokens = torch.empty(0, self.hidden_dim, device=device)

        return {
            "hidden_tokens": hidden_tokens,
            "num_batches_used": num_batches_used,
            "num_clean_tokens": sum(
                int(hidden_group.shape[0]) for hidden_group in clean_hidden_tokens
            ),
            "num_synthetic_normal_tokens": sum(
                int(hidden_group.shape[0])
                for hidden_group in synthetic_normal_hidden_tokens
            ),
        }

    def _initialize_memory_buffers_from_token_pool(
        self,
        hidden_tokens: torch.Tensor,
    ) -> None:
        if hidden_tokens.shape[0] == 0:
            raise ValueError("hidden_tokens must contain at least one normal token")

        if self.continuous_prototype_bank is not None:
            continuous_seed_vectors = self._select_covering_vectors(
                hidden_tokens,
                self.continuous_num_prototypes,
            )
            self.continuous_prototype_bank.copy_(continuous_seed_vectors)

        if self.discrete_codebook is not None:
            discrete_seed_vectors = self._select_covering_vectors(
                hidden_tokens.flip(0),
                self.discrete_codebook_size,
            )
            self.discrete_codebook.copy_(discrete_seed_vectors)
            if self.discrete_ema_counts is not None:
                self.discrete_ema_counts.fill_(1.0)
            if self.discrete_ema_sums is not None:
                self.discrete_ema_sums.copy_(discrete_seed_vectors)

    def _update_continuous_memory_bank(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.continuous_prototype_bank is None:
            raise ValueError("continuous_prototype_bank is not available")

        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        if token_mask is not None:
            selected_hidden = normalized_hidden[token_mask]
            if selected_hidden.numel() == 0:
                return self._normalize_memory_vectors(self.continuous_prototype_bank)
            normalized_hidden = selected_hidden.reshape(1, -1, self.hidden_dim)
        normalized_memory = self._normalize_memory_vectors(
            self.continuous_prototype_bank
        )

        # k prototypes with each prototype having h dimensions (containing h numbers)
        # b windows with each window having l timesteps,
        # with each timestep having h dimensions (containing h numbers)
        # k @ b.T is making each prototype attending to each timestep in a window
        # across all b windows in a batch.
        prototype_to_token_logits = torch.einsum(
            "kh,blh->kbl",
            normalized_memory,  # (n_continuous_prototypes, d_model)
            normalized_hidden,  # (batch_size, n_timesteps, d_model)
        ) / math.sqrt(
            self.hidden_dim
        )  # (n_continuous_prototypes, batch_size, n_timesteps)

        # n_continuous_prototypes là self.continuous_num_prototypes
        # d_model là h

        prototype_to_token_weights = torch.softmax(
            prototype_to_token_logits.reshape(self.continuous_num_prototypes, -1),
            # (n_continuous_prototypes, batch_size * n_timesteps)
            dim=-1,
        ).reshape_as(
            prototype_to_token_logits
        )  # (n_continuous_prototypes, batch_size, n_timesteps)

        weighted_hidden_summary = torch.einsum(
            "kbl,blh->kh",
            prototype_to_token_weights,
            normalized_hidden,
        )

        weighted_hidden_summary = self._normalize_memory_vectors(
            weighted_hidden_summary
        )

        gate_input = torch.cat(
            [normalized_memory, weighted_hidden_summary],
            dim=-1,
        )

        update_gate = self.continuous_update_gate(gate_input)

        updated_memory = (
            1.0 - update_gate
        ) * normalized_memory + update_gate * weighted_hidden_summary
        updated_memory = self._normalize_memory_vectors(updated_memory)

        with torch.no_grad():
            self.continuous_prototype_bank.copy_(updated_memory.detach())

        return updated_memory

    def _update_discrete_codebook_memory(
        self,
        hidden: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.discrete_assignment is None
            or self.discrete_codebook is None
            or self.discrete_ema_counts is None
            or self.discrete_ema_sums is None
        ):
            raise ValueError("discrete memory state is not available")

        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        if token_mask is not None:
            selected_hidden = normalized_hidden[token_mask]
            if selected_hidden.numel() == 0:
                assignment_logits = hidden.new_zeros(
                    hidden.shape[0],
                    hidden.shape[1],
                    self.discrete_codebook_size,
                )
                assignment_probabilities = torch.softmax(assignment_logits, dim=-1)
                return (
                    assignment_logits,
                    assignment_probabilities,
                    self._normalize_memory_vectors(self.discrete_codebook),
                )
            normalized_hidden = selected_hidden.reshape(1, -1, self.hidden_dim)
        assignment_logits = self.discrete_assignment(normalized_hidden)
        assignment_probabilities = F.gumbel_softmax(
            assignment_logits,
            tau=self.gumbel_temperature,
            hard=False,
            dim=-1,
        )
        flattened_probabilities = assignment_probabilities.reshape(
            -1,
            self.discrete_codebook_size,
        )
        flattened_hidden = normalized_hidden.reshape(-1, self.hidden_dim)
        batch_counts = flattened_probabilities.sum(dim=0)
        batch_sums = flattened_probabilities.T @ flattened_hidden

        with torch.no_grad():
            self.discrete_ema_counts.mul_(self.discrete_ema_decay).add_(
                (1.0 - self.discrete_ema_decay) * batch_counts.detach()
            )
            self.discrete_ema_sums.mul_(self.discrete_ema_decay).add_(
                (1.0 - self.discrete_ema_decay) * batch_sums.detach()
            )
            normalized_codebook = (
                self.discrete_ema_sums
                / self.discrete_ema_counts.clamp_min(
                    self.memory_norm_epsilon
                ).unsqueeze(-1)
            )
            normalized_codebook = self._normalize_memory_vectors(normalized_codebook)
            self.discrete_codebook.copy_(normalized_codebook)

        return (
            assignment_logits,
            assignment_probabilities,
            self._normalize_memory_vectors(self.discrete_codebook),
        )

    def prepare_realistic_validation_epoch(self, anomaly_probability: float) -> None:
        if not 0.0 <= float(anomaly_probability) <= 1.0:
            raise ValueError("realistic validation anomaly_probability must be in [0, 1]")
        self.synthetic_validation_injector.anomaly_probability = float(anomaly_probability)
        self.synthetic_validation_injector.reset_rng()

    def prepare_synthetic_validation_epoch(self) -> None:
        # Backward-compatible wrapper for older call sites.
        self.prepare_realistic_validation_epoch(
            anomaly_probability=self.synthetic_validation_injector.anomaly_probability
        )

    def _continuous_prototype_lookup(
        self,
        hidden: torch.Tensor,
        *,
        stage_name: str,
        active_memory_bank: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # The continuous branch keeps a soft weighted prototype mixture.
        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        continuous_hidden = normalized_hidden
        attention_logits = None
        attention_weights = None
        memory_bypass_active = self._should_bypass_memory_for_stage(stage_name)
        memory_bank_for_read = active_memory_bank

        if memory_bank_for_read is not None and not memory_bypass_active:
            attention_logits = torch.einsum(
                "blh,kh->blk",
                normalized_hidden,
                memory_bank_for_read,
            ) / math.sqrt(self.hidden_dim)
            attention_weights = torch.softmax(attention_logits, dim=-1)
            continuous_hidden = torch.einsum(
                "blk,kh->blh",
                attention_weights,
                memory_bank_for_read,
            )
            continuous_hidden = self._normalize_hidden_for_memory(continuous_hidden)

        return {
            "hidden": hidden,
            "prototype_context": continuous_hidden,
            "prototype_logits": attention_logits,
            "prototype_weights": attention_weights,
            "aux": {
                "branch_name": "continuous",
                "enabled": self.continuous_memory_enabled,
                "num_prototypes": self.continuous_num_prototypes,
                "memory_bypass_active": memory_bypass_active,
                "memory_initialized": self.memory_initialized,
            },
        }

    def _discrete_prototype_lookup(
        self,
        hidden: torch.Tensor,
        *,
        stage_name: str,
        active_codebook: torch.Tensor | None = None,
        precomputed_assignment_logits: torch.Tensor | None = None,
        precomputed_assignment_probabilities: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # The discrete branch keeps a quantized codebook view of the same tokens.
        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        discrete_hidden = normalized_hidden
        assignment_logits = precomputed_assignment_logits
        assignment_probabilities = precomputed_assignment_probabilities
        code_indices = None
        memory_bypass_active = self._should_bypass_memory_for_stage(stage_name)
        normalized_codebook = active_codebook
        if normalized_codebook is None and self.discrete_codebook is not None:
            normalized_codebook = self._normalize_memory_vectors(self.discrete_codebook)

        if normalized_codebook is not None and not memory_bypass_active:
            if assignment_logits is None or assignment_probabilities is None:
                if self.discrete_assignment is None:
                    raise ValueError("discrete_assignment is not available")
                assignment_logits = self.discrete_assignment(normalized_hidden)
                assignment_probabilities = F.gumbel_softmax(
                    assignment_logits,
                    tau=self.gumbel_temperature,
                    hard=False,
                    dim=-1,
                )
            discrete_hidden = torch.einsum(
                "blk,kh->blh",
                assignment_probabilities,
                normalized_codebook,
            )
            discrete_hidden = self._normalize_hidden_for_memory(discrete_hidden)
            code_indices = torch.argmax(assignment_probabilities, dim=-1)

        return {
            "hidden": hidden,
            "quantized_hidden": discrete_hidden,
            "assignment_logits": assignment_logits,
            "assignment_probabilities": assignment_probabilities,
            "code_indices": code_indices,
            "aux": {
                "branch_name": "discrete",
                "enabled": self.discrete_memory_enabled,
                "codebook_size": self.discrete_codebook_size,
                "temperature": self.gumbel_temperature,
                "memory_bypass_active": memory_bypass_active,
                "memory_initialized": self.memory_initialized,
            },
        }

    def _compute_fusion_outputs(
        self,
        continuous_hidden: torch.Tensor,
        discrete_hidden: torch.Tensor,
        *,
        base_hidden: torch.Tensor | None = None,
        paired_hidden: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # Fusion is expressed as exact limiting cases of the same equations.
        # That is why continuous-only and discrete-only ablations need no second model.
        if self.active_alpha_override is None:
            alpha_scalar = torch.sigmoid(self.alpha_logit)
        else:
            alpha_scalar = continuous_hidden.new_tensor(
                float(self.active_alpha_override)
            )
        if self.active_beta_override is None:
            beta_scalar = torch.sigmoid(self.beta_logit)
        else:
            beta_scalar = continuous_hidden.new_tensor(float(self.active_beta_override))

        alpha = alpha_scalar.expand(continuous_hidden.shape[0])
        beta = beta_scalar.expand(continuous_hidden.shape[0])
        cka_reconstruction = continuous_hidden.new_zeros(continuous_hidden.shape[0])
        cka_classification = continuous_hidden.new_zeros(continuous_hidden.shape[0])
        if (
            self.enable_cka_gated_fusion
            and base_hidden is not None
            and paired_hidden is not None
        ):
            cka_reconstruction = self._compute_batch_linear_cka_scores(
                base_hidden,
                continuous_hidden,
            )
            cka_classification = self._compute_batch_linear_cka_scores(
                paired_hidden,
                discrete_hidden,
            )
            cka_features = torch.stack(
                [cka_reconstruction, cka_classification],
                dim=-1,
            )
            alpha = torch.sigmoid(
                self.classification_fusion_gate(cka_features)
            ).squeeze(-1)
            beta = torch.sigmoid(self.reconstruction_fusion_gate(cka_features)).squeeze(
                -1
            )
        alpha_expanded = alpha.view(-1, 1, 1)
        beta_expanded = beta.view(-1, 1, 1)

        # Beta là mức độ mà tác vụ tái tạo chuỗi (reconstruction) sử dụng
        # nhánh các vec-tơ nguyên mẫu rời rạc (discrete prototype).
        # Mình kì vọng giá trị này sẽ nhỏ hơn alpha.
        hidden_reconstruction = (
            beta_expanded * discrete_hidden + (1.0 - beta_expanded) * continuous_hidden
        )

        # Alpha là mức độ mà tác vụ phân loại (classification) sử dụng nhánh các
        # vec-tơ nguyên mẫu rời rạc.
        # Mình kì vọng giá trị này sẽ lớn hơn beta.
        hidden_classification = (
            alpha_expanded * discrete_hidden
            + (1.0 - alpha_expanded) * continuous_hidden
        )

        return {
            "hidden_reconstruction": hidden_reconstruction,
            "hidden_classification": hidden_classification,
            "alpha": alpha,
            "beta": beta,
            "aux": {
                "fusion_mode": "learnable_sigmoid_scalars",
                "alpha": float(alpha.mean().detach().cpu()),
                "beta": float(beta.mean().detach().cpu()),
                "alpha_std": float(alpha.std(unbiased=False).detach().cpu()),
                "beta_std": float(beta.std(unbiased=False).detach().cpu()),
                "alpha_logit": float(alpha_scalar.detach().cpu()),
                "beta_logit": float(beta_scalar.detach().cpu()),
                "cka_reconstruction_mean": float(
                    cka_reconstruction.mean().detach().cpu()
                ),
                "cka_reconstruction_std": float(
                    cka_reconstruction.std(unbiased=False).detach().cpu()
                ),
                "cka_classification_mean": float(
                    cka_classification.mean().detach().cpu()
                ),
                "cka_classification_std": float(
                    cka_classification.std(unbiased=False).detach().cpu()
                ),
                "warmup_active": self.schedule_state["warmup_active"],
                "temperature": self.gumbel_temperature,
            },
        }

    def _compute_linear_cka_score(
        self,
        lhs_tokens: torch.Tensor,
        rhs_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if lhs_tokens.shape[0] <= 1:
            return lhs_tokens.new_zeros(())
        centered_lhs = lhs_tokens - lhs_tokens.mean(dim=0, keepdim=True)
        centered_rhs = rhs_tokens - rhs_tokens.mean(dim=0, keepdim=True)
        lhs_gram = centered_lhs @ centered_lhs.T
        rhs_gram = centered_rhs @ centered_rhs.T
        hsic_lhs_rhs = torch.sum(lhs_gram * rhs_gram)
        hsic_lhs_lhs = torch.sum(lhs_gram * lhs_gram)
        hsic_rhs_rhs = torch.sum(rhs_gram * rhs_gram)
        return hsic_lhs_rhs / torch.sqrt(
            hsic_lhs_lhs * hsic_rhs_rhs + self.cka_eps
        ).clamp_min(self.cka_eps)

    def _compute_batch_linear_cka_scores(
        self,
        lhs_hidden: torch.Tensor,
        rhs_hidden: torch.Tensor,
    ) -> torch.Tensor:
        sample_scores: list[torch.Tensor] = []
        for sample_index in range(lhs_hidden.shape[0]):
            sample_scores.append(
                self._compute_linear_cka_score(
                    lhs_hidden[sample_index],
                    rhs_hidden[sample_index],
                )
            )
        return torch.stack(sample_scores)

    def _compute_two_view_contrastive_loss(
        self,
        anchor_hidden: torch.Tensor,
        positive_hidden: torch.Tensor,
        synthetic_anomaly_mask: torch.Tensor,
    ) -> torch.Tensor:
        normal_token_mask = (synthetic_anomaly_mask == 0).reshape(-1)
        if int(normal_token_mask.sum().item()) == 0:
            return self._zero_loss(anchor_hidden)
        anchor_tokens = anchor_hidden.reshape(-1, self.hidden_dim)[normal_token_mask]
        positive_tokens = positive_hidden.reshape(-1, self.hidden_dim)[
            normal_token_mask
        ]
        normalized_anchors = F.normalize(anchor_tokens, dim=-1, eps=self.epsilon)
        normalized_positives = F.normalize(positive_tokens, dim=-1, eps=self.epsilon)
        logits = (normalized_anchors @ normalized_positives.T) / max(
            self.contrastive_temperature, self.epsilon
        )
        targets = torch.arange(
            logits.shape[0],
            device=logits.device,
            dtype=torch.long,
        )
        return F.cross_entropy(logits, targets)

    def _clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = [
                    dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                cloned_batch[key] = value
        return cloned_batch

    def _prepare_clean_batch(
        self, batch: dict[str, Any], stage_name: str
    ) -> dict[str, Any]:
        # The model owns augmentation timing because synthetic supervision is
        # part of the multitask objective, not a separate preprocessing pipeline.
        if (
            "classification_labels" in batch
            and "synthetic_anomaly_mask" in batch
            and "augmentation_metadata" in batch
        ):
            console_print(
                stage_name.upper(),
                "Received pre-augmented multitask batch",
                **summarize_batch(batch),
                classification_label_distribution=summarize_label_distribution(
                    batch["classification_labels"]
                ),
            )
            return self._clone_batch(batch)

        if stage_name == "train" and self.use_synthetic_augmentation:
            return self.synthetic_anomaly_injector.augment_batch(batch)

        prepared_batch = self._clone_batch(batch)
        batch_size, window_size, _ = prepared_batch["x"].shape
        prepared_batch["classification_labels"] = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["classification_class_names"] = (
            self._classification_class_names()
        )
        prepared_batch["synthetic_anomaly_mask"] = torch.zeros(
            batch_size,
            window_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["augmentation_metadata"] = [
            {
                "is_synthetic_anomaly": False,
                "anomaly_family": "clean",
                "anomaly_family_index": None,
                "start_index": None,
                "end_index": None,
                "affected_channels": [],
                "family_parameters_by_channel": {},
            }
            for _ in range(batch_size)
        ]
        if prepared_batch["point_labels"] is None:
            prepared_batch["point_labels"] = prepared_batch[
                "synthetic_anomaly_mask"
            ].clone()
        console_print(
            stage_name.upper(),
            "Prepared clean multitask batch",
            **summarize_batch(prepared_batch),
            classification_label_distribution=summarize_label_distribution(
                prepared_batch["classification_labels"]
            ),
        )
        return prepared_batch

    def _prepare_batch(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        if stage_name in {"val_synth", "val_realistic"} and self.use_synthetic_validation:
            return self.synthetic_validation_injector.augment_batch(batch)
        return self._prepare_clean_batch(batch, stage_name)

    def _prepare_contrastive_pair_batches(
        self,
        batch: dict[str, Any],
        stage_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not self.enable_two_view_contrastive:
            return None
        if stage_name not in {"train", "val_synth", "val_realistic"}:
            return None
        clean_batch = self._prepare_clean_batch(
            self._clone_batch(batch), stage_name="val"
        )
        augmented_batch = self._prepare_batch(self._clone_batch(batch), stage_name)
        return clean_batch, augmented_batch

    def forward(
        self,
        batch: dict[str, Any],
        stage_name: str = "train",
    ) -> dict[str, Any]:
        """
        Hàm forward này là hàm tính toán quan trọng nhất của mô hình
        trong giai đoạn offline pre-training theo proposal của phương pháp.
        """

        # The forward pass is the main representation story of the thesis model:
        # encode once, build two branch views, fuse per task, then score.
        validate_batch(batch)
        console_print(
            "MODEL", "Multitask forward input batch", **summarize_batch(batch)
        )
        forward_start_time = time.perf_counter()

        # Truyền lô dữ liệu qua encoder để mã hoá
        # Thu được các vec-tơ ẩn (hidden vector)
        # Một vec-tơ ẩn cho mỗi bước thời gian (timestep)
        encoder_outputs = self.encoder(batch)
        hidden = encoder_outputs["hidden"]
        anomaly_mask = batch.get("synthetic_anomaly_mask")
        normal_token_mask = None
        anomaly_token_mask = None
        if anomaly_mask is not None and self.enable_two_view_contrastive:
            normal_token_mask = anomaly_mask == 0
            anomaly_token_mask = anomaly_mask == 1
        if self.continuous_prototype_bank is not None and self._should_update_memory(
            stage_name
        ):
            active_continuous_memory_bank = self._update_continuous_memory_bank(
                hidden,
                token_mask=normal_token_mask,
            )
        elif self.continuous_prototype_bank is not None:
            active_continuous_memory_bank = self._normalize_memory_vectors(
                self.continuous_prototype_bank
            )
        else:
            active_continuous_memory_bank = None
        if self.discrete_codebook is not None and self._should_update_memory(
            stage_name
        ):
            self._update_discrete_codebook_memory(
                hidden,
                token_mask=anomaly_token_mask,
            )
            active_assignment_logits = None
            active_assignment_probabilities = None
            active_discrete_codebook = self._normalize_memory_vectors(
                self.discrete_codebook
            )
        elif self.discrete_codebook is not None:
            active_assignment_logits = None
            active_assignment_probabilities = None
            active_discrete_codebook = self._normalize_memory_vectors(
                self.discrete_codebook
            )
        else:
            active_assignment_logits = None
            active_assignment_probabilities = None
            active_discrete_codebook = None

        # Tái tạo các vec-tơ ẩn sử dụng các vec-tơ nguyên mẫu liên tục
        continuous_outputs = self._continuous_prototype_lookup(
            hidden,
            stage_name=stage_name,
            active_memory_bank=active_continuous_memory_bank,
        )

        # Tái tạo các vec-tơ ẩn sử dụng các vec-tơ nguyên mẫu rời rạc
        discrete_outputs = self._discrete_prototype_lookup(
            hidden,
            stage_name=stage_name,
            active_codebook=active_discrete_codebook,
            precomputed_assignment_logits=active_assignment_logits,
            precomputed_assignment_probabilities=active_assignment_probabilities,
        )

        # Kết hợp vec-tơ ẩn từ hai nhánh lại với nhau
        fusion_outputs = self._compute_fusion_outputs(
            continuous_hidden=continuous_outputs["prototype_context"],
            discrete_hidden=discrete_outputs["quantized_hidden"],
            base_hidden=hidden,
            paired_hidden=batch.get("paired_hidden_for_fusion"),
        )

        # Lấy ra vec-tơ kết hợp dùng cho từng tác vụ
        hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
        hidden_classification = fusion_outputs["hidden_classification"]

        # Truyền vec-tơ kết hợp dùng cho
        # tác vụ tái tạo qua mạng tái tạo (reconstruction head)
        recon = self.reconstruction_head(hidden_reconstruction)

        # Xét vec-tơ kết hợp dùng cho tác vụ phân loại.
        # RedLamp dùng toàn bộ các vec-tơ ẩn theo thời gian bằng cách trải phẳng
        # cửa sổ thành một biểu diễn cấp-window trước classifier head.
        if hidden_classification.shape[1] != self.window_size:
            raise ValueError(
                "hidden_classification must have window dimension "
                f"{self.window_size}, but received {hidden_classification.shape[1]}"
            )
        flattened_classification_hidden = hidden_classification.reshape(
            hidden_classification.shape[0],
            self.window_size * self.hidden_dim,
        )
        logits = None
        class_probabilities = None
        if self.enable_classification_path:
            logits = self.classification_head(flattened_classification_hidden)
            class_probabilities = torch.softmax(logits, dim=-1)

        # Độ bất thường được tính bằng cách
        # Tính toán độ lỗi (error) giữa bản gốc và bản tái tạo
        # ở từng bước thời gian.
        # Xong, bình phương tất cả độ lỗi lên rồi cộng lại chia lấy trung bình.
        point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)

        # Chuẩn bị đầu ra theo thoả thuận (contract) đã được thiết lập.
        # Xem: src/core/contracts.py
        outputs = {
            "hidden": hidden,
            "pooled": flattened_classification_hidden,
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {
                "encoder": encoder_outputs["aux"],
                "continuous_branch": continuous_outputs,
                "discrete_branch": discrete_outputs,
                "fusion": fusion_outputs["aux"],
                "active_continuous_memory_bank": active_continuous_memory_bank,
                "active_discrete_codebook": active_discrete_codebook,
                "hidden_reconstruction": hidden_reconstruction,
                "hidden_classification": hidden_classification,
                "alpha": fusion_outputs["alpha"],
                "beta": fusion_outputs["beta"],
                "class_probabilities": class_probabilities,
                "classification_class_names": self._classification_class_names(),
                "memory": self.get_memory_lifecycle_state(),
                "forward_pass_seconds": time.perf_counter() - forward_start_time,
            },
        }
        validate_model_outputs(outputs)
        console_print(
            "MODEL",
            "Multitask forward outputs",
            hidden=summarize_tensor(outputs["hidden"]),
            pooled=summarize_tensor(outputs["pooled"]),
            recon=summarize_tensor(outputs["recon"]),
            logits=summarize_tensor(outputs["logits"]),
            point_scores=summarize_tensor(outputs["point_scores"]),
            window_scores=summarize_tensor(outputs["window_scores"]),
            continuous_hidden=summarize_tensor(
                outputs["aux"]["continuous_branch"]["prototype_context"]
            ),
            discrete_hidden=summarize_tensor(
                outputs["aux"]["discrete_branch"]["quantized_hidden"]
            ),
            assignment_probabilities=summarize_tensor(
                outputs["aux"]["discrete_branch"]["assignment_probabilities"]
            ),
            hidden_reconstruction=summarize_tensor(
                outputs["aux"]["hidden_reconstruction"]
            ),
            hidden_classification=summarize_tensor(
                outputs["aux"]["hidden_classification"]
            ),
            forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
        )
        return outputs

    def _normalize_branch_tokens(
        self, branch_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_hidden = self.branch_layer_norm(branch_hidden).reshape(
            -1, self.hidden_dim
        )
        feature_mean = normalized_hidden.mean(dim=0, keepdim=True)
        feature_std = normalized_hidden.std(dim=0, unbiased=False, keepdim=True)
        standardized_hidden = (normalized_hidden - feature_mean) / (
            feature_std + self.epsilon
        )
        return normalized_hidden, standardized_hidden

    def _compute_reconstruction_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        squared_reconstruction_error = (outputs["recon"] - batch["x"]) ** 2
        if not self.reconstruction_normal_only or "synthetic_anomaly_mask" not in batch:
            return torch.mean(squared_reconstruction_error)

        normal_time_step_mask = self._build_normal_time_step_mask(
            batch, squared_reconstruction_error
        )
        expanded_normal_mask = normal_time_step_mask.unsqueeze(-1).expand_as(
            squared_reconstruction_error
        )
        active_normal_cells = torch.count_nonzero(expanded_normal_mask)
        if int(active_normal_cells.item()) == 0:
            return torch.mean(squared_reconstruction_error)

        return (
            torch.sum(squared_reconstruction_error * expanded_normal_mask)
            / expanded_normal_mask.sum()
        )

    def _compute_reconstruction_diagnostics(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> dict[str, float]:
        squared_reconstruction_error = (outputs["recon"] - batch["x"]) ** 2
        per_window_error = squared_reconstruction_error.mean(dim=(1, 2))
        diagnostics = {
            "recon_mse_mean": float(torch.mean(per_window_error).detach().cpu()),
            "recon_mse_std": float(
                torch.std(per_window_error, unbiased=False).detach().cpu()
            ),
            "active_normal_cells": float(squared_reconstruction_error.numel()),
            "normal_cell_ratio": 1.0,
            "synthetic_cell_ratio": 0.0,
            "fallback_to_full_mse_flag": 0.0,
        }
        if "synthetic_anomaly_mask" not in batch:
            return diagnostics

        normal_time_step_mask = self._build_normal_time_step_mask(
            batch, squared_reconstruction_error
        )
        expanded_normal_mask = normal_time_step_mask.unsqueeze(-1).expand_as(
            squared_reconstruction_error
        )
        active_normal_cells = float(torch.count_nonzero(expanded_normal_mask).item())
        total_cells = float(expanded_normal_mask.numel())
        normal_cell_ratio = active_normal_cells / max(total_cells, 1.0)
        synthetic_cell_ratio = 1.0 - normal_cell_ratio
        fallback_to_full_mse_flag = 0.0
        if self.reconstruction_normal_only and active_normal_cells <= 0.0:
            fallback_to_full_mse_flag = 1.0

        diagnostics["active_normal_cells"] = active_normal_cells
        diagnostics["normal_cell_ratio"] = normal_cell_ratio
        diagnostics["synthetic_cell_ratio"] = synthetic_cell_ratio
        diagnostics["fallback_to_full_mse_flag"] = fallback_to_full_mse_flag
        return diagnostics

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if outputs.get("logits") is None:
            return self._zero_loss(outputs["recon"])
        if self.use_label_refurbishment:
            target_probabilities = self._build_refurbished_classification_targets(
                batch["classification_labels"],
                outputs["logits"].dtype,
            )
            log_probabilities = F.log_softmax(outputs["logits"], dim=-1)
            return torch.mean(
                torch.sum(-target_probabilities * log_probabilities, dim=-1)
            )

        return F.cross_entropy(outputs["logits"], batch["classification_labels"].long())

    def _build_normal_time_step_mask(
        self,
        batch: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        anomaly_mask = batch["synthetic_anomaly_mask"].to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        normal_time_step_mask = 1.0 - anomaly_mask
        return torch.clamp(normal_time_step_mask, min=0.0, max=1.0)

    def _build_refurbished_classification_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        hard_labels = classification_labels.long()
        target_probabilities = F.one_hot(
            hard_labels,
            num_classes=self.num_classes,
        ).to(dtype=target_dtype)

        if self.classification_label_mode == "binary":
            if self.num_classes != 2:
                raise ValueError("Binary label refurbishment requires num_classes == 2")
            target_probabilities[:, 0] = torch.where(
                hard_labels == 0,
                1.0 - self.refurbishment_beta,
                self.refurbishment_alpha,
            )
            target_probabilities[:, 1] = torch.where(
                hard_labels == 0,
                self.refurbishment_beta,
                1.0 - self.refurbishment_alpha,
            )
            return target_probabilities

        target_probabilities = torch.where(
            target_probabilities > 0.0,
            1.0
            - (
                self.refurbishment_alpha
                + self.refurbishment_beta * self.num_classes
                - self.refurbishment_beta
            ),
            self.refurbishment_beta,
        )
        target_probabilities[:, 0] = target_probabilities[:, 0] + (
            self.refurbishment_alpha
        )
        return target_probabilities / target_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.epsilon)

    def _build_refurbished_binary_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._build_refurbished_classification_targets(
            classification_labels=classification_labels,
            target_dtype=target_dtype,
        )

    def _compute_cross_branch_diversity_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        # This discourages the two branches from collapsing onto the same signal.
        continuous_hidden = outputs["aux"]["continuous_branch"]["prototype_context"]
        discrete_hidden = outputs["aux"]["discrete_branch"]["quantized_hidden"]
        _, standardized_continuous = self._normalize_branch_tokens(continuous_hidden)
        _, standardized_discrete = self._normalize_branch_tokens(discrete_hidden)
        num_tokens = standardized_continuous.shape[0]
        cross_branch_correlation = (
            standardized_continuous.T @ standardized_discrete / num_tokens
        )
        return cross_branch_correlation.pow(2).mean()

    def _compute_variance_floor_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        variance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context"
                if branch_name == "continuous_branch"
                else "quantized_hidden"
            ]
            normalized_hidden, _ = self._normalize_branch_tokens(branch_hidden)
            feature_std = normalized_hidden.std(dim=0, unbiased=False)
            variance_losses.append(
                F.relu(self.variance_floor_gamma - feature_std).pow(2).mean()
            )
        return torch.stack(variance_losses).sum()

    def _compute_covariance_reduction_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        covariance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context"
                if branch_name == "continuous_branch"
                else "quantized_hidden"
            ]
            _, standardized_hidden = self._normalize_branch_tokens(branch_hidden)
            num_tokens = standardized_hidden.shape[0]
            covariance_matrix = standardized_hidden.T @ standardized_hidden / num_tokens
            diagonal_matrix = torch.diag(torch.diag(covariance_matrix))
            off_diagonal_matrix = covariance_matrix - diagonal_matrix
            if self.hidden_dim == 1:
                covariance_losses.append(self._zero_loss(branch_hidden))
            else:
                covariance_losses.append(
                    off_diagonal_matrix.pow(2).sum()
                    / (self.hidden_dim * (self.hidden_dim - 1))
                )
        return torch.stack(covariance_losses).sum()

    # Version 1 with pure simple loss (only classification and reconstruction) was trained.
    # Based on real diagnostic, this usage loss will be added to prevent over-centralizing in few discrete prototypes.
    # So in version 2, the final loss function has 3 terms: one for classification, one for reconstruction
    # and one for usage of discrete prototypes.
    def _compute_prototype_usage_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        # Usage balancing is the main protection against dead or ignored codes.
        assignment_probabilities = outputs["aux"]["discrete_branch"][
            "assignment_probabilities"
        ]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            return self._zero_loss(outputs["hidden"])
        average_usage = assignment_probabilities.mean(dim=(0, 1))
        target_usage = torch.full_like(average_usage, 1.0 / self.discrete_codebook_size)
        return torch.sum((average_usage - target_usage) ** 2)

    def _compute_gate_regularization_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        # Gate entropy regularization keeps the fusion scalars from collapsing
        # too confidently unless the data actually supports that decision.
        alpha = outputs["aux"]["alpha"]
        beta = outputs["aux"]["beta"]
        max_entropy = math.log(2.0)
        alpha_clamped = torch.clamp(alpha, self.epsilon, 1.0 - self.epsilon)
        beta_clamped = torch.clamp(beta, self.epsilon, 1.0 - self.epsilon)
        alpha_entropy = -(
            alpha_clamped * torch.log(alpha_clamped)
            + (1.0 - alpha_clamped) * torch.log(1.0 - alpha_clamped)
        )
        beta_entropy = -(
            beta_clamped * torch.log(beta_clamped)
            + (1.0 - beta_clamped) * torch.log(1.0 - beta_clamped)
        )
        alpha_penalty = 1.0 - alpha_entropy / max_entropy
        beta_penalty = 1.0 - beta_entropy / max_entropy
        return 0.5 * (alpha_penalty + beta_penalty).mean()

    def _compute_optional_loss_terms(
        self, outputs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        optional_loss_values: dict[str, torch.Tensor] = {}
        for loss_name, loss_config in self.optional_loss_configs.items():
            compute_fn: Callable[[dict[str, Any]], torch.Tensor] = loss_config[
                "compute_fn"
            ]
            if loss_config["enabled"]:
                optional_loss_values[loss_name] = compute_fn(outputs)
            else:
                optional_loss_values[loss_name] = self._zero_loss(outputs["hidden"])
        return optional_loss_values

    def _compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
        optional_loss_values: dict[str, torch.Tensor],
        classification_weight: float,
    ) -> torch.Tensor:
        # The weighted sum is intentionally explicit so readers can map each
        # `lambda_*` config field directly to one line of the objective. The
        # default beginning of training is still the small objective
        # `L_recon + lambda_cls * L_cls`.
        total_loss = reconstruction_loss + classification_weight * classification_loss
        for loss_name, loss_value in optional_loss_values.items():
            loss_weight = self.optional_loss_configs[loss_name]["weight"]
            if loss_name == "usage_loss":
                loss_weight = self.current_usage_lambda
            total_loss = total_loss + loss_weight * loss_value
        return total_loss

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        loss_terms: dict[str, torch.Tensor],
        batch: dict[str, Any],
        *,
        include_classification_metrics: bool,
    ) -> dict[str, float]:
        # These logs are part of the branch-collapse observability surface, not
        # just convenience metrics. They are meant to support ablation reading.
        assignment_probabilities = outputs["aux"]["discrete_branch"][
            "assignment_probabilities"
        ]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            discrete_usage_top1 = 0.0
            discrete_usage_entropy = 0.0
            discrete_usage_concentration = 0.0
            discrete_usage_active_codes = 0.0
        else:
            average_usage = assignment_probabilities.mean(dim=(0, 1))
            average_usage = average_usage / average_usage.sum().clamp_min(self.epsilon)
            discrete_usage_top1 = float(average_usage.max().detach().cpu())
            discrete_usage_entropy = float(
                (
                    -(
                        average_usage * torch.log(average_usage.clamp_min(self.epsilon))
                    ).sum()
                )
                .detach()
                .cpu()
            )
            discrete_usage_concentration = float(
                torch.sum(average_usage.pow(2)).detach().cpu()
            )
            discrete_usage_active_codes = float(
                torch.sum(
                    (
                        average_usage > (1.0 / max(self.discrete_codebook_size * 2, 1))
                    ).float()
                )
                .detach()
                .cpu()
            )
        stage_log = {
            f"{stage_name}_loss": float(loss_terms["total_loss"].detach().cpu()),
            f"{stage_name}_reconstruction_loss": float(
                loss_terms["reconstruction_loss"].detach().cpu()
            ),
            f"{stage_name}_diversity_loss": float(
                loss_terms["diversity_loss"].detach().cpu()
            ),
            f"{stage_name}_variance_loss": float(
                loss_terms["variance_loss"].detach().cpu()
            ),
            f"{stage_name}_covariance_loss": float(
                loss_terms["covariance_loss"].detach().cpu()
            ),
            f"{stage_name}_usage_loss": float(loss_terms["usage_loss"].detach().cpu()),
            f"{stage_name}_gate_loss": float(loss_terms["gate_loss"].detach().cpu()),
            f"{stage_name}_contrastive_loss": float(
                loss_terms["contrastive_loss"].detach().cpu()
            ),
            f"{stage_name}_alpha": float(outputs["aux"]["alpha"].mean().detach().cpu()),
            f"{stage_name}_beta": float(outputs["aux"]["beta"].mean().detach().cpu()),
            f"{stage_name}_alpha_std": float(
                outputs["aux"]["alpha"].std(unbiased=False).detach().cpu()
            ),
            f"{stage_name}_beta_std": float(
                outputs["aux"]["beta"].std(unbiased=False).detach().cpu()
            ),
            f"{stage_name}_continuous_norm": float(
                outputs["aux"]["continuous_branch"]["prototype_context"]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            f"{stage_name}_discrete_norm": float(
                outputs["aux"]["discrete_branch"]["quantized_hidden"]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            f"{stage_name}_discrete_usage_top1": discrete_usage_top1,
            f"{stage_name}_discrete_usage_entropy": discrete_usage_entropy,
            f"{stage_name}_discrete_usage_concentration": discrete_usage_concentration,
            f"{stage_name}_discrete_usage_active_codes": discrete_usage_active_codes,
            f"{stage_name}_temperature": float(self.gumbel_temperature),
            f"{stage_name}_usage_lambda": float(self.current_usage_lambda),
            f"{stage_name}_warmup_active": float(self.schedule_state["warmup_active"]),
            f"{stage_name}_memory_initialized": float(
                outputs["aux"]["memory"]["memory_initialized"]
            ),
            f"{stage_name}_memory_training_enabled": float(
                outputs["aux"]["memory"]["memory_training_enabled"]
            ),
            f"{stage_name}_memory_ready_for_initialization": float(
                outputs["aux"]["memory"]["memory_ready_for_initialization"]
            ),
            f"{stage_name}_memory_mode": float(
                outputs["aux"]["memory"]["train_memory_mode"]
            ),
        }
        if stage_name in {"train", "val_synth", "val_realistic"}:
            cka_reconstruction_mean = float(
                outputs["aux"]["fusion"]["cka_reconstruction_mean"]
            )
            cka_reconstruction_std = float(
                outputs["aux"]["fusion"]["cka_reconstruction_std"]
            )
            stage_log[f"{stage_name}_cka_reconstruction_mean"] = cka_reconstruction_mean
            stage_log[f"{stage_name}_cka_reconstruction_std"] = cka_reconstruction_std
            stage_log[f"diag/cka/{stage_name}_reconstruction_mean"] = (
                cka_reconstruction_mean
            )
            stage_log[f"diag/cka/{stage_name}_reconstruction_std"] = (
                cka_reconstruction_std
            )
            if self.enable_classification_path:
                cka_classification_mean = float(
                    outputs["aux"]["fusion"]["cka_classification_mean"]
                )
                cka_classification_std = float(
                    outputs["aux"]["fusion"]["cka_classification_std"]
                )
                stage_log[f"{stage_name}_cka_classification_mean"] = (
                    cka_classification_mean
                )
                stage_log[f"{stage_name}_cka_classification_std"] = (
                    cka_classification_std
                )
                stage_log[f"diag/cka/{stage_name}_classification_mean"] = (
                    cka_classification_mean
                )
                stage_log[f"diag/cka/{stage_name}_classification_std"] = (
                    cka_classification_std
                )
        if include_classification_metrics and outputs.get("logits") is not None:
            predicted_labels = torch.argmax(outputs["logits"], dim=-1)
            classification_accuracy = float(
                (predicted_labels == batch["classification_labels"])
                .float()
                .mean()
                .detach()
                .cpu()
            )
            stage_log[f"{stage_name}_classification_loss"] = float(
                loss_terms["classification_loss"].detach().cpu()
            )
            stage_log[f"{stage_name}_classification_accuracy"] = classification_accuracy
        reconstruction_diagnostics = self._compute_reconstruction_diagnostics(
            outputs=outputs,
            batch=batch,
        )
        for metric_name, metric_value in reconstruction_diagnostics.items():
            stage_log[f"diag/recon/{stage_name}_{metric_name}"] = metric_value
        return stage_log

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage_name: str,
        *,
        classification_weight: float,
        include_classification_metrics: bool,
    ) -> dict[str, Any]:
        # This is the one place where the actual multitask training objective is assembled.

        # Chuẩn bị batch dữ liệu nghĩa là tải các mẫu dữ liệu lên từ
        # dataset và tiêm bất thường nhân tạo vào nếu cần.
        contrastive_pair = self._prepare_contrastive_pair_batches(batch, stage_name)
        if contrastive_pair is None:
            prepared_batch = self._prepare_batch(batch, stage_name)
            contrastive_loss = self._zero_loss(prepared_batch["x"])
        else:
            clean_batch, augmented_batch = contrastive_pair
            clean_outputs = self.forward(clean_batch, stage_name="val")
            prepared_batch = augmented_batch
            prepared_batch["paired_hidden_for_fusion"] = clean_outputs[
                "hidden"
            ].detach()
            contrastive_loss = self._compute_two_view_contrastive_loss(
                anchor_hidden=clean_outputs["hidden"],
                positive_hidden=self.encoder(prepared_batch)["hidden"],
                synthetic_anomaly_mask=prepared_batch["synthetic_anomaly_mask"],
            )

        # Đưa các mẫu dữ liệu qua mạng để tính toán ra kết quả
        outputs = self.forward(prepared_batch, stage_name=stage_name)

        # Tính toán các hàm loss thành phần
        reconstruction_loss = self._compute_reconstruction_loss(outputs, prepared_batch)
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        optional_loss_values = self._compute_optional_loss_terms(outputs)

        # Tính toán hàm loss tổng
        total_loss = self._compute_total_loss(
            reconstruction_loss=reconstruction_loss,
            classification_loss=classification_loss,
            optional_loss_values=optional_loss_values,
            classification_weight=(
                classification_weight if self.enable_classification_path else 0.0
            ),
        )
        total_loss = total_loss + self.lambda_contrastive * contrastive_loss

        loss_terms = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            "contrastive_loss": contrastive_loss,
            **optional_loss_values,
        }
        console_print(
            stage_name.upper(),
            "Completed multitask stage step",
            batch_size=prepared_batch["x"].shape[0],
            total_loss=float(total_loss.detach().cpu()),
            reconstruction_loss=float(reconstruction_loss.detach().cpu()),
            classification_loss=float(classification_loss.detach().cpu()),
            diversity_loss=float(optional_loss_values["diversity_loss"].detach().cpu()),
            variance_loss=float(optional_loss_values["variance_loss"].detach().cpu()),
            covariance_loss=float(
                optional_loss_values["covariance_loss"].detach().cpu()
            ),
            usage_loss=float(optional_loss_values["usage_loss"].detach().cpu()),
            gate_loss=float(optional_loss_values["gate_loss"].detach().cpu()),
            contrastive_loss=float(contrastive_loss.detach().cpu()),
            classification_label_distribution=(
                summarize_label_distribution(prepared_batch["classification_labels"])
                if self.enable_classification_path
                else {}
            ),
            alpha=float(outputs["aux"]["alpha"].mean().detach().cpu()),
            beta=float(outputs["aux"]["beta"].mean().detach().cpu()),
            forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
        )
        return {
            "loss": total_loss,
            "log": self._build_stage_log(
                stage_name,
                outputs,
                loss_terms,
                prepared_batch,
                include_classification_metrics=include_classification_metrics,
            ),
            "outputs": outputs,
            "loss_terms": loss_terms,
            "batch": prepared_batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="train",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val",
            classification_weight=0.0,
            include_classification_metrics=False,
        )

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val_synth",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def realistic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val_realistic",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="test",
            classification_weight=0.0,
            include_classification_metrics=False,
        )

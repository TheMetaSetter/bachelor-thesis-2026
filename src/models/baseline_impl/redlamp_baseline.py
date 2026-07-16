from __future__ import annotations

"""Self-contained RedLamp-inspired baseline.

The baseline keeps the repository batch and output contracts while using a
timestep encoder for a controlled comparison against the thesis model. It
remains a simple autoencoder and multi-class synthetic anomaly classifier
without prototype memory, fusion gates, or online adaptation state.
"""

from typing import Any
from collections import OrderedDict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import (
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    SyntheticAnomalyInjector,
)
from src.models.base_model import BaseModel
from src.models.neural_blocks import SimpleWindowCnnEncoder, build_multilayer_perceptron


from src.models.baseline_impl import redlamp_baseline_helpers as redlamp_helpers


class RedLampBaseline(BaseModel):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        latent_dim: int = 128,
        encoder_family: str = "mlp",
        mlp_num_linear_layers: int = 3,
        cnn_num_layers: int = 3,
        cnn_kernel_size: int = 3,
        cnn_hidden_channels: int = 64,
        cnn_dropout: float | None = None,
        classifier_dim: int = 32,
        num_classes: int = len(REDLAMP_MULTICLASS_CLASS_NAMES),
        dropout: float = 0.1,
        lambda_recon: float = 0.9,
        lambda_cls: float = 0.1,
        use_label_refurbishment: bool = True,
        refurbishment_alpha: float = 0.1,
        refurbishment_beta: float = 0.01,
        anomaly_probability: float = 0.5,
        min_segment_fraction: float = 0.2,
        max_segment_fraction: float = 0.3,
        spike_scale: float = 3.0,
        anomaly_visibility_boost: float = 1.5,
        anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
        use_synthetic_augmentation: bool = True,
        use_synthetic_validation: bool = True,
        synthetic_train_seed: int | None = None,
        synthetic_validation_seed: int = 7,
        classification_label_mode: str = "redlamp_multiclass",
        train_balance_classes: bool = True,
        balance_classes_within_batch: bool | None = None,
        balance_binary_classes_within_batch: bool = False,
        enable_gradient_conflict_profiling: bool = False,
        gradient_profiling_scope: str = "encoder_all",
        gradient_focus_layer_name: str = "encoder_last_affine",
        gradient_log_every_n_steps: int = 1,
        gradient_ema_alpha: float = 0.1,
        gradient_sma_window: int = 50,
        gradient_profile_include_bias: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()
        del unused_kwargs
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if classifier_dim <= 0:
            raise ValueError("classifier_dim must be positive")
        if encoder_family not in {"mlp", "cnn_simple"}:
            raise ValueError("encoder_family must be one of: mlp, cnn_simple")
        if mlp_num_linear_layers < 2:
            raise ValueError("mlp_num_linear_layers must be at least 2")
        if cnn_num_layers < 2:
            raise ValueError("cnn_num_layers must be at least 2")
        if cnn_kernel_size <= 0:
            raise ValueError("cnn_kernel_size must be positive")
        if cnn_hidden_channels <= 0:
            raise ValueError("cnn_hidden_channels must be positive")
        if cnn_dropout is not None and not 0.0 <= cnn_dropout <= 1.0:
            raise ValueError("cnn_dropout must be between 0 and 1")
        if classification_label_mode != "redlamp_multiclass":
            raise ValueError(
                "RedLampBaseline supports classification_label_mode="
                "'redlamp_multiclass'"
            )
        if gradient_profiling_scope not in {"encoder_all"}:
            raise ValueError("gradient_profiling_scope must be one of {'encoder_all'}")
        if gradient_focus_layer_name not in {
            "encoder_last_linear",
            "encoder_last_affine",
        }:
            raise ValueError(
                "gradient_focus_layer_name must be one of: "
                "encoder_last_linear, encoder_last_affine"
            )
        if gradient_log_every_n_steps < 1:
            raise ValueError("gradient_log_every_n_steps must be >= 1")
        if gradient_sma_window < 1:
            raise ValueError("gradient_sma_window must be >= 1")
        if not (0.0 < gradient_ema_alpha <= 1.0):
            raise ValueError("gradient_ema_alpha must satisfy 0 < alpha <= 1")
        if lambda_recon < 0.0:
            raise ValueError("lambda_recon must be non-negative")
        if lambda_cls < 0.0:
            raise ValueError("lambda_cls must be non-negative")

        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.encoder_family = encoder_family
        self.mlp_num_linear_layers = mlp_num_linear_layers
        self.cnn_num_layers = cnn_num_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_hidden_channels = cnn_hidden_channels
        self.cnn_dropout = dropout if cnn_dropout is None else cnn_dropout
        self.classifier_dim = classifier_dim
        self.num_classes = num_classes
        self.lambda_recon = lambda_recon
        self.lambda_cls = lambda_cls
        self.use_label_refurbishment = use_label_refurbishment
        self.refurbishment_alpha = refurbishment_alpha
        self.refurbishment_beta = refurbishment_beta
        self.use_synthetic_augmentation = use_synthetic_augmentation
        self.use_synthetic_validation = use_synthetic_validation
        self.epsilon = 1.0e-6
        self.enable_gradient_conflict_profiling = enable_gradient_conflict_profiling
        self.gradient_profiling_scope = gradient_profiling_scope
        self.gradient_focus_layer_name = gradient_focus_layer_name
        self.gradient_log_every_n_steps = gradient_log_every_n_steps
        self.gradient_ema_alpha = gradient_ema_alpha
        self.gradient_sma_window = gradient_sma_window
        self.gradient_profile_include_bias = gradient_profile_include_bias
        self._gradient_profile_train_step_count = 0
        self._gradient_profile_ema_state: dict[str, float] = {}
        self._gradient_profile_sma_buffers: dict[str, deque[float]] = {}

        if encoder_family == "mlp":
            self.encoder = build_multilayer_perceptron(
                input_dim=input_dim,
                intermediate_dim=latent_dim,
                output_dim=latent_dim,
                num_linear_layers=mlp_num_linear_layers,
                dropout=dropout,
                apply_output_activation=True,
            )
        elif encoder_family == "cnn_simple":
            self.encoder = SimpleWindowCnnEncoder(
                input_dim=input_dim,
                output_dim=latent_dim,
                hidden_channels=cnn_hidden_channels,
                kernel_size=cnn_kernel_size,
                num_layers=cnn_num_layers,
                dropout=self.cnn_dropout,
            )
        else:
            raise ValueError(f"Unsupported encoder_family: {encoder_family}")
        self.decoder = build_multilayer_perceptron(
            input_dim=latent_dim,
            intermediate_dim=latent_dim,
            output_dim=input_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
        self.classification_head = build_multilayer_perceptron(
            input_dim=window_size * latent_dim,
            intermediate_dim=classifier_dim,
            output_dim=num_classes,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
        # The older alias is preserved for compatibility with binary-era call
        # sites, but the active contract is generic class balancing.
        if balance_classes_within_batch is not None:
            effective_balance_classes_within_batch = bool(balance_classes_within_batch)
        else:
            effective_balance_classes_within_batch = bool(
                balance_binary_classes_within_batch or train_balance_classes
            )
        self.train_balance_classes = effective_balance_classes_within_batch
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_visibility_boost=anomaly_visibility_boost,
            anomaly_families=anomaly_families,
            balance_binary_classes_within_batch=(
                effective_balance_classes_within_batch
            ),
            deterministic_seed=synthetic_train_seed,
            classification_label_mode="redlamp_multiclass",
        )
        self.synthetic_validation_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_visibility_boost=anomaly_visibility_boost,
            anomaly_families=anomaly_families,
            balance_binary_classes_within_batch=(
                effective_balance_classes_within_batch
            ),
            deterministic_seed=synthetic_validation_seed,
            classification_label_mode="redlamp_multiclass",
        )
        self._encoder_profiled_parameters = self._get_encoder_profiled_parameters()

    def prepare_synthetic_training_epoch(self) -> None:
        self.synthetic_anomaly_injector.reset_rng()

    def prepare_synthetic_validation_epoch(self) -> None:
        self.synthetic_validation_injector.reset_rng()

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

    def _prepare_batch(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        if (
            "classification_labels" in batch
            and "synthetic_anomaly_mask" in batch
            and "augmentation_metadata" in batch
        ):
            return self._clone_batch(batch)
        if stage_name == "train" and self.use_synthetic_augmentation:
            return self.synthetic_anomaly_injector.augment_batch(batch)
        if stage_name == "val_synth" and self.use_synthetic_validation:
            return self.synthetic_validation_injector.augment_batch(batch)

        prepared_batch = self._clone_batch(batch)
        batch_size, window_size, _ = prepared_batch["x"].shape
        prepared_batch["classification_labels"] = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["classification_class_names"] = REDLAMP_MULTICLASS_CLASS_NAMES
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
        return prepared_batch

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch)
        x_tensor = batch["x"]
        _batch_size, window_size, input_dim = x_tensor.shape
        if window_size != self.window_size or input_dim != self.input_dim:
            raise ValueError(
                f"batch['x'] must have shape [B, {self.window_size}, {self.input_dim}]"
            )

        hidden = self.encoder(x_tensor)
        if hidden.shape[1] != self.window_size:
            raise ValueError(
                f"encoder must preserve window_size={self.window_size}, "
                f"but received hidden.shape[1]={hidden.shape[1]}"
            )
        flattened_classification_hidden = hidden.reshape(
            hidden.shape[0],
            self.window_size * self.latent_dim,
        )
        recon = self.decoder(hidden)
        logits = self.classification_head(flattened_classification_hidden)
        class_probabilities = torch.softmax(logits, dim=-1)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)

        outputs = {
            "hidden": hidden,
            "pooled": flattened_classification_hidden,
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {
                "class_probabilities": class_probabilities,
                "classification_class_names": REDLAMP_MULTICLASS_CLASS_NAMES,
            },
        }
        validate_model_outputs(outputs)
        return outputs

    def _get_encoder_profiled_parameters(self) -> OrderedDict[str, nn.Parameter]:
        profiled_parameters: OrderedDict[str, nn.Parameter] = OrderedDict()
        encoder_layers = (
            self.encoder.network if hasattr(self.encoder, "network") else self.encoder
        )
        affine_layer_indices = [
            layer_index
            for layer_index, layer_module in enumerate(encoder_layers)
            if isinstance(layer_module, (nn.Linear, nn.Conv1d))
        ]
        for layer_index in affine_layer_indices:
            layer_module = encoder_layers[layer_index]
            layer_type = "linear" if isinstance(layer_module, nn.Linear) else "conv"
            weight_key = f"encoder.{layer_type}{layer_index}.weight"
            profiled_parameters[weight_key] = layer_module.weight
            if self.gradient_profile_include_bias and layer_module.bias is not None:
                bias_key = f"encoder.{layer_type}{layer_index}.bias"
                profiled_parameters[bias_key] = layer_module.bias
        if len(profiled_parameters) == 0:
            raise ValueError("No encoder parameters available for gradient profiling")
        return profiled_parameters

    def _flatten_tensor_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return redlamp_helpers._flatten_tensor_for_metrics(self, tensor)

    def _compute_cosine_similarity(
        self,
        gradient_ce: torch.Tensor,
        gradient_mse: torch.Tensor,
    ) -> float:
        return redlamp_helpers._compute_cosine_similarity(
            self,
            gradient_ce,
            gradient_mse,
        )

    def _compute_preservation_ratio(
        self,
        gradient_ce: torch.Tensor,
        gradient_mse: torch.Tensor,
        gradient_total: torch.Tensor,
    ) -> float:
        return redlamp_helpers._compute_preservation_ratio(
            self,
            gradient_ce,
            gradient_mse,
            gradient_total,
        )

    def _extract_layerwise_gradients(
        self,
        loss: torch.Tensor,
        encoder_parameters: list[nn.Parameter],
    ) -> list[torch.Tensor]:
        return redlamp_helpers._extract_layerwise_gradients(
            self,
            loss,
            encoder_parameters,
        )

    def _update_ema(self, metric_key: str, metric_value: float) -> float:
        return redlamp_helpers._update_ema(self, metric_key, metric_value)

    def _update_sma(self, metric_key: str, metric_value: float) -> float:
        return redlamp_helpers._update_sma(self, metric_key, metric_value)

    def _resolve_focus_layer_parameter_name(self) -> str:
        return redlamp_helpers._resolve_focus_layer_parameter_name(self)

    def _build_gradient_conflict_log_dict(
        self,
        raw_metrics: dict[str, float],
    ) -> dict[str, float]:
        return redlamp_helpers._build_gradient_conflict_log_dict(self, raw_metrics)

    def _build_refurbished_classification_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        return redlamp_helpers._build_refurbished_classification_targets(
            self,
            classification_labels,
            target_dtype,
        )

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return redlamp_helpers._compute_classification_loss(self, outputs, batch)

    def _profile_encoder_gradient_conflict(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
    ) -> dict[str, float]:
        return redlamp_helpers._profile_encoder_gradient_conflict(
            self,
            reconstruction_loss,
            classification_loss,
        )

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage_name: str,
        classification_weight: float | None = None,
        include_classification_metrics: bool = True,
    ) -> dict[str, Any]:
        return redlamp_helpers._shared_step(
            self,
            batch,
            stage_name,
            classification_weight=classification_weight,
            include_classification_metrics=include_classification_metrics,
        )

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch, "train", include_classification_metrics=True)

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch,
            "val",
            classification_weight=0.0,
            include_classification_metrics=False,
        )

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch,
            "val_synth",
            include_classification_metrics=True,
        )

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch,
            "test",
            classification_weight=0.0,
            include_classification_metrics=False,
        )

from __future__ import annotations

"""Self-contained RedLamp-inspired MLP baseline.

The baseline keeps the repository batch and output contracts while using a
timestep encoder for a controlled comparison against the thesis model. It
remains an MLP autoencoder and multi-class synthetic anomaly classifier without
prototype memory, fusion gates, or online adaptation state.
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
from src.models.thesis_multitask import build_multilayer_perceptron


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


class RedLampMLPBaseline(BaseModel):
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
        lambda_cls: float = 0.1,
        use_label_refurbishment: bool = True,
        refurbishment_alpha: float = 0.1,
        refurbishment_beta: float = 0.01,
        anomaly_probability: float = 0.5,
        min_segment_fraction: float = 0.1,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
        anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
        use_synthetic_augmentation: bool = True,
        use_synthetic_validation: bool = True,
        synthetic_validation_seed: int = 7,
        classification_label_mode: str = "redlamp_multiclass",
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
                "RedLampMLPBaseline supports classification_label_mode='redlamp_multiclass'"
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
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_families=anomaly_families,
            balance_binary_classes_within_batch=balance_binary_classes_within_batch,
            classification_label_mode="redlamp_multiclass",
        )
        self.synthetic_validation_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_families=anomaly_families,
            balance_binary_classes_within_batch=balance_binary_classes_within_batch,
            deterministic_seed=synthetic_validation_seed,
            classification_label_mode="redlamp_multiclass",
        )
        self._encoder_profiled_parameters = self._get_encoder_profiled_parameters()

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

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
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

    def _get_encoder_profiled_parameters(self) -> OrderedDict[str, nn.Parameter]:
        profiled_parameters: OrderedDict[str, nn.Parameter] = OrderedDict()
        encoder_layers = (
            self.encoder.network
            if hasattr(self.encoder, "network")
            else self.encoder
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
        return tensor.reshape(-1)

    def _compute_cosine_similarity(
        self,
        gradient_ce: torch.Tensor,
        gradient_mse: torch.Tensor,
    ) -> float:
        gradient_ce_flattened = self._flatten_tensor_for_metrics(gradient_ce)
        gradient_mse_flattened = self._flatten_tensor_for_metrics(gradient_mse)
        dot_product = torch.dot(gradient_ce_flattened, gradient_mse_flattened)
        norm_ce = torch.linalg.vector_norm(gradient_ce_flattened)
        norm_mse = torch.linalg.vector_norm(gradient_mse_flattened)
        cosine_similarity = dot_product / ((norm_ce * norm_mse).clamp_min(1.0e-12))
        return float(cosine_similarity.detach().cpu())

    def _compute_preservation_ratio(
        self,
        gradient_ce: torch.Tensor,
        gradient_mse: torch.Tensor,
        gradient_total: torch.Tensor,
    ) -> float:
        norm_ce = torch.linalg.vector_norm(
            self._flatten_tensor_for_metrics(gradient_ce)
        )
        norm_mse = torch.linalg.vector_norm(
            self._flatten_tensor_for_metrics(gradient_mse)
        )
        norm_total = torch.linalg.vector_norm(
            self._flatten_tensor_for_metrics(gradient_total)
        )
        preservation_ratio = norm_total / (norm_ce + norm_mse).clamp_min(1.0e-12)
        return float(preservation_ratio.detach().cpu())

    def _extract_layerwise_gradients(
        self,
        loss: torch.Tensor,
        encoder_parameters: list[nn.Parameter],
    ) -> list[torch.Tensor]:
        gradients = torch.autograd.grad(
            loss,
            encoder_parameters,
            retain_graph=True,
            allow_unused=False,
        )
        return [gradient.detach().clone() for gradient in gradients]

    def _update_ema(self, metric_key: str, metric_value: float) -> float:
        previous_ema = self._gradient_profile_ema_state.get(metric_key)
        if previous_ema is None:
            updated_ema = metric_value
        else:
            updated_ema = (
                self.gradient_ema_alpha * metric_value
                + (1.0 - self.gradient_ema_alpha) * previous_ema
            )
        self._gradient_profile_ema_state[metric_key] = updated_ema
        return updated_ema

    def _update_sma(self, metric_key: str, metric_value: float) -> float:
        if metric_key not in self._gradient_profile_sma_buffers:
            self._gradient_profile_sma_buffers[metric_key] = deque(
                maxlen=self.gradient_sma_window
            )
        self._gradient_profile_sma_buffers[metric_key].append(metric_value)
        sma_value = sum(self._gradient_profile_sma_buffers[metric_key]) / len(
            self._gradient_profile_sma_buffers[metric_key]
        )
        return float(sma_value)

    def _profile_encoder_gradient_conflict(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
    ) -> dict[str, float]:
        encoder_parameter_items = list(self._encoder_profiled_parameters.items())
        encoder_parameter_names = [
            parameter_name for parameter_name, _ in encoder_parameter_items
        ]
        encoder_parameters = [
            parameter_tensor for _, parameter_tensor in encoder_parameter_items
        ]
        weighted_classification_loss = self.lambda_cls * classification_loss
        weighted_reconstruction_loss = reconstruction_loss
        gradients_ce = self._extract_layerwise_gradients(
            weighted_classification_loss,
            encoder_parameters,
        )
        gradients_mse = self._extract_layerwise_gradients(
            weighted_reconstruction_loss,
            encoder_parameters,
        )

        raw_metrics: dict[str, float] = {}
        for layer_name, gradient_ce, gradient_mse in zip(
            encoder_parameter_names, gradients_ce, gradients_mse
        ):
            gradient_total = gradient_ce + gradient_mse
            norm_ce = float(
                torch.linalg.vector_norm(self._flatten_tensor_for_metrics(gradient_ce))
                .detach()
                .cpu()
            )
            norm_mse = float(
                torch.linalg.vector_norm(self._flatten_tensor_for_metrics(gradient_mse))
                .detach()
                .cpu()
            )
            norm_total = float(
                torch.linalg.vector_norm(
                    self._flatten_tensor_for_metrics(gradient_total)
                )
                .detach()
                .cpu()
            )
            cosine_similarity = self._compute_cosine_similarity(
                gradient_ce, gradient_mse
            )
            preservation_ratio = self._compute_preservation_ratio(
                gradient_ce=gradient_ce,
                gradient_mse=gradient_mse,
                gradient_total=gradient_total,
            )
            metric_prefix = f"train_gradconf_raw/{layer_name}"
            raw_metrics[f"{metric_prefix}/cosine_sim"] = cosine_similarity
            raw_metrics[f"{metric_prefix}/r_ratio"] = preservation_ratio
            raw_metrics[f"{metric_prefix}/norm_ce"] = norm_ce
            raw_metrics[f"{metric_prefix}/norm_mse"] = norm_mse
            raw_metrics[f"{metric_prefix}/norm_total"] = norm_total
            if layer_name == self._resolve_focus_layer_parameter_name():
                focus_prefix = "train_gradconf_raw/focus"
                raw_metrics[f"{focus_prefix}/cosine_sim"] = cosine_similarity
                raw_metrics[f"{focus_prefix}/r_ratio"] = preservation_ratio
                raw_metrics[f"{focus_prefix}/norm_ce"] = norm_ce
                raw_metrics[f"{focus_prefix}/norm_mse"] = norm_mse
                raw_metrics[f"{focus_prefix}/norm_total"] = norm_total
        return self._build_gradient_conflict_log_dict(raw_metrics)

    def _resolve_focus_layer_parameter_name(self) -> str:
        if self.gradient_focus_layer_name in {
            "encoder_last_linear",
            "encoder_last_affine",
        }:
            parameter_names = list(self._encoder_profiled_parameters.keys())
            return parameter_names[-1]
        raise ValueError(
            f"Unsupported gradient_focus_layer_name: {self.gradient_focus_layer_name}"
        )

    def _build_gradient_conflict_log_dict(
        self,
        raw_metrics: dict[str, float],
    ) -> dict[str, float]:
        gradient_conflict_logs: dict[str, float] = {}
        for raw_key, raw_value in raw_metrics.items():
            gradient_conflict_logs[raw_key] = raw_value
            metric_suffix = raw_key.split("train_gradconf_raw/", 1)[1]
            ema_key = f"train_gradconf_ema/{metric_suffix}"
            sma_key = f"train_gradconf_sma/{metric_suffix}"
            gradient_conflict_logs[ema_key] = self._update_ema(raw_key, raw_value)
            gradient_conflict_logs[sma_key] = self._update_sma(raw_key, raw_value)
        return gradient_conflict_logs

    def _shared_step(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        prepared_batch = self._prepare_batch(batch, stage_name)
        outputs = self.forward(prepared_batch)
        reconstruction_loss = F.mse_loss(outputs["recon"], prepared_batch["x"])
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        total_loss = reconstruction_loss + self.lambda_cls * classification_loss
        predicted_labels = torch.argmax(outputs["logits"], dim=-1)
        classification_accuracy = (
            (predicted_labels == prepared_batch["classification_labels"]).float().mean()
        )
        log = {
            f"{stage_name}_loss": float(total_loss.detach().cpu()),
            f"{stage_name}_reconstruction_loss": float(
                reconstruction_loss.detach().cpu()
            ),
            f"{stage_name}_classification_loss": float(
                classification_loss.detach().cpu()
            ),
            f"{stage_name}_classification_accuracy": float(
                classification_accuracy.detach().cpu()
            ),
        }
        if stage_name == "train":
            self._gradient_profile_train_step_count += 1
            should_log_gradient_conflict = (
                self.enable_gradient_conflict_profiling
                and self._gradient_profile_train_step_count
                % self.gradient_log_every_n_steps
                == 0
            )
            if should_log_gradient_conflict:
                gradient_conflict_logs = self._profile_encoder_gradient_conflict(
                    reconstruction_loss=reconstruction_loss,
                    classification_loss=classification_loss,
                )
                log.update(gradient_conflict_logs)
        return {
            "loss": total_loss,
            "log": log,
            "outputs": outputs,
            "loss_terms": {
                "total_loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "classification_loss": classification_loss,
            },
            "batch": prepared_batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch, "val")

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch, "val_synth")

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch, "test")

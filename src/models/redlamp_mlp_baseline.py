from __future__ import annotations

"""Self-contained RedLamp-inspired MLP baseline.

The baseline keeps the repository batch and output contracts while using a
timestep encoder for a controlled comparison against the thesis model. It
remains an MLP autoencoder and multi-class synthetic anomaly classifier without
prototype memory, fusion gates, or online adaptation state.
"""

from typing import Any

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


class RedLampMLPBaseline(BaseModel):
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        latent_dim: int = 128,
        mlp_num_linear_layers: int = 3,
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
        if mlp_num_linear_layers < 2:
            raise ValueError("mlp_num_linear_layers must be at least 2")
        if classification_label_mode != "redlamp_multiclass":
            raise ValueError(
                "RedLampMLPBaseline supports classification_label_mode='redlamp_multiclass'"
            )

        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.mlp_num_linear_layers = mlp_num_linear_layers
        self.classifier_dim = classifier_dim
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.use_label_refurbishment = use_label_refurbishment
        self.refurbishment_alpha = refurbishment_alpha
        self.refurbishment_beta = refurbishment_beta
        self.use_synthetic_augmentation = use_synthetic_augmentation
        self.use_synthetic_validation = use_synthetic_validation
        self.epsilon = 1.0e-6

        self.encoder = build_multilayer_perceptron(
            input_dim=input_dim,
            intermediate_dim=latent_dim,
            output_dim=latent_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=True,
        )
        self.decoder = build_multilayer_perceptron(
            input_dim=latent_dim,
            intermediate_dim=latent_dim,
            output_dim=input_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
        self.classification_head = build_multilayer_perceptron(
            input_dim=latent_dim,
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
                "batch['x'] must have shape [B, "
                f"{self.window_size}, {self.input_dim}]"
            )

        hidden = self.encoder(x_tensor)
        pooled_hidden = hidden.mean(dim=1)
        recon = self.decoder(hidden)
        logits = self.classification_head(pooled_hidden)
        class_probabilities = torch.softmax(logits, dim=-1)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)

        outputs = {
            "hidden": hidden,
            "pooled": pooled_hidden,
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

    def _shared_step(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        prepared_batch = self._prepare_batch(batch, stage_name)
        outputs = self.forward(prepared_batch)
        reconstruction_loss = F.mse_loss(outputs["recon"], prepared_batch["x"])
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        total_loss = reconstruction_loss + self.lambda_cls * classification_loss
        predicted_labels = torch.argmax(outputs["logits"], dim=-1)
        classification_accuracy = (
            (predicted_labels == prepared_batch["classification_labels"])
            .float()
            .mean()
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

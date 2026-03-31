from __future__ import annotations

from typing import Any

import torch

from src.data.augment import SyntheticAnomalyInjector
from src.losses.classification import compute_classification_loss
from src.losses.prototype import compute_prototype_regularization
from src.models.base_model import BaseModel
from src.tasks.base_task import BaseTask


class MultitaskTSADTask(BaseTask):
    def __init__(
        self,
        reconstruction_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        prototype_loss_weight: float = 0.0,
        use_synthetic_augmentation: bool = True,
        anomaly_probability: float = 0.5,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
    ) -> None:
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.prototype_loss_weight = prototype_loss_weight
        self.use_synthetic_augmentation = use_synthetic_augmentation
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
        )

    def _prepare_batch(self, batch: dict[str, Any], stage: str) -> dict[str, Any]:
        if "classification_labels" in batch:
            return batch

        if stage == "train" and self.use_synthetic_augmentation:
            return self.synthetic_anomaly_injector.augment_batch(batch)

        prepared_batch = dict(batch)
        prepared_batch["classification_labels"] = torch.zeros(
            batch["x"].shape[0],
            dtype=torch.long,
            device=batch["x"].device,
        )
        prepared_batch["augmentation_metadata"] = [
            {"is_synthetic_anomaly": False, "anomaly_type": "clean"} for _ in range(batch["x"].shape[0])
        ]
        return prepared_batch

    def _shared_step(self, model: BaseModel, batch: dict[str, Any], stage: str) -> dict[str, Any]:
        prepared_batch = self._prepare_batch(batch, stage)
        outputs = model(prepared_batch)

        reconstruction_loss = torch.mean((outputs["recon"] - prepared_batch["x"]) ** 2)
        classification_loss = compute_classification_loss(
            outputs["logits"],
            prepared_batch["classification_labels"],
        )
        prototype_loss = compute_prototype_regularization(outputs["aux"])
        total_loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.classification_loss_weight * classification_loss
            + self.prototype_loss_weight * prototype_loss.to(reconstruction_loss.device)
        )

        predicted_labels = torch.argmax(outputs["logits"], dim=-1)
        classification_accuracy = (
            (predicted_labels == prepared_batch["classification_labels"]).float().mean()
        )

        return {
            "loss": total_loss,
            "metrics": {
                f"{stage}_loss": float(total_loss.detach().cpu()),
                f"{stage}_reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                f"{stage}_classification_loss": float(classification_loss.detach().cpu()),
                f"{stage}_prototype_loss": float(prototype_loss.detach().cpu()),
                f"{stage}_classification_accuracy": float(classification_accuracy.detach().cpu()),
            },
            "loss_terms": {
                "reconstruction_loss": reconstruction_loss,
                "classification_loss": classification_loss,
                "prototype_loss": prototype_loss,
            },
            "outputs": outputs,
            "batch": prepared_batch,
        }

    def training_step(self, model: BaseModel, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(model=model, batch=batch, stage="train")

    def validation_step(self, model: BaseModel, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(model=model, batch=batch, stage="val")

    def test_step(self, model: BaseModel, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(model=model, batch=batch, stage="test")

from __future__ import annotations

from typing import Any

import torch

from src.models.base_model import BaseModel
from src.tasks.base_task import BaseTask


class ReconstructionTask(BaseTask):
    def _shared_step(self, model: BaseModel, batch: dict[str, Any], stage: str) -> dict[str, Any]:
        outputs = model(batch)
        reconstruction_loss = torch.mean((outputs["recon"] - batch["x"]) ** 2)
        return {
            "loss": reconstruction_loss,
            "metrics": {
                f"{stage}_loss": float(reconstruction_loss.detach().cpu()),
                f"{stage}_mean_point_score": float(outputs["point_scores"].mean().detach().cpu()),
            },
            "outputs": outputs,
            "batch": batch,
        }

    def training_step(self, model: BaseModel, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(model=model, batch=batch, stage="train")

    def validation_step(self, model: BaseModel, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(model=model, batch=batch, stage="val")

    def test_step(self, model: BaseModel, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(model=model, batch=batch, stage="test")


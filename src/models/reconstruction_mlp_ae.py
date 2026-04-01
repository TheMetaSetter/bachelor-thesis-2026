from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.contracts import validate_batch, validate_model_outputs
from src.models.base_model import BaseModel


class ReconstructionMLPAutoencoder(BaseModel):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        loss_name: str = "mse",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, input_dim),
        )
        if loss_name != "mse":
            raise ValueError("ReconstructionMLPAutoencoder currently supports only loss_name='mse'")
        self.loss_name = loss_name

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch)
        x_tensor = batch["x"]
        batch_size, window_size, num_channels = x_tensor.shape
        flattened_x = x_tensor.reshape(batch_size * window_size, num_channels)

        hidden_flat = self.encoder(flattened_x)
        recon_flat = self.decoder(hidden_flat)

        hidden = hidden_flat.reshape(batch_size, window_size, self.hidden_dim)
        recon = recon_flat.reshape(batch_size, window_size, num_channels)
        reconstruction_error = torch.mean((recon - x_tensor) ** 2, dim=-1)

        outputs = {
            "hidden": hidden,
            "pooled": hidden.mean(dim=1),
            "recon": recon,
            "logits": None,
            "point_scores": reconstruction_error,
            "window_scores": reconstruction_error.mean(dim=1),
            "aux": {},
        }
        validate_model_outputs(outputs)
        return outputs

    def _compute_reconstruction_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """MSE loss"""
        return torch.mean((outputs["recon"] - batch["x"]) ** 2)

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        loss: torch.Tensor,
    ) -> dict[str, float]:
        return {
            f"{stage_name}_loss": float(loss.detach().cpu()),
            f"{stage_name}_mean_point_score": float(outputs["point_scores"].mean().detach().cpu()),
        }

    def _shared_step(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        outputs = self.forward(batch)
        loss = self._compute_reconstruction_loss(outputs, batch)
        return {
            "loss": loss,
            "log": self._build_stage_log(stage_name, outputs, loss),
            "outputs": outputs,
            "loss_terms": {"reconstruction_loss": loss},
            "batch": batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="train")

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="val")

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="test")

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

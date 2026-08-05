from __future__ import annotations

"""Small convolutional window autoencoder for online deep baselines."""

import torch
import torch.nn as nn

from src.models.neural_blocks import SimpleWindowCnnEncoder


class SimpleWindowCnnAutoencoder(nn.Module):
    """Reconstruct windows with the shared simple 1D-CNN encoder."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SimpleWindowCnnEncoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = nn.Conv1d(latent_dim, input_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent.transpose(1, 2)).transpose(1, 2)
        return reconstruction, latent

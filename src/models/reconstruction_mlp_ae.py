from __future__ import annotations
"""Self-contained reconstruction baseline for the thesis codebase.

A new reader should start here if they want the smallest full model in the
repository. The same file owns architecture, forward logic, loss computation,
and stage methods so the baseline can be read top-to-bottom without jumping to
task-specific helper files.
"""

from typing import Any

import torch
import torch.nn as nn

from src.core.console import console_print, print_parameter_summary, summarize_batch, summarize_tensor
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

        # Encoder and decoder stay deliberately plain here because this file is
        # the minimal vertical slice that all later models are compared against.
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
        print_parameter_summary(
            "MODEL",
            "ReconstructionMLPAutoencoder",
            self,
            {
                "encoder": self.encoder,
                "decoder": self.decoder,
            },
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            loss_name=loss_name,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        # The baseline flattens the window temporarily so the MLP can work on
        # per-timestep vectors, then restores the `[B, L, *]` thesis contract.
        validate_batch(batch)
        console_print("MODEL", "Baseline forward input batch", **summarize_batch(batch))
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
        console_print(
            "MODEL",
            "Baseline forward outputs",
            hidden=summarize_tensor(outputs["hidden"]),
            pooled=summarize_tensor(outputs["pooled"]),
            recon=summarize_tensor(outputs["recon"]),
            point_scores=summarize_tensor(outputs["point_scores"]),
            window_scores=summarize_tensor(outputs["window_scores"]),
        )
        return outputs

    def _compute_reconstruction_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        # The baseline objective stays intentionally narrow: reconstruction only.
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
        # Training, validation, and test all share the same mechanics here so
        # the only changing part is the stage prefix used in the logs.
        outputs = self.forward(batch)
        loss = self._compute_reconstruction_loss(outputs, batch)
        console_print(
            stage_name.upper(),
            "Baseline stage step completed",
            batch_size=batch["x"].shape[0],
            loss=float(loss.detach().cpu()),
            mean_point_score=float(outputs["point_scores"].mean().detach().cpu()),
        )
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

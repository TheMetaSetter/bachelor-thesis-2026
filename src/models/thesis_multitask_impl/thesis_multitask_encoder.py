"""Encoder primitive used by the thesis multitask model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.nn as nn

from src.models.neural_blocks import SimpleWindowCnnEncoder, build_multilayer_perceptron

if TYPE_CHECKING:
    from src.models.thesis_multitask_impl.thesis_multitask_components import MultitaskArchitectureConfig


class MultitaskWindowEncoder(nn.Module):
    """Build an MLP/CNN encoder and preserve the ``hidden`` output contract."""

    def __init__(self, architecture: "MultitaskArchitectureConfig") -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder_family = architecture.encoder_family
        if architecture.encoder_family == "mlp":
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
        """Encode ``batch['x']`` and return sequence and pooled representations."""
        hidden = self.network(batch["x"])
        if hidden.shape[1] != self.architecture.window_size:
            raise ValueError(
                "encoder must preserve window_size="
                f"{self.architecture.window_size}, but received "
                f"hidden.shape[1]={hidden.shape[1]}"
            )
        return {
            "hidden": hidden,
            "pooled": hidden.mean(dim=1),
            "aux": {"encoder_name": "multitask_window_encoder"},
        }

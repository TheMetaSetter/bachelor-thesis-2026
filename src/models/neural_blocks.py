"""Small, reusable neural-network building blocks.

These blocks own no experiment phase, checkpoint state, or model lifecycle.
Both offline models use them so their public model files stay independent.
"""

import torch
import torch.nn as nn


def build_multilayer_perceptron(
    *,
    input_dim: int,
    intermediate_dim: int,
    output_dim: int,
    num_linear_layers: int,
    dropout: float,
    apply_output_activation: bool,
) -> nn.Sequential:
    """Build an initialized MLP with an explicit depth contract."""
    if num_linear_layers < 2:
        raise ValueError("num_linear_layers must be at least 2")
    layer_dims = [input_dim] + [intermediate_dim] * (num_linear_layers - 1)
    layer_dims.append(output_dim)
    layers = _build_mlp_layers(layer_dims, dropout, apply_output_activation)
    network = nn.Sequential(*layers)
    _initialize_mlp_linear_layers(network)
    return network


def _build_mlp_layers(
    layer_dims: list[int], dropout: float, apply_output_activation: bool
) -> list[nn.Module]:
    layers: list[nn.Module] = []
    last_index = len(layer_dims) - 2
    for index, (input_size, output_size) in enumerate(
        zip(layer_dims[:-1], layer_dims[1:])
    ):
        layers.append(nn.Linear(input_size, output_size))
        if index != last_index:
            layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        elif apply_output_activation:
            layers.append(nn.ReLU())
    return layers


def _initialize_mlp_linear_layers(network: nn.Sequential) -> None:
    """Use Kaiming for hidden ReLU layers and Xavier for output layers."""
    for index, layer in enumerate(network):
        if not isinstance(layer, nn.Linear):
            continue
        next_layer = network[index + 1] if index + 1 < len(network) else None
        if isinstance(next_layer, nn.ReLU):
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SimpleWindowCnnEncoder(nn.Module):
    """Preserve time length while encoding a ``[batch, time, feature]`` window."""

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
        _validate_cnn_dimensions(
            input_dim, output_dim, hidden_channels, kernel_size, num_layers
        )
        self.network = nn.Sequential(
            *_build_cnn_layers(
                input_dim, output_dim, hidden_channels, kernel_size, num_layers, dropout
            )
        )
        _initialize_conv_layers(self.network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, D]")
        return self.network(x.transpose(1, 2)).transpose(1, 2)


def _validate_cnn_dimensions(
    input_dim: int,
    output_dim: int,
    hidden_channels: int,
    kernel_size: int,
    num_layers: int,
) -> None:
    for name, value in {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_channels": hidden_channels,
        "kernel_size": kernel_size,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2")


def _build_cnn_layers(
    input_dim: int,
    output_dim: int,
    hidden_channels: int,
    kernel_size: int,
    num_layers: int,
    dropout: float,
) -> list[nn.Module]:
    dimensions = [input_dim] + [hidden_channels] * (num_layers - 1) + [output_dim]
    padding_left = (kernel_size - 1) // 2
    padding_right = kernel_size - 1 - padding_left
    layers: list[nn.Module] = []
    for index, (input_size, output_size) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        layers.extend(
            (
                nn.ConstantPad1d((padding_left, padding_right), 0.0),
                nn.Conv1d(input_size, output_size, kernel_size),
                nn.ReLU(),
            )
        )
        if index != num_layers - 1:
            layers.append(nn.Dropout(dropout))
    return layers


def _initialize_conv_layers(network: nn.Sequential) -> None:
    for layer in network:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

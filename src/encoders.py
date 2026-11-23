"""Create the common time-preserving window encoder."""

import torch
from torch import nn


class _WindowEncoder(nn.Module):
    def __init__(self, network: nn.Module, window_size: int, input_features: int):
        super().__init__()
        self.network = network
        self.shape = (window_size, input_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != self.shape:
            raise ValueError(
                f"encoder expects shape [batch, {self.shape[0]}, {self.shape[1]}]"
            )
        return self.network(x.transpose(1, 2)).transpose(1, 2)


def create_encoder(
    settings: dict, *, input_features: int, window_size: int
) -> nn.Module:
    """Create an independent encoder with one feature vector per time point."""
    hidden, output = settings["hidden_channels"], settings["output_dim"]
    kernel, depth, dropout = (
        settings["kernel_size"],
        settings["num_layers"],
        settings["dropout"],
    )
    if any(
        type(count) is not int or count < 1
        for count in (hidden, output, kernel, input_features, window_size)
    ):
        raise ValueError("encoder dimensions must be positive integers")
    if type(depth) is not int or depth < 2 or not 0 <= dropout < 1:
        raise ValueError("encoder needs at least two layers and dropout in [0, 1)")
    dimensions = [input_features, *([hidden] * (depth - 1)), output]
    left = (kernel - 1) // 2
    padding = (left, kernel - 1 - left)
    layers = []
    convolutions = []
    for index, (in_channels, out_channels) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        convolution = nn.Conv1d(in_channels, out_channels, kernel)
        layers.extend((nn.ConstantPad1d(padding, 0.0), convolution, nn.ReLU()))
        convolutions.append(convolution)
        if index < depth - 1:
            layers.append(nn.Dropout(dropout))
    for convolution in convolutions:
        nn.init.kaiming_uniform_(convolution.weight, nonlinearity="relu")
        nn.init.zeros_(convolution.bias)
    return _WindowEncoder(nn.Sequential(*layers), window_size, input_features)


class NearIdentityMLPProjector(nn.Module):
    # The projector is residual and near-identity on purpose. That keeps the
    # first online step close to the calibrated latent space.
    def __init__(
        self,
        hidden_dim: int,
        projector_hidden_dim: int,
        dropout: float = 0.0,
        init_alpha: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, projector_hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(projector_hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        nn.init.kaiming_uniform_(self.fc1.weight, a=5**0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # At initialization the residual branch is tiny, so the projector is
        # almost an identity map but still trainable.
        residual = self.fc1(hidden)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        return hidden + self.alpha * residual

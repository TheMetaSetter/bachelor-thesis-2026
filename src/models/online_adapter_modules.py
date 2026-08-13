from __future__ import annotations

"""Small method-owned modules shared by the online M2N2/CANDI adapters."""

import torch
from torch import nn


class Detrender(nn.Module):
    """Reference M2N2 mean-only exponential detrending state."""

    def __init__(self, *, num_features: int, gamma: float = 0.99999) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        self.num_features = int(num_features)
        self.gamma = float(gamma)
        self.mean = nn.Parameter(
            torch.zeros(1, 1, self.num_features), requires_grad=False
        )

    @torch.no_grad()
    def update_statistics(self, x: torch.Tensor) -> None:
        if x.ndim < 2 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"x must end with {self.num_features} features, got {tuple(x.shape)}"
            )
        reduce_dims = tuple(range(0, x.ndim - 1))
        batch_mean = torch.mean(x, dim=reduce_dims, keepdim=True).detach()
        self.mean.lerp_(batch_mean, 1.0 - self.gamma)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mean


class _FullAttention(nn.Module):
    """Full, non-causal self-attention used by reference SANA."""

    def __init__(self, *, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        _, _, _, head_dimension = queries.shape
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores = scores / (head_dimension**0.5)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attention, values)


class _AttentionLayer(nn.Module):
    """Projection wrapper matching the reference attention layer."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dimension = d_model // n_heads
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.inner_attention = _FullAttention(dropout=dropout)
        self.head_dimension = head_dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        queries = self.query_projection(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dimension
        )
        keys = self.key_projection(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dimension
        )
        values = self.value_projection(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dimension
        )
        attended = self.inner_attention(queries, keys, values)
        return self.out_projection(attended.reshape(batch_size, sequence_length, -1))


class _ReferenceEncoderLayer(nn.Module):
    """One attention plus convolutional feed-forward reference layer."""

    def __init__(
        self, *, d_model: int, d_ff: int, n_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention = _AttentionLayer(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        x = self.norm1(x + self.dropout(attended))
        feed_forward = self.conv1(x.transpose(1, 2))
        feed_forward = torch.nn.functional.gelu(feed_forward)
        feed_forward = self.dropout(self.conv2(feed_forward)).transpose(1, 2)
        return self.norm2(x + feed_forward)


class _ReferenceEncoder(nn.Module):
    """Stack of reference encoder layers with a final LayerNorm."""

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ReferenceEncoderLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _TemporalEmbedding(nn.Module):
    """Reference one-layer temporal convolution for one variable."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x).mean(dim=2)


class SANA(nn.Module):
    """CANDI's trainable residual adaptation module.

    ``TCN_iTrans`` follows the reference topology: one temporal embedding per
    variable, cross-variable transformer encoding and one projection per
    variable. ``Linear`` is retained as an explicit lightweight configuration
    option for controlled ablations.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        sana_type: str = "TCN_iTrans",
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.0,
        gating_init: float = 0.0,
    ) -> None:
        super().__init__()
        if sana_type not in {"TCN_iTrans", "Linear"}:
            raise ValueError("sana_type must be 'TCN_iTrans' or 'Linear'")
        if input_dim <= 0 or window_size <= 0:
            raise ValueError("input_dim and window_size must be positive")
        if d_model <= 0 or d_ff <= 0 or n_heads <= 0:
            raise ValueError("SANA dimensions must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.sana_type = sana_type
        self.gating = nn.Parameter(torch.full((self.input_dim,), float(gating_init)))
        if sana_type == "Linear":
            self.projection = nn.ModuleList(
                [
                    nn.Linear(self.window_size, self.window_size)
                    for _ in range(self.input_dim)
                ]
            )
            return

        self.temporal_embedding = nn.ModuleList(
            [_TemporalEmbedding(d_model=d_model) for _ in range(self.input_dim)]
        )
        self.encoder = _ReferenceEncoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            num_layers=1,
        )
        self.projection = nn.ModuleList(
            [nn.Linear(d_model, self.window_size) for _ in range(self.input_dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.window_size, self.input_dim):
            raise ValueError(
                "x must have shape "
                f"[B, {self.window_size}, {self.input_dim}], got {tuple(x.shape)}"
            )
        channels_first = x.permute(0, 2, 1)
        if self.sana_type == "Linear":
            decoded = torch.stack(
                [
                    projection(channels_first[:, index, :])
                    for index, projection in enumerate(self.projection)
                ],
                dim=1,
            )
        else:
            embedded = torch.stack(
                [
                    embedding(channels_first[:, index : index + 1, :])
                    for index, embedding in enumerate(self.temporal_embedding)
                ],
                dim=1,
            )
            encoded = self.encoder(embedded)
            decoded = torch.stack(
                [
                    projection(encoded[:, index, :])
                    for index, projection in enumerate(self.projection)
                ],
                dim=1,
            )
        return decoded.permute(0, 2, 1) * torch.tanh(self.gating)

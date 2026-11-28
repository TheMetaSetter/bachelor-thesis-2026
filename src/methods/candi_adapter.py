"""CANDI SANA residual adapter using grouped temporal convolution and cross-channel attention."""

import torch
from torch import nn


class _AttentionLayer(nn.Module):
    """Multi-head cross-variable self-attention."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        q = self.q_proj(x).view(b, c, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, c, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(b, c, self.n_heads, self.head_dim)
        scores = torch.einsum("blhe,bshe->bhls", q, k) / (self.head_dim**0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, v).reshape(b, c, -1)
        return self.out_proj(out)


class _ReferenceEncoderLayer(nn.Module):
    """Cross-variable attention followed by 1x1 convolutional feed-forward."""

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
        x = self.norm1(x + self.dropout(self.attention(x)))
        ff = torch.nn.functional.gelu(self.conv1(x.transpose(1, 2)))
        ff = self.dropout(self.conv2(ff)).transpose(1, 2)
        return self.norm2(x + ff)


class SANA(nn.Module):
    """CANDI's trainable per-channel residual adaptation module."""

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.0,
        gating_init: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            input_dim <= 0
            or window_size <= 0
            or d_model <= 0
            or d_ff <= 0
            or n_heads <= 0
        ):
            raise ValueError("SANA dimensions must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.gating = nn.Parameter(torch.full((self.input_dim,), float(gating_init)))
        self.temporal_conv = nn.Conv1d(
            self.input_dim,
            self.input_dim * self.d_model,
            kernel_size=3,
            padding=1,
            groups=self.input_dim,
        )
        self.encoder_layer = _ReferenceEncoderLayer(
            d_model=self.d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout
        )
        self.encoder_norm = nn.LayerNorm(self.d_model)
        self.proj_weight = nn.Parameter(
            torch.empty(self.input_dim, self.d_model, self.window_size)
        )
        self.proj_bias = nn.Parameter(torch.zeros(self.input_dim, self.window_size))
        nn.init.kaiming_uniform_(self.proj_weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.window_size, self.input_dim):
            raise ValueError(
                f"x must have shape [B, {self.window_size}, {self.input_dim}], got {tuple(x.shape)}"
            )
        b = x.shape[0]
        embedded = torch.relu(self.temporal_conv(x.transpose(1, 2))).mean(dim=-1)
        embedded = embedded.view(b, self.input_dim, self.d_model)
        encoded = self.encoder_norm(self.encoder_layer(embedded))
        decoded = (
            torch.einsum("bcd,cdl->bcl", encoded, self.proj_weight) + self.proj_bias
        )
        return decoded.transpose(1, 2) * torch.tanh(self.gating)

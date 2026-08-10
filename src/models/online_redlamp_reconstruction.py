from __future__ import annotations

"""RedLamp encoder model surface used by the online adapters.

The repository provides a trained simple 1D-CNN encoder checkpoint. The online
adapter owns the reconstruction head and updates it with the reference M2N2 or
CANDI objective. This is explicitly an encoder-checkpoint repository variant,
not a native MLP/TimesNet checkpoint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.core.artifact_integrity import sha256_file
from src.models.neural_blocks import (
    SimpleWindowCnnEncoder,
    build_multilayer_perceptron,
)


@dataclass(frozen=True)
class RedLampReconstructionCheckpoint:
    """Identity for a RedLamp encoder checkpoint used by an online adapter."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_role: str
    checkpoint_contract: str
    epoch: int | None


class RedLampReconstructionModel(nn.Module):
    """Reconstruction part of the trained RedLamp baseline."""

    def __init__(
        self,
        *,
        input_dim: int,
        window_size: int,
        latent_dim: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
        mlp_num_linear_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.latent_dim = int(latent_dim)
        self.encoder_family = "cnn_simple"
        self.encoder = SimpleWindowCnnEncoder(
            input_dim=self.input_dim,
            output_dim=self.latent_dim,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = build_multilayer_perceptron(
            input_dim=self.latent_dim,
            intermediate_dim=self.latent_dim,
            output_dim=self.input_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, L, D]")
        if x.shape[1:] != (self.window_size, self.input_dim):
            raise ValueError(
                "x must have shape "
                f"[B, {self.window_size}, {self.input_dim}], got {tuple(x.shape)}"
            )
        hidden = self.encoder(x)
        return self.decoder(hidden)

    @torch.no_grad()
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reference CANDI representation contract."""
        hidden = self.encoder(x)
        representation = hidden.mean(dim=1)
        return torch.nn.functional.normalize(representation, p=2, dim=1)


def _load_checkpoint_payload(checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "The installed PyTorch must support safe weights_only checkpoint loading"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"RedLamp reconstruction checkpoint payload must be a mapping: {checkpoint_path}"
        )
    return payload


def _model_state_dict(
    payload: Mapping[str, Any], checkpoint_path: Path
) -> Mapping[str, Any]:
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"RedLamp checkpoint has no model_state_dict mapping: {checkpoint_path}"
        )
    return state_dict


def _encoder_state_dict(
    state_dict: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    prefixed = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    if prefixed:
        return prefixed
    direct = {
        key: value
        for key, value in state_dict.items()
        if isinstance(key, str)
    }
    if direct:
        return direct
    raise ValueError(f"RedLamp checkpoint has no encoder tensors: {checkpoint_path}")


def _validate_state_dict(
    model: nn.Module,
    observed: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    expected = model.state_dict()
    expected_keys = set(expected)
    observed_keys = set(observed)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"RedLamp reconstruction keys do not match {checkpoint_path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, expected_tensor in expected.items():
        observed_tensor = observed[key]
        if not isinstance(observed_tensor, torch.Tensor):
            raise TypeError(f"RedLamp checkpoint value is not a tensor: {key}")
        if tuple(observed_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"RedLamp reconstruction shape mismatch for {key}: "
                f"expected={tuple(expected_tensor.shape)}, "
                f"observed={tuple(observed_tensor.shape)}"
            )


def load_redlamp_reconstruction_checkpoint(
    *,
    model: RedLampReconstructionModel,
    checkpoint_path: str | Path,
) -> RedLampReconstructionCheckpoint:
    """Load the trained encoder and initialize the adapter-owned head."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"pretrained RedLamp reconstruction checkpoint does not exist: {resolved_path}"
        )
    payload = _load_checkpoint_payload(resolved_path)
    observed = _encoder_state_dict(
        _model_state_dict(payload, resolved_path), resolved_path
    )
    _validate_state_dict(model.encoder, observed, resolved_path)
    model.encoder.load_state_dict(dict(observed), strict=True)
    epoch_value = payload.get("epoch")
    epoch = int(epoch_value) if isinstance(epoch_value, int) else None
    return RedLampReconstructionCheckpoint(
        checkpoint_path=str(resolved_path),
        checkpoint_sha256=sha256_file(resolved_path),
        checkpoint_role="pretrained_encoder",
        checkpoint_contract="reference_adapter_redlamp_encoder",
        epoch=epoch,
    )

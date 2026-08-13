from __future__ import annotations

"""Load only the encoder from a RedLamp checkpoint."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from src.core.artifact_integrity import sha256_file


@dataclass(frozen=True)
class RedLampEncoderCheckpoint:
    """Identity and tensors loaded from one RedLamp checkpoint."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_role: str
    epoch: int | None


def _load_checkpoint_payload(checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "The installed PyTorch must support safe weights_only checkpoint loading"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"RedLamp checkpoint payload must be a mapping: {checkpoint_path}"
        )
    return payload


def _encoder_state_dict(
    payload: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        model_state_dict = payload.get("state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise ValueError(
            f"RedLamp checkpoint has no model_state_dict mapping: {checkpoint_path}"
        )
    encoder_state_dict = {
        key.removeprefix("encoder."): value
        for key, value in model_state_dict.items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    if not encoder_state_dict:
        raise ValueError(
            f"RedLamp checkpoint has no encoder.* tensors: {checkpoint_path}"
        )
    return encoder_state_dict


def _validate_encoder_state_dict(
    encoder: nn.Module,
    encoder_state_dict: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    expected_state_dict = encoder.state_dict()
    expected_keys = set(expected_state_dict)
    observed_keys = set(encoder_state_dict)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"RedLamp encoder keys do not match {checkpoint_path}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, expected_tensor in expected_state_dict.items():
        observed_tensor = encoder_state_dict[key]
        if not isinstance(observed_tensor, torch.Tensor):
            raise TypeError(f"RedLamp encoder value is not a tensor: {key}")
        if tuple(observed_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"RedLamp encoder shape mismatch for {key}: "
                f"expected={tuple(expected_tensor.shape)}, "
                f"observed={tuple(observed_tensor.shape)}"
            )


def load_redlamp_encoder_checkpoint(
    *,
    encoder: nn.Module,
    checkpoint_path: str | Path,
) -> RedLampEncoderCheckpoint:
    """Load strict ``encoder.*`` tensors and ignore all RedLamp heads."""

    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"pretrained RedLamp encoder checkpoint does not exist: {resolved_path}"
        )
    payload = _load_checkpoint_payload(resolved_path)
    encoder_state_dict = _encoder_state_dict(payload, resolved_path)
    _validate_encoder_state_dict(encoder, encoder_state_dict, resolved_path)
    encoder.load_state_dict(dict(encoder_state_dict), strict=True)
    epoch_value = payload.get("epoch")
    epoch = int(epoch_value) if isinstance(epoch_value, int) else None
    return RedLampEncoderCheckpoint(
        checkpoint_path=str(resolved_path),
        checkpoint_sha256=sha256_file(resolved_path),
        checkpoint_role="pretrained_encoder",
        epoch=epoch,
    )

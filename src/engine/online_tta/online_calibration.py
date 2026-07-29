"""Pure stream-scoring helpers for THESIS online threshold calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.data.stream import OnlineWindowBatcher, SMDOnlineStream
from src.data.window import slice_sequence_into_windows
from src.protocols.point_scores import (
    ewma_scores,
    window_scores_to_causal_endpoint_scores,
)


def build_online_stream(
    *,
    sequences: list[dict[str, Any]],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
) -> OnlineWindowBatcher:
    stream = SMDOnlineStream(
        sequences=sequences,
        window_size=window_size,
        stride=1,
        clean_stream_only=True,
        stream_window_mode="sliding_stride_1",
    )
    return OnlineWindowBatcher(
        stream=stream,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
        include_legacy_views=False,
    )


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _collect_batch_scores(
    outputs: dict[str, Any], batch_on_device: dict[str, Any]
) -> tuple[list[float], list[float], list[float]]:
    endpoint_scores = outputs["point_scores"][:, -1].detach().cpu().tolist()
    input_scores = (
        ((outputs["recon"] - batch_on_device["x"]) ** 2)
        .mean(dim=(1, 2))
        .detach()
        .cpu()
        .tolist()
    )
    latent_scores = outputs["aux"].get("latent_window_score")
    if not isinstance(latent_scores, torch.Tensor):
        raise KeyError("online model must expose aux.latent_window_score")
    return (
        endpoint_scores,
        input_scores,
        latent_scores.reshape(-1).detach().cpu().tolist(),
    )


def run_stride1_sequence_scores(
    *,
    model: torch.nn.Module,
    sequence: dict[str, Any],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
) -> dict[str, list[float]]:
    batcher = build_online_stream(
        sequences=[sequence],
        window_size=window_size,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
    )
    endpoint_scores: list[float] = []
    input_window_scores: list[float] = []
    latent_window_scores: list[float] = []
    for batch in batcher:
        batch_on_device = move_batch_to_device(batch, device)
        model.eval()
        with torch.no_grad():
            outputs = model.forward(batch_on_device)
        endpoint, input_scores, latent_scores = _collect_batch_scores(
            outputs, batch_on_device
        )
        endpoint_scores.extend(endpoint)
        input_window_scores.extend(input_scores)
        latent_window_scores.extend(latent_scores)
    causal_scores = window_scores_to_causal_endpoint_scores(
        window_scores=endpoint_scores,
        sequence_length=int(sequence["x"].shape[0]),
        window_size=window_size,
    )
    return {
        "point": [float(value) for value in causal_scores if not np.isnan(value)],
        "input_window": [float(value) for value in input_window_scores],
        "latent_window": [float(value) for value in latent_window_scores],
    }


def _collect_offline_scores(
    model: torch.nn.Module, sequence: dict[str, Any], window_size: int, device: str
) -> list[float]:
    collected: list[float] = []
    windows = slice_sequence_into_windows(
        sequence, window_size=window_size, stride=window_size, tail_policy="end_align"
    )
    for window in windows:
        batch = {
            "x": window["x"].unsqueeze(0).to(device),
            "point_labels": None,
            "mask": None,
            "timestamps": None,
            "meta": [window["meta"]],
        }
        model.eval()
        with torch.no_grad():
            outputs = (
                model.forward_source(batch)
                if hasattr(model, "forward_source")
                else model.forward(batch)
            )
        collected.extend(outputs["point_scores"].reshape(-1).detach().cpu().tolist())
    return collected


def collect_nonoverlap_offline_scores(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    window_size: int,
    device: str,
) -> list[float]:
    """Collect offline calibration scores from non-overlapping validation windows."""
    collected: list[float] = []
    for sequence in clean_validation_sequences:
        collected.extend(_collect_offline_scores(model, sequence, window_size, device))
    return collected


def collect_stride1_online_scores(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
    current_weight: float,
    previous_weight: float,
) -> dict[str, list[float]]:
    """Collect stride-1 causal validation scores for online thresholding."""
    collected = {key: [] for key in ("point", "ewma", "input_window", "latent_window")}
    for sequence in clean_validation_sequences:
        scores = run_stride1_sequence_scores(
            model=model,
            sequence=sequence,
            window_size=window_size,
            batch_size=batch_size,
            view_noise_std=view_noise_std,
            view_dropout_probability=view_dropout_probability,
            device=device,
        )
        smoothed = ewma_scores(
            np.asarray(scores["point"], dtype=float),
            current_weight=current_weight,
            previous_weight=previous_weight,
        )
        collected["point"].extend(scores["point"])
        collected["ewma"].extend(
            float(score) for score in smoothed if not np.isnan(score)
        )
        collected["input_window"].extend(scores["input_window"])
        collected["latent_window"].extend(scores["latent_window"])
    return collected


def collect_clean_validation_scores(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    window_size: int,
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
    current_weight: float,
    previous_weight: float,
) -> dict[str, list[float]]:
    collected = {"offline_point": []}
    collected["offline_point"] = collect_nonoverlap_offline_scores(
        model=model,
        clean_validation_sequences=clean_validation_sequences,
        window_size=window_size,
        device=device,
    )
    stride1_scores = collect_stride1_online_scores(
        model=model,
        clean_validation_sequences=clean_validation_sequences,
        window_size=window_size,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
        device=device,
        current_weight=current_weight,
        previous_weight=previous_weight,
    )
    collected.update(stride1_scores)
    return collected

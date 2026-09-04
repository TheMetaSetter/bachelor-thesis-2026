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
from src.engine.online_tta.point_ewma import update_window_point_ewma
from src.protocols.reconstruction_scores import score_reconstruction


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


def _forward_calibration_window(
    model: torch.nn.Module, batch_on_device: dict[str, Any]
) -> dict[str, Any]:
    """Use the frozen source path when calibrating an A0 online model."""
    if getattr(model, "online_variant", None) == "A0" and hasattr(
        model, "forward_source"
    ):
        return model.forward_source(batch_on_device)
    return model.forward(batch_on_device)


def _collect_batch_scores(
    outputs: dict[str, Any],
    batch_on_device: dict[str, Any],
    scaler: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float], list[float], list[float]]:
    if scaler is None:
        point_scores = outputs["point_scores"].detach()
        normalized_point_scores = point_scores
        input_scores = (
            ((outputs["recon"] - batch_on_device["x"]) ** 2)
            .mean(dim=(1, 2))
            .detach()
            .cpu()
            .tolist()
        )
        normalized_input_scores = input_scores
    else:
        stochastic_query = outputs["aux"].get("stochastic_query") or {}
        reconstruction = stochastic_query.get("reconstruction_samples")
        if not isinstance(reconstruction, torch.Tensor):
            reconstruction = outputs["recon"]
        scores = score_reconstruction(batch_on_device["x"], reconstruction, scaler)
        point_scores = scores["raw_input_point_mse"].detach()
        normalized_point_scores = scores["normalized_input_point_mse"].detach()
        input_scores = scores["raw_input_window_mse"].detach().cpu().tolist()
        normalized_input_scores = (
            scores["normalized_input_window_mse"].detach().cpu().tolist()
        )
    latent_scores = outputs["aux"].get("latent_window_score")
    if not isinstance(latent_scores, torch.Tensor):
        geometry = outputs["aux"].get("deterministic_geometry")
        if isinstance(geometry, dict):
            latent_scores = geometry.get("latent_window_score")
    if not isinstance(latent_scores, torch.Tensor):
        latent_scores = outputs.get("window_scores")
    if not isinstance(latent_scores, torch.Tensor):
        raise KeyError("online model must expose aux.latent_window_score")
    return (
        point_scores,
        normalized_point_scores,
        input_scores,
        normalized_input_scores,
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
    current_weight: float,
    previous_weight: float,
    scaler: Any | None = None,
) -> dict[str, list[float]]:
    batcher = build_online_stream(
        sequences=[sequence],
        window_size=window_size,
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
    )
    active_ewma_point_scores: dict[int, float] = {}
    point_scores: list[float] = []
    normalized_point_score_values: list[float] = []
    input_window_scores: list[float] = []
    normalized_input_window_scores: list[float] = []
    latent_window_scores: list[float] = []
    for batch in batcher:
        batch_on_device = move_batch_to_device(batch, device)
        model.eval()
        with torch.no_grad():
            outputs = _forward_calibration_window(model, batch_on_device)
        (
            current_point_scores,
            normalized_point_scores,
            input_scores,
            normalized_input_scores,
            latent_scores,
        ) = _collect_batch_scores(
            outputs, batch_on_device, scaler=scaler
        )
        if current_point_scores.shape[0] != 1:
            raise ValueError("online threshold calibration requires batch_size=1")
        current_ewma_scores, active_ewma_point_scores = update_window_point_ewma(
            previous_scores=active_ewma_point_scores,
            absolute_indices=batch_on_device["absolute_indices"][0],
            window_point_scores=current_point_scores[0],
            current_weight=current_weight,
            previous_weight=previous_weight,
        )
        point_scores.extend(float(value) for value in current_ewma_scores.tolist())
        normalized_point_score_values.extend(
            float(value) for value in normalized_point_scores[0].tolist()
        )
        input_window_scores.extend(input_scores)
        normalized_input_window_scores.extend(normalized_input_scores)
        latent_window_scores.extend(latent_scores)
    return {
        "point": point_scores,
        "normalized_point": normalized_point_score_values,
        "input_window": [float(value) for value in input_window_scores],
        "normalized_input_window": normalized_input_window_scores,
        "latent_window": [float(value) for value in latent_window_scores],
    }


def _collect_offline_scores(
    model: torch.nn.Module,
    sequence: dict[str, Any],
    window_size: int,
    device: str,
    scaler: Any | None = None,
) -> list[float]:
    collected: list[float] = []
    windows = slice_sequence_into_windows(
        sequence, window_size=window_size, stride=window_size, tail_policy="end_align"
    )
    for window in windows:
        absolute_start_index = int(window["meta"]["absolute_start_index"])
        absolute_end_index = int(window["meta"]["absolute_end_index"])
        batch = {
            "x": window["x"].unsqueeze(0).to(device),
            "absolute_indices": torch.arange(
                absolute_start_index,
                absolute_end_index,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0),
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
        if scaler is None:
            window_scores = outputs["point_scores"]
        else:
            stochastic_query = outputs["aux"].get("stochastic_query") or {}
            reconstruction = stochastic_query.get("reconstruction_samples")
            if not isinstance(reconstruction, torch.Tensor):
                reconstruction = outputs["recon"]
            window_scores = score_reconstruction(
                batch["x"], reconstruction, scaler
            )["raw_input_point_mse"]
        collected.extend(window_scores.reshape(-1).detach().cpu().tolist())
    return collected


def collect_nonoverlap_offline_scores(
    *,
    model: torch.nn.Module,
    clean_validation_sequences: list[dict[str, Any]],
    window_size: int,
    device: str,
    scaler: Any | None = None,
) -> list[float]:
    """Collect offline calibration scores from non-overlapping validation windows."""
    collected: list[float] = []
    for sequence in clean_validation_sequences:
        collected.extend(
            _collect_offline_scores(
                model, sequence, window_size, device, scaler=scaler
            )
        )
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
    scaler: Any | None = None,
) -> dict[str, list[float]]:
    """Collect stride-1 causal validation scores for online thresholding."""
    collected = {
        key: []
        for key in (
            "point",
            "ewma",
            "normalized_point",
            "input_window",
            "normalized_input_window",
            "latent_window",
        )
    }
    for sequence in clean_validation_sequences:
        scores = run_stride1_sequence_scores(
            model=model,
            sequence=sequence,
            window_size=window_size,
            batch_size=batch_size,
            view_noise_std=view_noise_std,
            view_dropout_probability=view_dropout_probability,
            device=device,
            current_weight=current_weight,
            previous_weight=previous_weight,
            scaler=scaler,
        )
        collected["point"].extend(scores["point"])
        collected["ewma"].extend(scores["point"])
        collected["normalized_point"].extend(scores["normalized_point"])
        collected["input_window"].extend(scores["input_window"])
        collected["normalized_input_window"].extend(
            scores["normalized_input_window"]
        )
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
    scaler: Any | None = None,
) -> dict[str, list[float]]:
    collected = {"offline_point": []}
    collected["offline_point"] = collect_nonoverlap_offline_scores(
        model=model,
        clean_validation_sequences=clean_validation_sequences,
        window_size=window_size,
        device=device,
        scaler=scaler,
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
        scaler=scaler,
    )
    collected.update(stride1_scores)
    return collected

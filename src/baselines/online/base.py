from __future__ import annotations

"""Shared protocol and small helpers for online streaming baselines.

₍^. .^₎⟆ Online baseline flow

sequence
  -> stride-1 causal windows
  -> one scalar score per step
  -> EWMA calibration on clean validation
  -> benchmark records with the shared schema
"""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from src.data.stream import OnlineWindowBatcher, SMDOnlineStream
from src.protocols.point_scores import (
    ewma_scores,
    window_scores_to_causal_endpoint_scores,
)


def as_2d_sequence(sequence: Any) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array with shape [T, D]")
    return array


def build_stride1_window_matrix(
    sequence: Any,
    *,
    window_size: int,
    normalize: bool,
) -> tuple[np.ndarray, list[int]]:
    sequence_array = as_2d_sequence(sequence)
    if sequence_array.shape[0] < window_size:
        return np.zeros(
            (0, window_size * sequence_array.shape[1]), dtype=np.float64
        ), []

    window_starts = list(range(0, sequence_array.shape[0] - window_size + 1))
    window_rows = [
        sequence_array[start_index : start_index + window_size].reshape(-1)
        for start_index in window_starts
    ]
    window_matrix = np.stack(window_rows, axis=0).astype(np.float64, copy=False)
    if normalize:
        row_mean = np.nanmean(window_matrix, axis=1, keepdims=True)
        row_std = np.nanstd(window_matrix, axis=1, keepdims=True, ddof=1)
        row_std = np.where(np.isfinite(row_std) & (row_std >= 1.0e-6), row_std, 1.0)
        window_matrix = (window_matrix - row_mean) / row_std
    return window_matrix, window_starts


@runtime_checkable
class OnlineStreamingBaselineProtocol(Protocol):
    def calibrate(
        self,
        *,
        clean_validation_sequences: list[dict[str, Any]],
        protocol_config: dict[str, Any],
        device: str,
    ) -> dict[str, Any]: ...

    def run_sequence(
        self,
        *,
        sequence: dict[str, Any],
        threshold_value: float,
        protocol_config: dict[str, Any],
        device: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...


def build_stride1_batcher(
    *,
    sequence: dict[str, Any],
    window_size: int,
) -> Iterable[dict[str, Any]]:
    stream = SMDOnlineStream(
        sequences=[sequence],
        window_size=window_size,
        stride=1,
        clean_stream_only=True,
        stream_window_mode="sliding_stride_1",
    )
    return OnlineWindowBatcher(
        stream=stream,
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        include_legacy_views=False,
    )


def build_online_thresholds(
    *,
    threshold_value: float,
) -> dict[str, float]:
    return {
        "strong_anomaly_threshold": float(threshold_value),
        "hard_old_normality_threshold": float(threshold_value) * 0.5,
        "pnn_candidate_input_threshold": float(threshold_value) * 0.75,
        "pnn_candidate_latent_threshold": float(threshold_value) * 0.75,
    }


def smooth_point_scores(
    point_scores: np.ndarray,
    *,
    current_weight: float,
    previous_weight: float,
) -> np.ndarray:
    return ewma_scores(
        np.asarray(point_scores, dtype=np.float64).reshape(-1),
        current_weight=current_weight,
        previous_weight=previous_weight,
    )


def causal_point_scores_from_windows(
    *,
    window_scores: np.ndarray,
    sequence_length: int,
    window_size: int,
) -> np.ndarray:
    return window_scores_to_causal_endpoint_scores(
        window_scores=np.asarray(window_scores, dtype=np.float64).reshape(-1),
        sequence_length=sequence_length,
        window_size=window_size,
    )


def move_batch_to_device(
    batch: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def extract_last_point_label(batch: dict[str, Any]) -> int:
    labels = batch["point_labels"]
    if isinstance(labels, torch.Tensor):
        return int(labels[0, -1].detach().cpu())
    return int(np.asarray(labels)[0, -1])


def build_online_record_schema(
    *,
    entity_id: str,
    stream_step: int,
    window_start_index: int,
    window_end_index: int,
    raw_point_score: float,
    ewma_point_score: float,
    latent_window_score: float,
    threshold: float,
    prediction: int,
    online_variant: str,
    triage_decision: str,
    did_update: bool,
    loss_total: float | None,
    verification_buffer_size: int = 0,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "point_index": window_end_index - 1,
        "window_start_index": window_start_index,
        "window_end_index": window_end_index,
        "stream_step": stream_step,
        "raw_point_score": float(raw_point_score),
        "ewma_point_score": float(ewma_point_score),
        "latent_window_score": float(latent_window_score),
        "threshold": float(threshold),
        "prediction": int(prediction),
        "online_variant": online_variant,
        "triage_decision": triage_decision,
        "did_update": bool(did_update),
        "loss_total": None if loss_total is None else float(loss_total),
        "online/verification_buffer_size": int(verification_buffer_size),
    }

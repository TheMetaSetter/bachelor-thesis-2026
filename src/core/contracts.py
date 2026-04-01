from __future__ import annotations

from typing import Any

import torch


def _require_keys(container: dict[str, Any], required_keys: list[str], object_name: str) -> None:
    for required_key in required_keys:
        if required_key not in container:
            raise ValueError(f"{object_name} is missing required key: {required_key}")


def _require_tensor_rank(tensor: torch.Tensor, rank: int, tensor_name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor")
    if tensor.ndim != rank:
        raise ValueError(f"{tensor_name} must have rank {rank}, got {tensor.ndim}")


def _require_same_shape(
    first_tensor: torch.Tensor,
    second_tensor: torch.Tensor,
    first_name: str,
    second_name: str,
) -> None:
    if first_tensor.shape != second_tensor.shape:
        raise ValueError(
            f"{first_name} shape {tuple(first_tensor.shape)} must match "
            f"{second_name} shape {tuple(second_tensor.shape)}"
        )


def validate_raw_sequence(raw_sequence: dict[str, Any]) -> None:
    _require_keys(raw_sequence, ["x", "point_labels", "mask", "timestamps", "meta"], "raw_sequence")
    _require_tensor_rank(raw_sequence["x"], 2, "raw_sequence['x']")
    if raw_sequence["point_labels"] is not None:
        _require_tensor_rank(raw_sequence["point_labels"], 1, "raw_sequence['point_labels']")
        if raw_sequence["point_labels"].shape[0] != raw_sequence["x"].shape[0]:
            raise ValueError("point_labels length must match sequence length")
    if raw_sequence["mask"] is not None:
        _require_tensor_rank(raw_sequence["mask"], 2, "raw_sequence['mask']")
        if raw_sequence["mask"].shape != raw_sequence["x"].shape:
            raise ValueError("mask must match x shape")
    meta = raw_sequence["meta"]
    _require_keys(
        meta,
        ["dataset_name", "entity_id", "split", "num_channels", "sequence_length"],
        "raw_sequence['meta']",
    )


def validate_window(window: dict[str, Any]) -> None:
    _require_keys(window, ["x", "point_labels", "mask", "timestamps", "meta"], "window")
    _require_tensor_rank(window["x"], 2, "window['x']")
    meta = window["meta"]
    _require_keys(
        meta,
        ["dataset_name", "entity_id", "split", "start_index", "end_index", "window_size"],
        "window['meta']",
    )


def validate_batch(batch: dict[str, Any]) -> None:
    _require_keys(batch, ["x", "point_labels", "mask", "timestamps", "meta"], "batch")
    _require_tensor_rank(batch["x"], 3, "batch['x']")
    if batch["point_labels"] is not None:
        _require_tensor_rank(batch["point_labels"], 2, "batch['point_labels']")
    if batch["mask"] is not None:
        _require_tensor_rank(batch["mask"], 3, "batch['mask']")
    if batch["timestamps"] is not None:
        _require_tensor_rank(batch["timestamps"], 2, "batch['timestamps']")
    if not isinstance(batch["meta"], list):
        raise TypeError("batch['meta'] must be a list of dictionaries")


def validate_online_batch(batch: dict[str, Any]) -> None:
    validate_batch(batch)
    _require_keys(batch, ["view_a", "view_b"], "online_batch")
    _require_tensor_rank(batch["view_a"], 3, "online_batch['view_a']")
    _require_tensor_rank(batch["view_b"], 3, "online_batch['view_b']")
    _require_same_shape(batch["view_a"], batch["x"], "online_batch['view_a']", "online_batch['x']")
    _require_same_shape(batch["view_b"], batch["x"], "online_batch['view_b']", "online_batch['x']")


def validate_model_outputs(outputs: dict[str, Any]) -> None:
    _require_keys(
        outputs,
        ["hidden", "pooled", "recon", "logits", "point_scores", "window_scores", "aux"],
        "outputs",
    )
    _require_tensor_rank(outputs["hidden"], 3, "outputs['hidden']")
    if outputs["recon"] is not None:
        _require_tensor_rank(outputs["recon"], 3, "outputs['recon']")
    if outputs["point_scores"] is not None:
        _require_tensor_rank(outputs["point_scores"], 2, "outputs['point_scores']")
    if outputs["window_scores"] is not None:
        _require_tensor_rank(outputs["window_scores"], 1, "outputs['window_scores']")
    if not isinstance(outputs["aux"], dict):
        raise TypeError("outputs['aux'] must be a dictionary")


def validate_evaluation_record(evaluation_record: dict[str, Any]) -> None:
    _require_keys(
        evaluation_record,
        ["entity_id", "point_scores", "point_labels", "num_points"],
        "evaluation_record",
    )
    _require_tensor_rank(evaluation_record["point_scores"], 1, "evaluation_record['point_scores']")
    _require_tensor_rank(evaluation_record["point_labels"], 1, "evaluation_record['point_labels']")
    if evaluation_record["point_scores"].shape[0] != evaluation_record["num_points"]:
        raise ValueError("point_scores length must equal num_points")

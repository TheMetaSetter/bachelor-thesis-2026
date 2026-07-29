from __future__ import annotations

"""Runtime contracts shared by the active thesis pipelines.

A new reader should start here when they want to understand what every model,
loader, and engine loop agrees on. These validators keep the offline baseline,
the multitask model, and the online adaptation path readable because they all
reuse the same batch and output vocabulary.
"""

from typing import Any

import torch


def _require_keys(
    container: dict[str, Any], required_keys: list[str], object_name: str
) -> None:
    for required_key in required_keys:
        if required_key not in container:
            raise ValueError(f"{object_name} is missing required key: {required_key}")


def _require_tensor_rank(tensor: torch.Tensor, rank: int, tensor_name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor")
    if tensor.ndim != rank:
        raise ValueError(f"{tensor_name} must have rank {rank}, got {tensor.ndim}")


def _require_optional_tensor_rank(
    tensor: torch.Tensor | None, rank: int, tensor_name: str
) -> None:
    if tensor is None:
        return
    _require_tensor_rank(tensor, rank, tensor_name)


def validate_raw_sequence(raw_sequence: dict[str, Any]) -> None:
    # Raw sequence validation happens before windowing so every later stage can
    # assume each entity already has the same basic fields and metadata keys.
    _require_keys(
        raw_sequence,
        ["x", "point_labels", "mask", "timestamps", "meta"],
        "raw_sequence",
    )
    _require_tensor_rank(raw_sequence["x"], 2, "raw_sequence['x']")
    if raw_sequence["point_labels"] is not None:
        _require_tensor_rank(
            raw_sequence["point_labels"], 1, "raw_sequence['point_labels']"
        )
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
    # A window is the bridge between full-length entity sequences and the fixed
    # `[B, L, D]` batch contract used everywhere else in the repository.
    _require_keys(window, ["x", "point_labels", "mask", "timestamps", "meta"], "window")
    _require_tensor_rank(window["x"], 2, "window['x']")
    meta = window["meta"]
    _require_keys(
        meta,
        [
            "dataset_name",
            "entity_id",
            "split",
            "start_index",
            "end_index",
            "window_size",
            "series_id",
            "absolute_start_index",
            "absolute_end_index",
            "source_sequence_length",
        ],
        "window['meta']",
    )


def validate_batch(batch: dict[str, Any]) -> None:
    # The offline batch contract is intentionally stable so the baseline model,
    # the multitask model, and the engine code can all share one reading path.
    # Synthetic augmentation may append metadata and binary supervision, but it
    # must not replace these base fields.
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
    # Full-spec-v2 uses exactly one input window. Online-only fields such as a
    # PNN mask may be appended later, but two augmented views are not required.
    validate_batch(batch)


def validate_model_outputs(outputs: dict[str, Any]) -> None:
    # The output contract is kept fixed at the top level so evaluators,
    # checkpoint consumers, and future readers do not need model-specific branches.
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
    stochastic_query = outputs["aux"].get("stochastic_query")
    if stochastic_query is not None:
        validate_stochastic_query_aux(stochastic_query)
    uncertainty = outputs["aux"].get("uncertainty")
    if uncertainty is not None:
        validate_uncertainty_aux(uncertainty)
    deterministic_geometry = outputs["aux"].get("deterministic_geometry")
    if deterministic_geometry is not None:
        validate_deterministic_geometry_aux(deterministic_geometry)


def validate_stochastic_query_aux(stochastic_query: dict[str, Any]) -> None:
    _require_keys(
        stochastic_query,
        ["schema_version", "enabled", "num_samples"],
        "outputs['aux']['stochastic_query']",
    )
    if int(stochastic_query["schema_version"]) != 3:
        raise ValueError("stochastic_query.schema_version must be 3")
    if not isinstance(stochastic_query["enabled"], bool):
        raise TypeError("stochastic_query.enabled must be a boolean")
    if not isinstance(stochastic_query["num_samples"], int) or (
        stochastic_query["num_samples"] <= 0
    ):
        raise ValueError("stochastic_query.num_samples must be a positive integer")
    num_samples = int(stochastic_query["num_samples"])
    for field_name in ["continuous_temperature", "discrete_temperature"]:
        field_value = stochastic_query.get(field_name)
        if field_value is not None and float(field_value) <= 0.0:
            raise ValueError(f"stochastic_query.{field_name} must be positive")
    for field_name in [
        "continuous_retrieved_samples",
        "discrete_retrieved_samples",
        "discrete_topk_ids",
        "reconstruction_samples",
        "classification_probability_samples",
        "point_score_samples",
        "window_score_samples",
    ]:
        field_value = stochastic_query.get(field_name)
        if field_value is not None and not isinstance(field_value, torch.Tensor):
            raise TypeError(
                f"stochastic_query.{field_name} must be a torch.Tensor or null"
            )
    sample_rank_contracts = {
        "continuous_retrieved_samples": 4,
        "discrete_retrieved_samples": 4,
        "discrete_topk_ids": 4,
        "reconstruction_samples": 4,
        "classification_probability_samples": 3,
        "point_score_samples": 3,
        "window_score_samples": 2,
    }
    for field_name, expected_rank in sample_rank_contracts.items():
        field_value = stochastic_query.get(field_name)
        if field_value is None:
            continue
        _require_tensor_rank(
            field_value,
            expected_rank,
            f"outputs['aux']['stochastic_query']['{field_name}']",
        )
        if int(field_value.shape[1]) != num_samples:
            raise ValueError(
                f"stochastic_query.{field_name} must have sample axis equal to num_samples"
            )
        if not torch.isfinite(field_value.float()).all().item():
            raise ValueError(
                f"stochastic_query.{field_name} must contain only finite values"
            )
    if not stochastic_query.get("enabled") and any(
        stochastic_query.get(field_name) is not None
        for field_name in sample_rank_contracts
    ):
        raise ValueError(
            "stochastic_query sample tensors require stochastic_query.enabled=True"
        )


def validate_uncertainty_aux(uncertainty: dict[str, Any]) -> None:
    for field_name in [
        "point_anomaly_score_variance",
        "continuous_retrieval_variance_point",
        "discrete_retrieval_variance_point",
        "reconstruction_variance_point",
    ]:
        tensor_value = uncertainty.get(field_name)
        _require_optional_tensor_rank(
            tensor_value, 2, f"outputs['aux']['uncertainty']['{field_name}']"
        )
        if (
            tensor_value is not None
            and not torch.isfinite(tensor_value.float()).all().item()
        ):
            raise ValueError(
                f"outputs['aux']['uncertainty']['{field_name}'] must contain only finite values"
            )
    for field_name in [
        "window_anomaly_score_variance",
        "continuous_retrieval_variance_window",
        "discrete_retrieval_variance_window",
        "reconstruction_variance_window",
        "classification_variance_mean",
    ]:
        tensor_value = uncertainty.get(field_name)
        _require_optional_tensor_rank(
            tensor_value, 1, f"outputs['aux']['uncertainty']['{field_name}']"
        )
        if (
            tensor_value is not None
            and not torch.isfinite(tensor_value.float()).all().item()
        ):
            raise ValueError(
                f"outputs['aux']['uncertainty']['{field_name}'] must contain only finite values"
            )
    _require_optional_tensor_rank(
        uncertainty.get("reconstruction_variance_full"),
        3,
        "outputs['aux']['uncertainty']['reconstruction_variance_full']",
    )
    reconstruction_variance_full = uncertainty.get("reconstruction_variance_full")
    if (
        reconstruction_variance_full is not None
        and not torch.isfinite(reconstruction_variance_full.float()).all().item()
    ):
        raise ValueError(
            "outputs['aux']['uncertainty']['reconstruction_variance_full'] must contain only finite values"
        )
    _require_optional_tensor_rank(
        uncertainty.get("classification_probability_variance"),
        2,
        "outputs['aux']['uncertainty']['classification_probability_variance']",
    )
    classification_probability_variance = uncertainty.get(
        "classification_probability_variance"
    )
    if (
        classification_probability_variance is not None
        and not torch.isfinite(classification_probability_variance.float()).all().item()
    ):
        raise ValueError(
            "outputs['aux']['uncertainty']['classification_probability_variance'] must contain only finite values"
        )


def validate_deterministic_geometry_aux(deterministic_geometry: dict[str, Any]) -> None:
    for field_name in [
        "nearest_codeword_ids",
        "nearest_codeword_distances",
        "known_anomaly_mask",
        "continuous_signature_ids",
        "latent_window_score",
    ]:
        field_value = deterministic_geometry.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, torch.Tensor):
            raise TypeError(
                f"outputs['aux']['deterministic_geometry']['{field_name}'] must be a torch.Tensor or null"
            )
    if deterministic_geometry.get("nearest_codeword_ids") is not None:
        _require_tensor_rank(
            deterministic_geometry["nearest_codeword_ids"],
            2,
            "outputs['aux']['deterministic_geometry']['nearest_codeword_ids']",
        )
    if deterministic_geometry.get("nearest_codeword_distances") is not None:
        _require_tensor_rank(
            deterministic_geometry["nearest_codeword_distances"],
            2,
            "outputs['aux']['deterministic_geometry']['nearest_codeword_distances']",
        )
    if deterministic_geometry.get("known_anomaly_mask") is not None:
        _require_tensor_rank(
            deterministic_geometry["known_anomaly_mask"],
            2,
            "outputs['aux']['deterministic_geometry']['known_anomaly_mask']",
        )
    if deterministic_geometry.get("continuous_signature_ids") is not None:
        _require_tensor_rank(
            deterministic_geometry["continuous_signature_ids"],
            3,
            "outputs['aux']['deterministic_geometry']['continuous_signature_ids']",
        )
    if deterministic_geometry.get("latent_window_score") is not None:
        _require_tensor_rank(
            deterministic_geometry["latent_window_score"],
            1,
            "outputs['aux']['deterministic_geometry']['latent_window_score']",
        )


def validate_evaluation_record(evaluation_record: dict[str, Any]) -> None:
    # Evaluation records are stored per entity after overlapping window scores
    # have been merged back onto the original time axis.
    _require_keys(
        evaluation_record,
        ["entity_id", "point_scores", "point_labels", "num_points"],
        "evaluation_record",
    )
    _require_tensor_rank(
        evaluation_record["point_scores"], 1, "evaluation_record['point_scores']"
    )
    _require_tensor_rank(
        evaluation_record["point_labels"], 1, "evaluation_record['point_labels']"
    )
    if evaluation_record["point_scores"].shape[0] != evaluation_record["num_points"]:
        raise ValueError("point_scores length must equal num_points")
    covered_point_mask = evaluation_record.get("covered_point_mask")
    if covered_point_mask is not None:
        _require_tensor_rank(
            covered_point_mask, 1, "evaluation_record['covered_point_mask']"
        )
        if covered_point_mask.shape[0] != evaluation_record["num_points"]:
            raise ValueError("covered_point_mask length must equal num_points")

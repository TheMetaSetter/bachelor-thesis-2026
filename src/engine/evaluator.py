from __future__ import annotations

"""Evaluation loop that merges overlapping window scores back to entity timelines.

A new reader should pair this file with the batch and output contracts. The
evaluator relies on the fixed `point_scores` field so it can stay model-agnostic.
"""

from typing import Any

import numpy as np
import torch

from src.core.console import console_print, summarize_batch, summarize_tensor
from src.core.contracts import validate_evaluation_record
from src.metrics.pointwise import (
    compute_pointwise_curve_payload,
    compute_pointwise_metrics,
)
from src.models.base_model import BaseModel


def select_point_score_threshold(
    point_scores: np.ndarray, quantile: float = 0.95
) -> float:
    """
    Hàm này có nhiệm vụ chọn threshold
    để biến anomaly score liên tục
    thành dự đoán nhị phân.
    """

    # The smoke runs can produce many exact zeros, so selecting a threshold from
    # only the positive support avoids the "everything is anomalous" failure
    # mode in plots and thresholded metrics.
    positive_scores = point_scores[point_scores > 0.0]
    reference_scores = positive_scores if positive_scores.size > 0 else point_scores
    threshold = float(np.quantile(reference_scores, quantile))

    if threshold <= 0.0 and positive_scores.size > 0:
        threshold = float(np.min(positive_scores))
    return threshold


def accumulate_pointwise_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payload: dict[str, Any],
    entity_score_sums: dict[str, torch.Tensor],
    entity_score_counts: dict[str, torch.Tensor],
    entity_point_labels: dict[str, torch.Tensor],
) -> None:
    batch_meta = batch_payload["meta"]
    point_scores = batch_payload["point_scores"]
    point_labels = batch_payload["point_labels"]

    if point_scores.ndim != 2:
        raise ValueError("batch_payload['point_scores'] must have shape [B, L]")
    if point_labels.ndim != 2:
        raise ValueError("batch_payload['point_labels'] must have shape [B, L]")
    if point_scores.shape != point_labels.shape:
        raise ValueError(
            "batch_payload['point_scores'] and batch_payload['point_labels'] must share the same shape"
        )
    if len(batch_meta) != point_scores.shape[0]:
        raise ValueError("batch_payload['meta'] length must match batch size")

    for window_index, meta in enumerate(batch_meta):
        entity_id = meta["entity_id"]
        start_index = int(meta["start_index"])
        end_index = int(meta["end_index"])
        sequence_length = int(sequences_by_entity[entity_id]["x"].shape[0])

        if entity_id not in entity_score_sums:
            entity_score_sums[entity_id] = torch.zeros(
                sequence_length,
                dtype=torch.float32,
            )
            entity_score_counts[entity_id] = torch.zeros(
                sequence_length,
                dtype=torch.float32,
            )
            entity_point_labels[entity_id] = torch.zeros(
                sequence_length,
                dtype=point_labels.dtype,
            )

        entity_score_sums[entity_id][start_index:end_index] += point_scores[
            window_index
        ]
        entity_score_counts[entity_id][start_index:end_index] += 1.0
        entity_point_labels[entity_id][start_index:end_index] = torch.maximum(
            entity_point_labels[entity_id][start_index:end_index],
            point_labels[window_index].to(entity_point_labels[entity_id].dtype),
        )


def reconstruct_pointwise_records_from_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_score_sums: dict[str, torch.Tensor] = {}
    entity_score_counts: dict[str, torch.Tensor] = {}
    entity_point_labels: dict[str, torch.Tensor] = {}

    for batch_payload in batch_payloads:
        accumulate_pointwise_window_payload(
            sequences_by_entity=sequences_by_entity,
            batch_payload=batch_payload,
            entity_score_sums=entity_score_sums,
            entity_score_counts=entity_score_counts,
            entity_point_labels=entity_point_labels,
        )

    evaluation_records: list[dict[str, Any]] = []
    for entity_id, score_sum in entity_score_sums.items():
        counts = torch.clamp(entity_score_counts[entity_id], min=1.0)
        averaged_scores = score_sum / counts
        evaluation_record = {
            "entity_id": entity_id,
            "point_scores": averaged_scores,
            "point_labels": entity_point_labels[entity_id],
            "num_points": int(averaged_scores.shape[0]),
        }
        validate_evaluation_record(evaluation_record)
        evaluation_records.append(evaluation_record)

    return evaluation_records


class Evaluator:
    def __init__(
        self,
        device: str = "cpu",
        vus_max_buffer_size: int | None = None,
        vus_num_thresholds: int = 200,
    ) -> None:
        self.device = device
        self.vus_max_buffer_size = vus_max_buffer_size
        self.vus_num_thresholds = vus_num_thresholds

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Only tensors need device placement; metadata stays as plain Python
        # values so the aggregation code can read entity ids and indices easily.
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _run_model_on_batch(
        self,
        model: BaseModel,
        batch_index: int,
        batch_on_device: dict[str, Any],
    ) -> dict[str, Any]:
        console_print(
            "EVAL",
            "Evaluating batch",
            batch_index=batch_index,
            **summarize_batch(batch_on_device),
        )

        # Tính anomaly score cho từng điểm dữ liệu (timestep)
        # trong từng window của batch.
        return model.test_step(batch_on_device)

    @staticmethod
    def _remember_forward_pass_seconds(
        step_output: dict[str, Any],
        forward_pass_seconds_history: list[float],
    ) -> None:
        output_aux = step_output["outputs"]["aux"]
        if "forward_pass_seconds" in output_aux:
            forward_pass_seconds_history.append(
                float(output_aux["forward_pass_seconds"])
            )

    @staticmethod
    def _log_batch_outputs(
        batch_index: int,
        step_output: dict[str, Any],
        point_scores: torch.Tensor,
    ) -> None:
        console_print(
            "EVAL",
            "Produced evaluation batch outputs",
            batch_index=batch_index,
            point_scores=summarize_tensor(point_scores),
            window_scores=summarize_tensor(step_output["outputs"]["window_scores"]),
        )

    @staticmethod
    def _build_sequences_by_entity(data_loader: Any) -> dict[str, dict[str, Any]]:
        return {
            sequence["meta"]["entity_id"]: sequence
            for sequence in data_loader.dataset.sequences
        }

    def evaluate(self, model: BaseModel, data_loader: Any) -> dict[str, Any]:
        # Window-level scores are accumulated back onto each entity because the
        # downstream metrics should be interpreted on the original timeline.
        model.to(self.device)
        model.eval()
        console_print(
            "EVAL",
            "Starting evaluation",
            device=self.device,
            num_batches=len(data_loader),
        )
        forward_pass_seconds_history: list[float] = []
        sequences_by_entity = self._build_sequences_by_entity(data_loader)
        pointwise_batch_payloads: list[dict[str, Any]] = []

        with torch.no_grad():
            # Với mỗi batch dữ liệu đọc được từ data_loader,
            for batch_index, batch in enumerate(data_loader, start=1):
                batch_on_device = self._move_batch_to_device(batch)
                step_output = self._run_model_on_batch(
                    model=model,
                    batch_index=batch_index,
                    batch_on_device=batch_on_device,
                )
                point_scores = step_output["outputs"]["point_scores"].detach().cpu()
                self._remember_forward_pass_seconds(
                    step_output=step_output,
                    forward_pass_seconds_history=forward_pass_seconds_history,
                )
                self._log_batch_outputs(
                    batch_index=batch_index,
                    step_output=step_output,
                    point_scores=point_scores,
                )
                pointwise_batch_payloads.append(
                    {
                        "meta": batch["meta"],
                        "point_scores": point_scores,
                        "point_labels": batch["point_labels"].detach().cpu(),
                    }
                )

        evaluation_records = reconstruct_pointwise_records_from_window_payload(
            sequences_by_entity=sequences_by_entity,
            batch_payloads=pointwise_batch_payloads,
        )
        all_point_scores: list[np.ndarray] = []
        all_point_labels: list[np.ndarray] = []
        for evaluation_record in evaluation_records:
            all_point_scores.append(evaluation_record["point_scores"].numpy())
            all_point_labels.append(evaluation_record["point_labels"].numpy())

        # Nối tất cả các điểm anomaly score từ từng timestep lại thành một danh sách
        concatenated_scores = np.concatenate(all_point_scores, axis=0)

        # Nối nhãn anomaly của từng timestep lại thành một danh sách
        concatenated_labels = np.concatenate(all_point_labels, axis=0)

        # Chọn threshold cho anomaly score
        # Một timestep có anomaly score vượt ngưỡng threshold này thì tính là anomaly
        threshold = select_point_score_threshold(concatenated_scores, quantile=0.95)

        # Tính toán các độ đo pointwise (pointwise metric)
        metrics = compute_pointwise_metrics(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
            threshold=threshold,
            vus_max_buffer_size=self.vus_max_buffer_size,
            vus_num_thresholds=self.vus_num_thresholds,
        )

        metrics["threshold"] = threshold
        if forward_pass_seconds_history:
            metrics["forward_pass_seconds_mean"] = sum(
                forward_pass_seconds_history
            ) / len(forward_pass_seconds_history)
        curves = compute_pointwise_curve_payload(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
        )
        console_print(
            "EVAL",
            "Completed evaluation",
            num_records=len(evaluation_records),
            concatenated_scores_length=len(concatenated_scores),
            concatenated_labels_length=len(concatenated_labels),
            threshold=threshold,
            metrics=metrics,
            roc_curve_points=len(curves["roc_curve"]["x"]),
            pr_curve_points=len(curves["pr_curve"]["x"]),
        )

        return {
            "metrics": metrics,
            "records": evaluation_records,
            "curves": curves,
        }

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.core.contracts import validate_evaluation_record
from src.metrics.pointwise import compute_pointwise_metrics
from src.models.base_model import BaseModel


class Evaluator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def evaluate(self, model: BaseModel, data_loader: Any) -> dict[str, Any]:
        model.to(self.device)
        model.eval()
        entity_score_sums: dict[str, torch.Tensor] = {}
        entity_score_counts: dict[str, torch.Tensor] = {}
        entity_labels: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for batch in data_loader:
                batch_on_device = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                step_output = model.test_step(batch_on_device)
                point_scores = step_output["outputs"]["point_scores"].detach().cpu()

                for batch_index, meta in enumerate(batch["meta"]):
                    entity_id = meta["entity_id"]
                    end_index = int(meta["end_index"])
                    start_index = int(meta["start_index"])
                    entity_length = int(
                        max(
                            end_index,
                            batch["point_labels"][batch_index].shape[0] + start_index,
                        )
                    )
                    if entity_id not in entity_score_sums:
                        sequence_length = next(
                            sequence["x"].shape[0]
                            for sequence in data_loader.dataset.sequences
                            if sequence["meta"]["entity_id"] == entity_id
                        )
                        entity_score_sums[entity_id] = torch.zeros(sequence_length, dtype=torch.float32)
                        entity_score_counts[entity_id] = torch.zeros(sequence_length, dtype=torch.float32)
                        entity_labels[entity_id] = next(
                            sequence["point_labels"].clone()
                            for sequence in data_loader.dataset.sequences
                            if sequence["meta"]["entity_id"] == entity_id
                        )

                    entity_score_sums[entity_id][start_index:end_index] += point_scores[batch_index]
                    entity_score_counts[entity_id][start_index:end_index] += 1.0

        evaluation_records: list[dict[str, Any]] = []
        all_point_scores: list[np.ndarray] = []
        all_point_labels: list[np.ndarray] = []

        for entity_id, score_sum in entity_score_sums.items():
            counts = torch.clamp(entity_score_counts[entity_id], min=1.0)
            averaged_scores = score_sum / counts
            evaluation_record = {
                "entity_id": entity_id,
                "point_scores": averaged_scores,
                "point_labels": entity_labels[entity_id],
                "num_points": int(averaged_scores.shape[0]),
            }
            validate_evaluation_record(evaluation_record)
            evaluation_records.append(evaluation_record)
            all_point_scores.append(averaged_scores.numpy())
            all_point_labels.append(entity_labels[entity_id].numpy())

        concatenated_scores = np.concatenate(all_point_scores, axis=0)
        concatenated_labels = np.concatenate(all_point_labels, axis=0)
        threshold = float(np.quantile(concatenated_scores, 0.95))
        metrics = compute_pointwise_metrics(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
            threshold=threshold,
        )
        metrics["threshold"] = threshold

        return {
            "metrics": metrics,
            "records": evaluation_records,
        }

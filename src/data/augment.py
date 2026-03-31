from __future__ import annotations

from typing import Any

import torch


class SyntheticAnomalyInjector:
    def __init__(
        self,
        anomaly_probability: float = 0.5,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
    ) -> None:
        if not 0.0 <= anomaly_probability <= 1.0:
            raise ValueError("anomaly_probability must be between 0 and 1")
        if not 0.0 < max_segment_fraction <= 1.0:
            raise ValueError("max_segment_fraction must be between 0 and 1")
        if spike_scale <= 0.0:
            raise ValueError("spike_scale must be positive")

        self.anomaly_probability = anomaly_probability
        self.max_segment_fraction = max_segment_fraction
        self.spike_scale = spike_scale

    def _clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = [dict(item) if isinstance(item, dict) else item for item in value]
            else:
                cloned_batch[key] = value
        return cloned_batch

    def _inject_single_window(self, clean_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        window_size, num_channels = clean_window.shape
        segment_length = max(1, int(window_size * self.max_segment_fraction))
        segment_length = min(segment_length, window_size)
        max_start_index = max(window_size - segment_length, 0)
        start_index = int(torch.randint(0, max_start_index + 1, (1,), device=clean_window.device).item())
        end_index = start_index + segment_length
        channel_index = int(torch.randint(0, num_channels, (1,), device=clean_window.device).item())
        anomaly_type_index = int(torch.randint(0, 3, (1,), device=clean_window.device).item())

        augmented_window = clean_window.clone()
        anomaly_mask = torch.zeros(window_size, dtype=torch.long, device=clean_window.device)
        anomaly_mask[start_index:end_index] = 1

        if anomaly_type_index == 0:
            segment_values = augmented_window[start_index:end_index, channel_index]
            segment_std = torch.std(segment_values, unbiased=False)
            spike = self.spike_scale * (segment_std + 1e-6)
            augmented_window[start_index:end_index, channel_index] = segment_values + spike
            anomaly_type = "spike"
        elif anomaly_type_index == 1:
            augmented_window[start_index:end_index, channel_index] = 0.0
            anomaly_type = "dropout"
        else:
            shift = augmented_window[:, channel_index].mean()
            augmented_window[start_index:end_index, channel_index] = (
                augmented_window[start_index:end_index, channel_index] - shift
            )
            anomaly_type = "level_shift"

        augmentation_metadata = {
            "anomaly_type": anomaly_type,
            "channel_index": channel_index,
            "start_index": start_index,
            "end_index": end_index,
        }
        return augmented_window, anomaly_mask, augmentation_metadata

    def augment_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "x" not in batch:
            raise ValueError("batch must contain 'x'")
        if batch["x"].ndim != 3:
            raise ValueError("batch['x'] must have shape [B, L, D]")

        augmented_batch = self._clone_batch(batch)
        clean_windows = batch["x"]
        batch_size, window_size, _ = clean_windows.shape

        anomaly_masks = torch.zeros(batch_size, window_size, dtype=torch.long, device=clean_windows.device)
        classification_labels = torch.zeros(batch_size, dtype=torch.long, device=clean_windows.device)
        augmentation_metadata: list[dict[str, Any]] = []

        for batch_index in range(batch_size):
            should_inject = bool(torch.rand(1, device=clean_windows.device).item() < self.anomaly_probability)
            if not should_inject:
                augmentation_metadata.append(
                    {
                        "is_synthetic_anomaly": False,
                        "anomaly_type": "clean",
                        "channel_index": None,
                        "start_index": None,
                        "end_index": None,
                    }
                )
                continue

            augmented_window, anomaly_mask, window_metadata = self._inject_single_window(clean_windows[batch_index])
            augmented_batch["x"][batch_index] = augmented_window
            anomaly_masks[batch_index] = anomaly_mask
            classification_labels[batch_index] = 1
            augmentation_metadata.append({"is_synthetic_anomaly": True, **window_metadata})

        original_point_labels = batch.get("point_labels")
        if original_point_labels is None:
            augmented_batch["point_labels"] = anomaly_masks
        else:
            augmented_batch["point_labels"] = torch.maximum(original_point_labels.clone(), anomaly_masks)

        augmented_batch["classification_labels"] = classification_labels
        augmented_batch["synthetic_anomaly_mask"] = anomaly_masks
        augmented_batch["augmentation_metadata"] = augmentation_metadata
        return augmented_batch

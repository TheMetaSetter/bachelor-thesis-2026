from __future__ import annotations

from typing import Any

import torch


class SyntheticAnomalyInjector:
    def __init__(
        self,
        anomaly_probability: float = 0.5,
        min_segment_fraction: float = 0.1,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
        anomaly_families: tuple[str, ...] = (
            "seasonal",
            "trend",
            "global",
            "contextual",
            "shapelet",
        ),
    ) -> None:
        if not 0.0 <= anomaly_probability <= 1.0:
            raise ValueError("anomaly_probability must be between 0 and 1")
        if not 0.0 < min_segment_fraction <= 1.0:
            raise ValueError("min_segment_fraction must be between 0 and 1")
        if not 0.0 < max_segment_fraction <= 1.0:
            raise ValueError("max_segment_fraction must be between 0 and 1")
        if min_segment_fraction > max_segment_fraction:
            raise ValueError("min_segment_fraction must not exceed max_segment_fraction")
        if spike_scale <= 0.0:
            raise ValueError("spike_scale must be positive")
        if not anomaly_families:
            raise ValueError("anomaly_families must not be empty")

        self.anomaly_probability = anomaly_probability
        self.min_segment_fraction = min_segment_fraction
        self.max_segment_fraction = max_segment_fraction
        self.spike_scale = spike_scale
        self.anomaly_families = anomaly_families

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

    def _sample_segment_bounds(self, window_size: int, device: torch.device) -> tuple[int, int]:
        min_segment_length = max(1, int(window_size * self.min_segment_fraction))
        max_segment_length = max(min_segment_length, int(window_size * self.max_segment_fraction))
        max_segment_length = min(max_segment_length, window_size)
        segment_length = int(
            torch.randint(min_segment_length, max_segment_length + 1, (1,), device=device).item()
        )
        max_start_index = max(window_size - segment_length, 0)
        start_index = int(torch.randint(0, max_start_index + 1, (1,), device=device).item())
        end_index = start_index + segment_length
        return start_index, end_index

    def _sample_affected_channels(self, num_channels: int, device: torch.device) -> list[int]:
        min_channels = max(1, num_channels // 10)
        max_channels = max(min_channels, num_channels // 2)
        num_selected_channels = int(torch.randint(min_channels, max_channels + 1, (1,), device=device).item())
        selected_indices = torch.randperm(num_channels, device=device)[:num_selected_channels]
        return [int(index.item()) for index in selected_indices.sort().values]

    def _carla_repeat_and_subsample(
        self,
        subsequence: torch.Tensor,
        compression_factor: int,
    ) -> torch.Tensor:
        repeated_subsequence = subsequence.repeat(compression_factor, 1)
        compressed_subsequence = repeated_subsequence[::compression_factor]
        return compressed_subsequence[: subsequence.shape[0]]

    def _inject_family_on_subsequence(
        self,
        clean_channel_window: torch.Tensor,
        anomaly_family: str,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, int, dict[str, Any]]:
        anomalous_channel_window = clean_channel_window.clone()
        anomalous_subsequence = anomalous_channel_window[start_index:end_index].clone()
        family_parameters: dict[str, Any] = {
            "compression_factor": 1,
            "scale_factor": 1.0,
            "trend_factor": 0.0,
            "trend_end": False,
            "shapelet_factor": False,
        }

        if anomaly_family == "seasonal":
            family_parameters["compression_factor"] = int(
                torch.randint(2, 5, (1,), device=clean_channel_window.device).item()
            )
        elif anomaly_family == "trend":
            family_parameters["trend_end"] = True
            family_parameters["trend_factor"] = float(torch.normal(1.0, 0.5, size=(1,)).item())
        elif anomaly_family == "global":
            family_parameters["scale_factor"] = float(self.spike_scale * 2.0)
        elif anomaly_family == "contextual":
            family_parameters["scale_factor"] = float(self.spike_scale)
        elif anomaly_family == "shapelet":
            family_parameters["shapelet_factor"] = True
        else:
            raise ValueError(f"Unsupported anomaly family: {anomaly_family}")

        if family_parameters["trend_end"]:
            end_index = clean_channel_window.shape[0]
            anomalous_subsequence = anomalous_channel_window[start_index:end_index].clone()

        compression_factor = int(family_parameters["compression_factor"])
        if compression_factor > 1:
            anomalous_subsequence = self._carla_repeat_and_subsample(
                anomalous_subsequence,
                compression_factor=compression_factor,
            )

        anomalous_subsequence = anomalous_subsequence * float(family_parameters["scale_factor"])

        if float(family_parameters["trend_factor"]) != 0.0:
            trend_sign = -1.0 if bool(torch.rand(1).item() < 0.5) else 1.0
            trend_ramp = torch.linspace(
                0.0,
                trend_sign * float(family_parameters["trend_factor"]),
                anomalous_subsequence.shape[0],
                device=clean_channel_window.device,
            ).unsqueeze(-1)
            anomalous_subsequence = anomalous_subsequence + trend_ramp

        if bool(family_parameters["shapelet_factor"]):
            anchor_value = anomalous_channel_window[start_index].unsqueeze(0)
            noise = torch.rand_like(anomalous_subsequence) * 0.1
            anomalous_subsequence = anchor_value + noise

        anomalous_channel_window[start_index:end_index] = anomalous_subsequence
        family_parameters["effective_end_index"] = end_index
        return anomalous_channel_window, end_index, family_parameters

    def _inject_single_window(
        self,
        clean_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        window_size, num_channels = clean_window.shape
        start_index, sampled_end_index = self._sample_segment_bounds(window_size, clean_window.device)
        anomaly_family_index = int(
            torch.randint(0, len(self.anomaly_families), (1,), device=clean_window.device).item()
        )
        anomaly_family = self.anomaly_families[anomaly_family_index]
        affected_channels = self._sample_affected_channels(num_channels, clean_window.device)

        augmented_window = clean_window.clone()
        family_parameters_by_channel: dict[str, dict[str, Any]] = {}
        effective_end_index = sampled_end_index

        for channel_index in affected_channels:
            channel_window = augmented_window[:, channel_index : channel_index + 1]
            anomalous_channel_window, effective_channel_end_index, family_parameters = (
                self._inject_family_on_subsequence(
                    clean_channel_window=channel_window,
                    anomaly_family=anomaly_family,
                    start_index=start_index,
                    end_index=sampled_end_index,
                )
            )
            augmented_window[:, channel_index] = anomalous_channel_window.squeeze(-1)
            family_parameters_by_channel[str(channel_index)] = family_parameters
            effective_end_index = max(effective_end_index, effective_channel_end_index)

        anomaly_mask = torch.zeros(window_size, dtype=torch.long, device=clean_window.device)
        anomaly_mask[start_index:effective_end_index] = 1

        augmentation_metadata = {
            "is_synthetic_anomaly": True,
            "anomaly_family": anomaly_family,
            "start_index": start_index,
            "end_index": effective_end_index,
            "affected_channels": affected_channels,
            "family_parameters_by_channel": family_parameters_by_channel,
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
                        "anomaly_family": "clean",
                        "start_index": None,
                        "end_index": None,
                        "affected_channels": [],
                        "family_parameters_by_channel": {},
                    }
                )
                continue

            augmented_window, anomaly_mask, window_metadata = self._inject_single_window(clean_windows[batch_index])
            augmented_batch["x"][batch_index] = augmented_window
            anomaly_masks[batch_index] = anomaly_mask
            classification_labels[batch_index] = 1
            augmentation_metadata.append(window_metadata)

        original_point_labels = batch.get("point_labels")
        if original_point_labels is None:
            augmented_batch["point_labels"] = anomaly_masks
        else:
            augmented_batch["point_labels"] = torch.maximum(original_point_labels.clone(), anomaly_masks)

        augmented_batch["classification_labels"] = classification_labels
        augmented_batch["synthetic_anomaly_mask"] = anomaly_masks
        augmented_batch["augmentation_metadata"] = augmentation_metadata
        return augmented_batch

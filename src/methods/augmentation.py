"""Add the RedLamp anomaly families to a training batch."""

from typing import Any

import torch

from . import anomaly_families


_FAMILIES = (
    "spike",
    "flip",
    "speedup",
    "noise",
    "cutoff",
    "average",
    "scale",
    "wander",
    "contextual",
    "upsidedown",
    "mixture",
)
_CLASS_NAMES = ("normal", *_FAMILIES)


class _AnomalyInjector:
    def __init__(self, settings, generator):
        self.min_segment_fraction = settings["min_segment_fraction"]
        self.max_segment_fraction = settings["max_segment_fraction"]
        self.spike_scale = settings["spike_scale"]
        self.anomaly_visibility_boost = settings["anomaly_visibility_boost"]
        self.family_intensity = settings.get("family_intensity", {})
        if not 0 < self.min_segment_fraction <= self.max_segment_fraction <= 1:
            raise ValueError("segment fractions must be in order in (0, 1]")
        if self.spike_scale <= 0 or self.anomaly_visibility_boost <= 0:
            raise ValueError("spike scale and visibility boost must be positive")
        self.epsilon = 1e-6
        self._rng = generator
        self._class_rotation_offset = settings.get("class_rotation_offset", 0)
        self.family_registry = {
            name: getattr(anomaly_families, f"inject_{name}_family")
            for name in _FAMILIES
        }
        if self.family_intensity.keys() - self.family_registry.keys():
            raise ValueError("unknown anomaly family in family_intensity")

    def _family_settings(self, family_name: str) -> dict[str, Any]:
        return self.family_intensity.get(family_name, {})

    def _rand(self, *shape: int, device: torch.device) -> torch.Tensor:
        return torch.rand(shape, generator=self._rng, dtype=torch.float32).to(
            device=device
        )

    def _randn(
        self, *shape: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        random_tensor = torch.randn(shape, generator=self._rng, dtype=dtype)
        return random_tensor.to(device=device)

    def _randint(
        self,
        low: int,
        high: int,
        shape: tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        random_tensor = torch.randint(low, high, shape, generator=self._rng)
        return random_tensor.to(device=device)

    def _randperm(self, size: int, *, device: torch.device) -> torch.Tensor:
        random_tensor = torch.randperm(size, generator=self._rng)
        return random_tensor.to(device=device)

    def _sample_segment_bounds(
        self, window_size: int, device: torch.device
    ) -> tuple[int, int]:
        # Every anomaly family starts from one contiguous subsequence so the
        # resulting mask can still be reasoned about on the original timeline.
        min_segment_length = max(1, int(window_size * self.min_segment_fraction))
        max_segment_length = max(
            min_segment_length, int(window_size * self.max_segment_fraction)
        )
        max_segment_length = min(max_segment_length, window_size)
        segment_length = int(
            self._randint(
                min_segment_length, max_segment_length + 1, (1,), device=device
            ).item()
        )
        max_start_index = max(window_size - segment_length, 0)
        start_index = int(
            self._randint(0, max_start_index + 1, (1,), device=device).item()
        )
        end_index = start_index + segment_length
        return start_index, end_index

    def _sample_affected_channels(
        self, num_channels: int, device: torch.device
    ) -> list[int]:
        min_channels = max(1, num_channels // 10)
        max_channels = max(min_channels, num_channels // 2)
        num_selected_channels = int(
            self._randint(min_channels, max_channels + 1, (1,), device=device).item()
        )
        selected_indices = self._randperm(num_channels, device=device)[
            :num_selected_channels
        ]
        return [int(index.item()) for index in selected_indices.sort().values]

    def _extract_channel_segment(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        return clean_channel_window[start_index:end_index, 0].clone()

    def _segment_scale(self, segment: torch.Tensor) -> torch.Tensor:
        # Several families need a magnitude reference. We use a robust fallback
        # so even low-variance windows can still receive visible corruption.
        centered_segment = segment - segment.mean()
        scale = centered_segment.abs().max()
        fallback_scale = segment.abs().mean()
        return torch.maximum(scale, fallback_scale).clamp_min(0.1)

    def _boost_visibility(
        self, original_segment: torch.Tensor, updated_segment: torch.Tensor
    ) -> torch.Tensor:
        if self.anomaly_visibility_boost == 1.0:
            return updated_segment
        return (
            original_segment
            + (updated_segment - original_segment) * self.anomaly_visibility_boost
        )

    def _attach_change_metadata(
        self,
        family_parameters: dict[str, Any],
        channel_mask: torch.Tensor,
    ) -> dict[str, Any]:
        changed_positions = [
            int(position.item())
            for position in torch.nonzero(channel_mask, as_tuple=False).flatten()
        ]
        family_parameters["changed_positions"] = changed_positions
        family_parameters["changed_position_count"] = len(changed_positions)
        return family_parameters

    def _interpolate_1d(
        self, values: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if values.numel() == 1:
            return values.repeat(positions.shape[0])
        lower_indices = (
            torch.floor(positions).long().clamp(min=0, max=values.shape[0] - 1)
        )
        upper_indices = (
            torch.ceil(positions).long().clamp(min=0, max=values.shape[0] - 1)
        )
        interpolation_weight = positions - lower_indices.to(dtype=values.dtype)
        return (
            values[lower_indices] * (1.0 - interpolation_weight)
            + values[upper_indices] * interpolation_weight
        )

    def _apply_segment_update(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
        updated_segment: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anomalous_channel_window = clean_channel_window.clone()
        anomalous_channel_window[start_index:end_index, 0] = updated_segment
        channel_mask = torch.zeros(
            clean_channel_window.shape[0],
            dtype=torch.long,
            device=clean_channel_window.device,
        )
        channel_mask[start_index:end_index] = local_mask.long()
        return anomalous_channel_window, channel_mask

    def _inject_window(
        self,
        clean_window: torch.Tensor,
        anomaly_family: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        window_size, num_channels = clean_window.shape
        start_index, end_index = self._sample_segment_bounds(
            window_size, clean_window.device
        )
        # Keep the original RNG step when the family is already known.
        self._randint(0, 1, (1,), device=clean_window.device)
        affected_channels = self._sample_affected_channels(
            num_channels, clean_window.device
        )

        augmented_window = clean_window.clone()
        family_parameters_by_channel: dict[str, dict[str, Any]] = {}
        anomaly_mask = torch.zeros(
            window_size, dtype=torch.long, device=clean_window.device
        )

        for channel_index in affected_channels:
            channel_window = augmented_window[:, channel_index : channel_index + 1]
            anomalous_channel_window, channel_mask, family_parameters = (
                self.family_registry[anomaly_family](
                    self,
                    channel_window,
                    start_index,
                    end_index,
                )
            )
            augmented_window[:, channel_index] = anomalous_channel_window.squeeze(-1)
            anomaly_mask = torch.maximum(anomaly_mask, channel_mask)
            family_parameters_by_channel[str(channel_index)] = family_parameters

        augmentation_metadata = {
            "is_synthetic_anomaly": True,
            "anomaly_family": anomaly_family,
            "anomaly_family_index": 0,
            "start_index": start_index,
            "end_index": end_index,
            "segment_length": end_index - start_index,
            "visibility_boost_factor": self.anomaly_visibility_boost,
            "affected_channels": affected_channels,
            "family_parameters_by_channel": family_parameters_by_channel,
        }
        return augmented_window, anomaly_mask, augmentation_metadata

    def _build_clean_metadata(self) -> dict[str, Any]:
        return {
            "is_synthetic_anomaly": False,
            "anomaly_family": "clean",
            "anomaly_family_index": None,
            "start_index": None,
            "end_index": None,
            "segment_length": None,
            "visibility_boost_factor": None,
            "affected_channels": [],
            "family_parameters_by_channel": {},
        }

    def _balanced_class_quota(self, batch_size: int) -> dict[int, int]:
        class_indices = range(len(_CLASS_NAMES))
        num_classes = len(class_indices)
        base_count_per_class = batch_size // num_classes
        remainder_count = batch_size % num_classes

        quota = {class_index: base_count_per_class for class_index in class_indices}
        for remainder_index in range(remainder_count):
            rotated_index = (
                self._class_rotation_offset + remainder_index
            ) % num_classes
            class_index = class_indices[rotated_index]
            quota[class_index] += 1

        self._class_rotation_offset = (
            self._class_rotation_offset + remainder_count
        ) % num_classes
        return quota

    def _sample_class_labels(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        class_quota = self._balanced_class_quota(batch_size)
        ordered_labels: list[int] = []
        for class_index, count in class_quota.items():
            ordered_labels.extend([class_index] * count)
        ordered_label_tensor = torch.tensor(
            ordered_labels, dtype=torch.long, device=device
        )
        shuffled_positions = self._randperm(batch_size, device=device)
        return ordered_label_tensor[shuffled_positions]

    def apply(self, x):
        batch_size, window_size, _ = x.shape
        output = x.clone()
        masks = torch.zeros(batch_size, window_size, dtype=torch.long, device=x.device)
        labels = self._sample_class_labels(batch_size=batch_size, device=x.device)
        metadata = []
        for index, label in enumerate(labels.tolist()):
            if label == 0:
                metadata.append(self._build_clean_metadata())
                continue
            output[index], masks[index], record = self._inject_window(
                x[index], _CLASS_NAMES[label]
            )
            metadata.append(record)
        return output, {
            "classification_labels": labels,
            "classification_class_names": _CLASS_NAMES,
            "synthetic_anomaly_mask": masks.bool(),
            "augmentation_metadata": metadata,
            "class_rotation_offset": self._class_rotation_offset,
        }


def augment(
    x: torch.Tensor, settings: dict, *, generator: torch.Generator
) -> tuple[torch.Tensor, dict]:
    """Return new observations and synthetic supervision for one batch."""
    if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError(
            "augmentation requires a nonempty floating-point [batch, time, feature] tensor"
        )
    if generator.device.type != "cpu":
        raise ValueError("augmentation requires a CPU generator")
    return _AnomalyInjector(settings, generator).apply(x)

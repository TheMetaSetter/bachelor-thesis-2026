from __future__ import annotations

"""Synthetic anomaly injection for the offline multitask path.

This file is the single owning surface for thesis-facing synthetic anomaly
injection. The active default taxonomy follows the 11 anomaly types from the
RedLamp reference, while CARLA remains a mechanism reference for keeping the
implementation subsequence-oriented and easy to inspect on fixed windows.
"""

from typing import Any, Callable

import torch

from src.core.console import (
    console_print,
    summarize_label_distribution,
    summarize_tensor,
)

REDLAMP_ANOMALY_FAMILIES: tuple[str, ...] = (
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


class SyntheticAnomalyInjector:
    def __init__(
        self,
        anomaly_probability: float = 0.5,
        min_segment_fraction: float = 0.1,
        max_segment_fraction: float = 0.2,
        spike_scale: float = 3.0,
        anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
        deterministic_seed: int | None = None,
    ) -> None:
        # These checks keep augmentation behavior explicit. Research code becomes
        # very hard to trust when synthetic data is allowed to silently drift.
        if not 0.0 <= anomaly_probability <= 1.0:
            raise ValueError("anomaly_probability must be between 0 and 1")
        if not 0.0 < min_segment_fraction <= 1.0:
            raise ValueError("min_segment_fraction must be between 0 and 1")
        if not 0.0 < max_segment_fraction <= 1.0:
            raise ValueError("max_segment_fraction must be between 0 and 1")
        if min_segment_fraction > max_segment_fraction:
            raise ValueError(
                "min_segment_fraction must not exceed max_segment_fraction"
            )
        if spike_scale <= 0.0:
            raise ValueError("spike_scale must be positive")
        if not anomaly_families:
            raise ValueError("anomaly_families must not be empty")

        self.anomaly_probability = anomaly_probability
        self.min_segment_fraction = min_segment_fraction
        self.max_segment_fraction = max_segment_fraction
        self.spike_scale = spike_scale
        self.epsilon = 1e-6
        self.deterministic_seed = deterministic_seed
        self._rng: torch.Generator | None = None

        # A new reader should notice that the taxonomy is visible in one place.
        # The rest of the file only dispatches through this registry.
        self.family_registry: dict[
            str,
            Callable[
                [torch.Tensor, int, int],
                tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
            ],
        ] = {
            "spike": self._inject_spike_family,
            "flip": self._inject_flip_family,
            "speedup": self._inject_speedup_family,
            "noise": self._inject_noise_family,
            "cutoff": self._inject_cutoff_family,
            "average": self._inject_average_family,
            "scale": self._inject_scale_family,
            "wander": self._inject_wander_family,
            "contextual": self._inject_contextual_family,
            "upsidedown": self._inject_upsidedown_family,
            "mixture": self._inject_mixture_family,
        }
        self.anomaly_families = tuple(anomaly_families)
        unknown_families = sorted(
            set(self.anomaly_families) - set(self.family_registry)
        )
        if unknown_families:
            raise ValueError(f"Unsupported anomaly_families: {unknown_families}")

        self.reset_rng()

    def __getstate__(self) -> dict[str, Any]:
        # The deterministic RNG is runtime-only state. Rebuild it after copy or
        # deserialization instead of trying to pickle the torch generator.
        serialized_state = dict(self.__dict__)
        serialized_state["_rng"] = None
        return serialized_state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.reset_rng()

    def reset_rng(self) -> None:
        # Validation-time synthetic augmentation needs repeatable corruption so
        # epoch-to-epoch classification curves remain comparable.
        if self.deterministic_seed is None:
            self._rng = None
            return

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.deterministic_seed))
        self._rng = generator

    def _rand(
        self, *shape: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        random_tensor = torch.rand(shape, generator=self._rng, dtype=dtype)
        return random_tensor.to(device=device)

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

    def _clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = [
                    dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                cloned_batch[key] = value
        return cloned_batch

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

    def _inject_spike_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        segment_length = segment.shape[0]
        num_spikes = int(
            self._randint(
                1, min(segment_length, 3) + 1, (1,), device=segment.device
            ).item()
        )
        spike_positions = (
            self._randperm(segment_length, device=segment.device)[:num_spikes]
            .sort()
            .values
        )
        spike_strength = self._segment_scale(segment) * self.spike_scale
        spike_noise = (
            self._randn(num_spikes, device=segment.device, dtype=segment.dtype)
            * spike_strength
        )
        updated_segment = segment.clone()
        updated_segment[spike_positions] = (
            updated_segment[spike_positions] + spike_noise
        )
        local_mask = torch.zeros(
            segment_length, dtype=torch.long, device=segment.device
        )
        local_mask[spike_positions] = 1
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window=clean_channel_window,
            start_index=start_index,
            end_index=end_index,
            updated_segment=updated_segment,
            local_mask=local_mask,
        )
        family_parameters = {
            "spike_positions": [int(position.item()) for position in spike_positions],
            "spike_strength": float(spike_strength.detach().cpu()),
        }
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_flip_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        updated_segment = torch.flip(segment, dims=(0,))
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"operation": "reverse_subsequence"}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_speedup_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        speed_factor = float(
            torch.tensor([1.5, 2.0, 3.0], device=segment.device)[
                self._randint(0, 3, (1,), device=segment.device)
            ].item()
        )
        source_positions = (
            torch.linspace(
                0.0,
                segment.shape[0] - 1,
                segment.shape[0],
                device=segment.device,
                dtype=segment.dtype,
            )
            * speed_factor
        )
        source_positions = source_positions.clamp(max=segment.shape[0] - 1)
        updated_segment = self._interpolate_1d(segment, source_positions)
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"speed_factor": speed_factor}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_noise_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        noise_std = float((self._segment_scale(segment) * 0.35).detach().cpu())
        updated_segment = (
            segment
            + self._randn(*segment.shape, device=segment.device, dtype=segment.dtype)
            * noise_std
        )
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"noise_std": noise_std}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_cutoff_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        cutoff_mode = (
            "zero"
            if bool(self._rand(1, device=segment.device).item() < 0.5)
            else "hold"
        )
        if cutoff_mode == "zero":
            updated_segment = torch.zeros_like(segment)
        else:
            updated_segment = torch.full_like(segment, float(segment[0].detach().cpu()))
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"cutoff_mode": cutoff_mode}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_average_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        segment_mean = float(segment.mean().detach().cpu())
        updated_segment = torch.full_like(segment, segment_mean)
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"segment_mean": segment_mean}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_scale_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        center = segment.mean()
        scale_factor_candidates = torch.tensor(
            [0.25, 0.5, 1.5, 2.0], device=segment.device, dtype=segment.dtype
        )
        scale_factor = float(
            scale_factor_candidates[
                self._randint(
                    0, scale_factor_candidates.shape[0], (1,), device=segment.device
                )
            ].item()
        )
        updated_segment = center + (segment - center) * scale_factor
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"scale_factor": scale_factor}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_wander_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        drift_scale = self._segment_scale(segment) * 0.15
        drift_noise = (
            self._randn(segment.shape[0], device=segment.device, dtype=segment.dtype)
            * drift_scale
        )
        drift_curve = torch.cumsum(drift_noise, dim=0)
        drift_curve = drift_curve - drift_curve[0]
        updated_segment = segment + drift_curve
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"drift_scale": float(drift_scale.detach().cpu())}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_contextual_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        full_channel = clean_channel_window[:, 0]
        outside_context_mask = torch.ones(
            full_channel.shape[0], dtype=torch.bool, device=full_channel.device
        )
        outside_context_mask[start_index:end_index] = False
        if outside_context_mask.any():
            outside_context_mean = full_channel[outside_context_mask].mean()
        else:
            outside_context_mean = full_channel.mean()
        contextual_offset = outside_context_mean - segment.mean()
        contextual_offset = contextual_offset + self._segment_scale(segment) * 0.5
        updated_segment = segment + contextual_offset
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {
            "contextual_offset": float(contextual_offset.detach().cpu())
        }
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_upsidedown_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        segment = self._extract_channel_segment(
            clean_channel_window, start_index, end_index
        )
        inversion_center = segment.mean()
        updated_segment = 2.0 * inversion_center - segment
        local_mask = torch.ones(
            segment.shape[0], dtype=torch.long, device=segment.device
        )
        anomalous_channel_window, channel_mask = self._apply_segment_update(
            clean_channel_window,
            start_index,
            end_index,
            updated_segment,
            local_mask,
        )
        family_parameters = {"inversion_center": float(inversion_center.detach().cpu())}
        return anomalous_channel_window, channel_mask, family_parameters

    def _inject_mixture_family(
        self,
        clean_channel_window: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        primitive_component_names = [
            family_name
            for family_name in self.anomaly_families
            if family_name != "mixture"
        ]
        if not primitive_component_names:
            # A mixture-only debug config should still produce a real anomaly.
            # Fall back to the full primitive RedLamp family set instead of
            # returning empty metadata with an all-zero mask.
            primitive_component_names = [
                family_name
                for family_name in REDLAMP_ANOMALY_FAMILIES
                if family_name != "mixture"
            ]

        max_component_count = min(3, len(primitive_component_names))
        component_count = int(
            self._randint(
                2, max_component_count + 1, (1,), device=clean_channel_window.device
            ).item()
        )
        component_indices = self._randperm(
            len(primitive_component_names),
            device=clean_channel_window.device,
        )[:component_count]
        selected_components = [
            primitive_component_names[int(index.item())] for index in component_indices
        ]

        working_channel_window = clean_channel_window.clone()
        combined_mask = torch.zeros(
            clean_channel_window.shape[0],
            dtype=torch.long,
            device=clean_channel_window.device,
        )
        mixture_components: list[dict[str, Any]] = []

        for component_name in selected_components:
            component_window, component_mask, component_parameters = (
                self.family_registry[component_name](
                    working_channel_window,
                    start_index,
                    end_index,
                )
            )
            working_channel_window = component_window
            combined_mask = torch.maximum(combined_mask, component_mask)
            mixture_components.append(
                {
                    "component_family": component_name,
                    "component_parameters": component_parameters,
                }
            )

        family_parameters = {"mixture_components": mixture_components}
        return working_channel_window, combined_mask, family_parameters

    def _inject_single_window(
        self,
        clean_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        window_size, num_channels = clean_window.shape
        start_index, end_index = self._sample_segment_bounds(
            window_size, clean_window.device
        )
        anomaly_family_index = int(
            self._randint(
                0, len(self.anomaly_families), (1,), device=clean_window.device
            ).item()
        )
        anomaly_family = self.anomaly_families[anomaly_family_index]
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
            "anomaly_family_index": anomaly_family_index,
            "start_index": start_index,
            "end_index": end_index,
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
            "affected_channels": [],
            "family_parameters_by_channel": {},
        }

    def augment_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        # The output batch keeps the original keys and only adds multitask
        # supervision fields, which is why the rest of the codepath stays small.
        if "x" not in batch:
            raise ValueError("batch must contain 'x'")
        if batch["x"].ndim != 3:
            raise ValueError("batch['x'] must have shape [B, L, D]")

        augmented_batch = self._clone_batch(batch)
        clean_windows = batch["x"]
        batch_size, window_size, _ = clean_windows.shape

        anomaly_masks = torch.zeros(
            batch_size, window_size, dtype=torch.long, device=clean_windows.device
        )
        classification_labels = torch.zeros(
            batch_size, dtype=torch.long, device=clean_windows.device
        )
        augmentation_metadata: list[dict[str, Any]] = []

        for batch_index in range(batch_size):
            should_inject = bool(
                self._rand(1, device=clean_windows.device).item()
                < self.anomaly_probability
            )
            if not should_inject:
                augmentation_metadata.append(self._build_clean_metadata())
                continue

            augmented_window, anomaly_mask, window_metadata = (
                self._inject_single_window(clean_windows[batch_index])
            )
            augmented_batch["x"][batch_index] = augmented_window
            anomaly_masks[batch_index] = anomaly_mask
            classification_labels[batch_index] = 1
            augmentation_metadata.append(window_metadata)

        original_point_labels = batch.get("point_labels")
        if original_point_labels is None:
            augmented_batch["point_labels"] = anomaly_masks
        else:
            augmented_batch["point_labels"] = torch.maximum(
                original_point_labels.clone(), anomaly_masks
            )

        augmented_batch["classification_labels"] = classification_labels
        augmented_batch["synthetic_anomaly_mask"] = anomaly_masks
        augmented_batch["augmentation_metadata"] = augmentation_metadata
        anomalous_windows = int(classification_labels.sum().detach().cpu())
        anomaly_families_present = sorted(
            {
                metadata["anomaly_family"]
                for metadata in augmentation_metadata
                if metadata["is_synthetic_anomaly"]
            }
        )
        console_print(
            "DATA",
            "Augmented multitask batch",
            input_x=summarize_tensor(batch["x"]),
            output_x=summarize_tensor(augmented_batch["x"]),
            anomaly_mask=summarize_tensor(anomaly_masks),
            anomalous_windows=anomalous_windows,
            classification_label_distribution=summarize_label_distribution(
                classification_labels
            ),
            anomaly_families_present=anomaly_families_present,
        )
        return augmented_batch

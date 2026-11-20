"""Apply synthetic anomaly changes to one channel."""

from typing import Any

import torch


def inject_spike_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    segment_length = segment.shape[0]
    settings = injector._family_settings("spike")
    max_spikes = int(settings.get("max_spikes", 3))
    num_spikes = int(
        injector._randint(
            1, min(segment_length, max_spikes) + 1, (1,), device=segment.device
        ).item()
    )
    spike_positions = (
        injector._randperm(segment_length, device=segment.device)[:num_spikes]
        .sort()
        .values
    )
    spike_multiplier = float(settings.get("spike_scale_multiplier", 1.0))
    spike_strength = injector._segment_scale(segment) * injector.spike_scale
    spike_strength = spike_strength * spike_multiplier
    spike_noise = (
        injector._randn(num_spikes, device=segment.device, dtype=segment.dtype)
        * spike_strength
    )
    updated_segment = segment.clone()
    updated_segment[spike_positions] = (
        updated_segment[spike_positions] + spike_noise
    )
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.zeros(
        segment_length, dtype=torch.long, device=segment.device
    )
    local_mask[spike_positions] = 1
    return _finish_spike(
        injector, clean_channel_window, start_index, end_index, updated_segment,
        local_mask, spike_positions, spike_strength,
    )


def _finish_spike(injector, clean_channel_window, start_index, end_index, updated_segment,
                  local_mask, spike_positions, spike_strength):
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window=clean_channel_window,
        start_index=start_index,
        end_index=end_index,
        updated_segment=updated_segment,
        local_mask=local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {
            "spike_positions": [
                int(position.item()) for position in spike_positions
            ],
            "spike_strength": float(spike_strength.detach().cpu()),
        },
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_flip_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    updated_segment = torch.flip(segment, dims=(0,))
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"operation": "reverse_subsequence"},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_speedup_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    factors = injector._family_settings("speedup").get("factors", [1.5, 2.0, 3.0])
    speed_factor = float(
        torch.tensor(factors, device=segment.device)[
            injector._randint(0, len(factors), (1,), device=segment.device)
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
    updated_segment = injector._interpolate_1d(segment, source_positions)
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"speed_factor": speed_factor},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_noise_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    noise_std = float((injector._segment_scale(segment) * 0.35).detach().cpu())
    updated_segment = (
        segment
        + injector._randn(*segment.shape, device=segment.device, dtype=segment.dtype)
        * noise_std
    )
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"noise_std": noise_std},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_cutoff_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    cutoff_mode = (
        "zero"
        if bool(injector._rand(1, device=segment.device).item() < 0.5)
        else "hold"
    )
    if cutoff_mode == "zero":
        updated_segment = torch.zeros_like(segment)
    else:
        updated_segment = torch.full_like(segment, float(segment[0].detach().cpu()))
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"cutoff_mode": cutoff_mode},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_average_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    segment_mean = float(segment.mean().detach().cpu())
    updated_segment = torch.full_like(segment, segment_mean)
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"segment_mean": segment_mean},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_scale_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    center = segment.mean()
    factors = injector._family_settings("scale").get("factors", [0.25, 0.5, 1.5, 2.0])
    scale_factor_candidates = torch.tensor(
        factors, device=segment.device, dtype=segment.dtype
    )
    scale_factor = float(
        scale_factor_candidates[
            injector._randint(
                0, scale_factor_candidates.shape[0], (1,), device=segment.device
            )
        ].item()
    )
    updated_segment = center + (segment - center) * scale_factor
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"scale_factor": scale_factor},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_wander_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    drift_scale = injector._segment_scale(segment) * 0.15
    drift_noise = (
        injector._randn(segment.shape[0], device=segment.device, dtype=segment.dtype)
        * drift_scale
    )
    drift_curve = torch.cumsum(drift_noise, dim=0)
    drift_curve = drift_curve - drift_curve[0]
    updated_segment = segment + drift_curve
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"drift_scale": float(drift_scale.detach().cpu())},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_contextual_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
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
    contextual_offset = contextual_offset + injector._segment_scale(segment) * 0.5
    updated_segment = segment + contextual_offset
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"contextual_offset": float(contextual_offset.detach().cpu())},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_upsidedown_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    segment = injector._extract_channel_segment(
        clean_channel_window, start_index, end_index
    )
    inversion_center = segment.mean()
    updated_segment = 2.0 * inversion_center - segment
    updated_segment = injector._boost_visibility(segment, updated_segment)
    local_mask = torch.ones(
        segment.shape[0], dtype=torch.long, device=segment.device
    )
    anomalous_channel_window, channel_mask = injector._apply_segment_update(
        clean_channel_window,
        start_index,
        end_index,
        updated_segment,
        local_mask,
    )
    family_parameters = injector._attach_change_metadata(
        {"inversion_center": float(inversion_center.detach().cpu())},
        channel_mask,
    )
    return anomalous_channel_window, channel_mask, family_parameters


def inject_mixture_family(
    injector,
    clean_channel_window: torch.Tensor,
    start_index: int,
    end_index: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    primitive_component_names = [name for name in injector.family_registry if name != "mixture"]
    max_component_count = min(3, len(primitive_component_names))
    component_count = int(
        injector._randint(
            2, max_component_count + 1, (1,), device=clean_channel_window.device
        ).item()
    )
    component_indices = injector._randperm(
        len(primitive_component_names),
        device=clean_channel_window.device,
    )[:component_count]
    selected_components = [primitive_component_names[int(index.item())] for index in component_indices]

    working_channel_window = clean_channel_window.clone()
    combined_mask = torch.zeros(
        clean_channel_window.shape[0],
        dtype=torch.long,
        device=clean_channel_window.device,
    )
    mixture_components: list[dict[str, Any]] = []

    for component_name in selected_components:
        component_window, component_mask, component_parameters = (
            injector.family_registry[component_name](
                injector, working_channel_window,
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

    family_parameters = injector._attach_change_metadata(
        {"mixture_components": mixture_components},
        combined_mask,
    )
    return working_channel_window, combined_mask, family_parameters

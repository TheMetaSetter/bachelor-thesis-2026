from __future__ import annotations

"""Shared absolute-range contract for online benchmark streams."""

from typing import Any


def select_online_stream_sequence(
    sequence: dict[str, Any],
    *,
    absolute_start_index: int | None,
    absolute_end_index: int | None,
) -> dict[str, Any]:
    """Select one entity-global half-open interval before windowization."""
    if absolute_start_index is None and absolute_end_index is None:
        return sequence
    if absolute_start_index is None or absolute_end_index is None:
        raise ValueError(
            "absolute_start_index and absolute_end_index must be set together"
        )

    source_length = int(sequence["x"].shape[0])
    if not 0 <= absolute_start_index < absolute_end_index <= source_length:
        raise ValueError(
            "Online stream range must satisfy "
            f"0 <= start < end <= {source_length}, got "
            f"[{absolute_start_index}, {absolute_end_index})"
        )

    selected_sequence = dict(sequence)
    for field_name in ("x", "point_labels", "mask", "timestamps"):
        selected_sequence[field_name] = _slice_sequence_field(
            sequence.get(field_name), absolute_start_index, absolute_end_index
        )

    selected_sequence["meta"] = {
        **sequence.get("meta", {}),
        "sequence_length": absolute_end_index - absolute_start_index,
        "source_sequence_length": source_length,
        "absolute_start_index": absolute_start_index,
        "absolute_end_index": absolute_end_index,
    }
    return selected_sequence


def _slice_sequence_field(
    value: Any,
    start_index: int,
    end_index: int,
) -> Any:
    if value is None:
        return None
    selected_value = value[start_index:end_index]
    clone = getattr(selected_value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(selected_value, "copy", None)
    if callable(copy):
        return copy()
    return selected_value

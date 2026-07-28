from __future__ import annotations

from typing import Any


def resolve_balance_classes_setting(
    *,
    canonical_value: bool | None,
    legacy_value: bool,
    default_value: bool,
) -> bool:
    """Resolve the canonical class-balancing flag and its legacy alias."""
    if canonical_value is not None:
        return bool(canonical_value)
    return bool(legacy_value or default_value)


def normalize_variance_correction_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in {"unbiased", "sample", "sample_unbiased"}:
            return 1
        if normalized_value in {"population", "biased", "none"}:
            return 0
    raise ValueError(
        "variance_correction must be 0, 1, or one of: unbiased, sample, population"
    )

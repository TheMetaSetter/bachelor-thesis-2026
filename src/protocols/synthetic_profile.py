from __future__ import annotations

from typing import Any


KNOWN_SYNTHETIC_FAMILIES = (
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


def validate_synthetic_profile_config(config: dict[str, Any]) -> None:
    family_intensity = config.get("family_intensity", {})
    unknown_names = sorted(set(family_intensity) - set(KNOWN_SYNTHETIC_FAMILIES))
    if unknown_names:
        raise ValueError(f"Unknown synthetic anomaly family names: {unknown_names}")


def injector_kwargs_from_synthetic_profile(config: dict[str, Any]) -> dict[str, Any]:
    validate_synthetic_profile_config(config)
    return {
        "window_size": int(config["window_size"]),
        "min_segment_fraction": float(config["min_segment_fraction"]),
        "max_segment_fraction": float(config["max_segment_fraction"]),
        "spike_scale": float(config["spike_scale"]),
        "anomaly_visibility_boost": float(config["anomaly_visibility_boost"]),
        "family_intensity": dict(config.get("family_intensity", {})),
    }

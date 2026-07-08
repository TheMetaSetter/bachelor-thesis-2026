from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector
from src.protocols.synthetic_profile import (
    injector_kwargs_from_synthetic_profile,
    validate_synthetic_profile_config,
)


def _load_profile(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _batch() -> dict:
    values = torch.linspace(-1.0, 1.0, steps=60, dtype=torch.float32).reshape(3, 20, 1)
    return {
        "x": values.repeat(1, 1, 3),
        "point_labels": torch.zeros(3, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(3)],
    }


def test_profile_validation_keeps_redlamp_family_order_unchanged() -> None:
    profile = _load_profile("configs/protocol/synthetic_redlamp12_visible_window20.yaml")

    validate_synthetic_profile_config(profile)

    assert REDLAMP_ANOMALY_FAMILIES == (
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


def test_visible_profile_builds_injector_kwargs_without_changing_mixture() -> None:
    profile = _load_profile("configs/protocol/synthetic_redlamp12_visible_window20.yaml")

    kwargs = injector_kwargs_from_synthetic_profile(profile)

    assert kwargs["window_size"] == 20
    assert kwargs["min_segment_fraction"] == 0.2
    assert kwargs["max_segment_fraction"] == 0.3
    assert kwargs["family_intensity"]["mixture"] == {"keep_legacy_behavior": True}


def test_visible_profile_is_deterministic_under_fixed_seed() -> None:
    profile = _load_profile("configs/protocol/synthetic_redlamp12_visible_window20.yaml")
    kwargs = injector_kwargs_from_synthetic_profile(profile)
    kwargs.pop("window_size")

    first = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        deterministic_seed=17,
        train_balance_classes=False,
        anomaly_families=("spike",),
        **kwargs,
    )
    second = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        deterministic_seed=17,
        train_balance_classes=False,
        anomaly_families=("spike",),
        **kwargs,
    )

    first_batch = first.augment_batch(_batch())
    second_batch = second.augment_batch(_batch())

    assert torch.equal(first_batch["x"], second_batch["x"])
    assert first_batch["augmentation_metadata"] == second_batch["augmentation_metadata"]


def test_profile_validation_rejects_unknown_family_names() -> None:
    profile = _load_profile("configs/protocol/synthetic_redlamp12_visible_window20.yaml")
    profile["family_intensity"]["not_a_family"] = {"scale": 2.0}

    try:
        validate_synthetic_profile_config(profile)
    except ValueError as error:
        assert "not_a_family" in str(error)
    else:
        raise AssertionError("unknown family name should be rejected")

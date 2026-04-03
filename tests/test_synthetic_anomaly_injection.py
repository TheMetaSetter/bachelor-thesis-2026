from __future__ import annotations

import pytest
import torch

from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector


def _build_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 20, 3),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def test_synthetic_anomaly_injector_defaults_to_redlamp_taxonomy() -> None:
    injector = SyntheticAnomalyInjector()

    assert injector.anomaly_families == REDLAMP_ANOMALY_FAMILIES
    assert tuple(injector.family_registry.keys()) == REDLAMP_ANOMALY_FAMILIES


def test_synthetic_anomaly_injection_preserves_shapes_and_adds_labels() -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.25,
        spike_scale=4.0,
    )
    batch = _build_batch()

    augmented_batch = injector.augment_batch(batch)

    assert augmented_batch["x"].shape == batch["x"].shape
    assert augmented_batch["point_labels"].shape == batch["point_labels"].shape
    assert augmented_batch["classification_labels"].shape == (2,)
    assert augmented_batch["synthetic_anomaly_mask"].shape == (2, 20)
    assert augmented_batch["classification_labels"].sum().item() == 2
    assert augmented_batch["synthetic_anomaly_mask"].sum().item() > 0
    assert augmented_batch["meta"] == batch["meta"]
    assert augmented_batch["augmentation_metadata"][0]["is_synthetic_anomaly"] is True
    assert augmented_batch["augmentation_metadata"][0]["anomaly_family"] != "clean"
    assert augmented_batch["augmentation_metadata"][0]["anomaly_family_index"] is not None
    assert augmented_batch["augmentation_metadata"][0]["affected_channels"]
    assert isinstance(augmented_batch["augmentation_metadata"][0]["family_parameters_by_channel"], dict)


@pytest.mark.parametrize("anomaly_family", REDLAMP_ANOMALY_FAMILIES)
def test_each_redlamp_family_is_reachable_and_records_metadata(anomaly_family: str) -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.2,
        anomaly_families=(anomaly_family,),
    )
    batch = _build_batch()

    augmented_batch = injector.augment_batch(batch)
    metadata = augmented_batch["augmentation_metadata"][0]

    assert metadata["anomaly_family"] == anomaly_family
    assert metadata["anomaly_family_index"] == 0
    assert metadata["start_index"] is not None
    assert metadata["end_index"] is not None
    assert metadata["affected_channels"]
    assert augmented_batch["synthetic_anomaly_mask"][0].sum().item() > 0
    assert not torch.equal(augmented_batch["x"][0], batch["x"][0])

    if anomaly_family == "mixture":
        first_channel = str(metadata["affected_channels"][0])
        assert "mixture_components" in metadata["family_parameters_by_channel"][first_channel]

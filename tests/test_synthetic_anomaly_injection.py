from __future__ import annotations

import torch

from src.data.augment import SyntheticAnomalyInjector


def test_synthetic_anomaly_injection_preserves_shapes_and_adds_labels() -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.25,
        spike_scale=4.0,
    )
    batch = {
        "x": torch.randn(2, 20, 3),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

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
    assert augmented_batch["augmentation_metadata"][0]["affected_channels"]
    assert isinstance(augmented_batch["augmentation_metadata"][0]["family_parameters_by_channel"], dict)

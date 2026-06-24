from __future__ import annotations

import pytest
import torch

from src.data.augment import (
    BINARY_SYNTHETIC_CLASS_NAMES,
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    SyntheticAnomalyInjector,
)


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
    assert REDLAMP_MULTICLASS_CLASS_NAMES == ("normal", *REDLAMP_ANOMALY_FAMILIES)
    assert injector.classification_label_mode == "redlamp_multiclass"
    assert injector.train_balance_classes is True


def test_synthetic_anomaly_injection_preserves_shapes_and_adds_labels() -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.25,
        spike_scale=4.0,
        classification_label_mode="binary",
        train_balance_classes=False,
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
    assert (
        augmented_batch["augmentation_metadata"][0]["anomaly_family_index"] is not None
    )
    assert augmented_batch["augmentation_metadata"][0]["affected_channels"]
    assert isinstance(
        augmented_batch["augmentation_metadata"][0]["family_parameters_by_channel"],
        dict,
    )
    assert augmented_batch["classification_class_names"] == BINARY_SYNTHETIC_CLASS_NAMES


@pytest.mark.parametrize("anomaly_family", REDLAMP_ANOMALY_FAMILIES)
def test_redlamp_multiclass_injection_maps_family_to_shared_class_index(
    anomaly_family: str,
) -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.2,
        anomaly_families=(anomaly_family,),
        classification_label_mode="redlamp_multiclass",
        train_balance_classes=False,
    )
    batch = _build_batch()

    augmented_batch = injector.augment_batch(batch)
    expected_class_index = REDLAMP_MULTICLASS_CLASS_NAMES.index(anomaly_family)

    assert augmented_batch["classification_class_names"] == (
        REDLAMP_MULTICLASS_CLASS_NAMES
    )
    assert torch.equal(
        augmented_batch["classification_labels"],
        torch.full((2,), expected_class_index, dtype=torch.long),
    )
    assert augmented_batch["augmentation_metadata"][0]["anomaly_family"] == (
        anomaly_family
    )


def test_balanced_binary_injection_uses_fixed_positive_quota() -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=0.5,
        min_segment_fraction=0.3,
        max_segment_fraction=0.6,
        anomaly_families=("spike", "noise"),
        train_balance_classes=True,
        deterministic_seed=7,
        classification_label_mode="binary",
    )
    batch = {
        "x": torch.randn(8, 10, 3),
        "point_labels": torch.zeros(8, 10, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(8)],
    }

    augmented_batch = injector.augment_batch(batch)

    assert augmented_batch["classification_labels"].sum().item() == 4
    assert (
        torch.count_nonzero(augmented_batch["classification_labels"] == 0).item() == 4
    )
    assert augmented_batch["synthetic_anomaly_mask"].sum().item() > 0


@pytest.mark.parametrize("anomaly_family", REDLAMP_ANOMALY_FAMILIES)
def test_each_redlamp_family_is_reachable_and_records_metadata(
    anomaly_family: str,
) -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.3,
        anomaly_families=(anomaly_family,),
        train_balance_classes=False,
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
        assert (
            "mixture_components"
            in metadata["family_parameters_by_channel"][first_channel]
        )


@pytest.mark.parametrize("anomaly_family", REDLAMP_ANOMALY_FAMILIES)
def test_redlamp_family_uses_four_to_six_timestep_segments_and_records_positions(
    anomaly_family: str,
) -> None:
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.3,
        anomaly_families=(anomaly_family,),
        classification_label_mode="redlamp_multiclass",
        train_balance_classes=False,
    )
    batch = _build_batch()

    augmented_batch = injector.augment_batch(batch)
    metadata = augmented_batch["augmentation_metadata"][0]
    segment_length = int(metadata["end_index"] - metadata["start_index"])
    first_channel = str(metadata["affected_channels"][0])
    family_parameters = metadata["family_parameters_by_channel"][first_channel]
    changed_positions = family_parameters["changed_positions"]

    assert 4 <= segment_length <= 6
    assert metadata["segment_length"] == segment_length
    assert metadata["visibility_boost_factor"] == pytest.approx(
        injector.anomaly_visibility_boost
    )
    assert isinstance(changed_positions, list)
    assert changed_positions
    assert all(
        metadata["start_index"] <= position < metadata["end_index"]
        for position in changed_positions
    )
    assert family_parameters["changed_position_count"] == len(changed_positions)


def test_balanced_multiclass_remainder_uses_round_robin_class_allocation() -> None:
    injector = SyntheticAnomalyInjector(
        train_balance_classes=True,
        classification_label_mode="redlamp_multiclass",
        deterministic_seed=23,
    )
    batch = {
        "x": torch.randn(14, 20, 3),
        "point_labels": torch.zeros(14, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(14)],
    }

    augmented_batch = injector.augment_batch(batch)
    class_counts = torch.bincount(
        augmented_batch["classification_labels"],
        minlength=len(REDLAMP_MULTICLASS_CLASS_NAMES),
    )

    assert class_counts.sum().item() == 14
    assert torch.count_nonzero(class_counts == 2).item() == 2
    assert torch.count_nonzero(class_counts == 1).item() == 10


def test_balanced_multiclass_small_batch_rotates_class_coverage() -> None:
    injector = SyntheticAnomalyInjector(
        train_balance_classes=True,
        classification_label_mode="redlamp_multiclass",
        deterministic_seed=31,
    )
    batch = {
        "x": torch.randn(5, 20, 3),
        "point_labels": torch.zeros(5, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(5)],
    }

    first_batch = injector.augment_batch(batch)
    second_batch = injector.augment_batch(batch)
    first_labels = set(first_batch["classification_labels"].tolist())
    second_labels = set(second_batch["classification_labels"].tolist())

    assert len(first_labels) == 5
    assert len(second_labels) == 5
    assert first_labels != second_labels


def test_visibility_boost_increases_deviation_for_same_synthetic_sample() -> None:
    batch = _build_batch()
    base_injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.3,
        anomaly_visibility_boost=1.0,
        anomaly_families=("flip",),
        deterministic_seed=19,
        classification_label_mode="redlamp_multiclass",
        train_balance_classes=False,
    )
    boosted_injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.3,
        anomaly_visibility_boost=1.5,
        anomaly_families=("flip",),
        deterministic_seed=19,
        classification_label_mode="redlamp_multiclass",
        train_balance_classes=False,
    )

    base_batch = base_injector.augment_batch(batch)
    boosted_batch = boosted_injector.augment_batch(batch)
    base_delta = torch.mean(torch.abs(base_batch["x"] - batch["x"]))
    boosted_delta = torch.mean(torch.abs(boosted_batch["x"] - batch["x"]))

    assert boosted_delta > base_delta

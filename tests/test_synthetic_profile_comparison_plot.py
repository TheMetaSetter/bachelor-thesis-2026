from __future__ import annotations

import torch

from scripts.compare_synthetic_profiles import (
    _ensure_demo_batch_has_enough_channels,
    _build_sample_plot_annotation,
    _build_family_gallery_batches,
    _injected_point_indices,
    _resolve_visualization_seed,
    _select_most_visible_channels,
    _select_most_visible_sample_channel,
    _select_random_window_indices,
)
from src.data.augment import REDLAMP_ANOMALY_FAMILIES


def test_injected_point_indices_come_from_synthetic_mask() -> None:
    batch = {
        "synthetic_anomaly_mask": torch.tensor(
            [
                [0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    }

    assert _injected_point_indices(batch, sample_index=0) == [1, 2, 4]
    assert _injected_point_indices(batch, sample_index=1) == [0]


def test_sample_plot_annotation_names_profile_family_and_segment() -> None:
    batch = {
        "augmentation_metadata": [
            {
                "anomaly_family": "spike",
                "start_index": 3,
                "end_index": 7,
                "affected_channels": [0, 2],
                "entity_id": "machine-1-6",
                "source_start_index": 120,
            }
        ],
        "synthetic_anomaly_mask": torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0]]),
    }

    annotation = _build_sample_plot_annotation(
        profile_name="Visible profile",
        batch=batch,
        sample_index=0,
    )

    assert annotation["title"] == (
        "Visible profile | entity=machine-1-6 | window_start=120 "
        "| anomaly=spike | segment=[3, 7)"
    )
    assert annotation["injected_point_indices"] == [3, 4]
    assert annotation["affected_channels"] == [0, 2]


def test_select_most_visible_sample_channel_uses_largest_masked_delta() -> None:
    clean_batch = {"x": torch.zeros(2, 5, 3)}
    augmented_batch = {
        "x": torch.zeros(2, 5, 3),
        "synthetic_anomaly_mask": torch.tensor(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
    }
    augmented_batch["x"][0, 1:3, 1] = 2.0
    augmented_batch["x"][1, 2:4, 2] = 5.0

    sample_index, channel_index = _select_most_visible_sample_channel(
        clean_batch,
        augmented_batch,
    )

    assert sample_index == 1
    assert channel_index == 2


def test_select_most_visible_channels_returns_top_injected_channels() -> None:
    clean = torch.zeros(6, 5)
    augmented = torch.zeros(6, 5)
    mask = torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.bool)
    augmented[1:4, 4] = 5.0
    augmented[1:4, 2] = 3.0
    augmented[1:4, 0] = 1.0
    augmented[1:4, 3] = 0.5

    channels = _select_most_visible_channels(
        clean,
        augmented,
        mask,
        max_channels=3,
    )

    assert channels == [4, 2, 0]


def test_ensure_demo_batch_has_enough_channels_expands_small_demo_batch() -> None:
    batch = {
        "x": torch.ones(2, 5, 1),
        "point_labels": torch.zeros(2, 5, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "demo-0"}, {"entity_id": "demo-1"}],
    }

    expanded = _ensure_demo_batch_has_enough_channels(batch, min_channels=6)

    assert expanded["x"].shape == (2, 5, 6)
    assert torch.equal(expanded["x"][:, :, 0], batch["x"][:, :, 0])


def test_resolve_visualization_seed_keeps_explicit_seed() -> None:
    assert _resolve_visualization_seed(7) == 7


def test_resolve_visualization_seed_uses_rng_when_seed_is_missing() -> None:
    class FakeRng:
        def integers(self, low: int, high: int) -> int:
            assert low == 0
            assert high == 2**31 - 1
            return 123

    assert _resolve_visualization_seed(None, rng=FakeRng()) == 123


def test_select_random_window_indices_samples_one_index_per_entity() -> None:
    class FakeRng:
        def integers(self, low: int, high: int) -> int:
            assert low == 0
            return high - 1

    indices = _select_random_window_indices(
        dataset_lengths={"machine-1-6": 3, "machine-3-4": 5},
        rng=FakeRng(),
    )

    assert indices == {"machine-1-6": 2, "machine-3-4": 4}


def test_build_family_gallery_batches_contains_all_redlamp_families() -> None:
    clean_batch = {
        "x": torch.randn(3, 20, 6),
        "point_labels": torch.zeros(3, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [
            {"entity_id": "machine-1-6", "start_index": 0, "end_index": 20},
            {"entity_id": "machine-3-4", "start_index": 0, "end_index": 20},
            {"entity_id": "machine-3-9", "start_index": 0, "end_index": 20},
        ],
    }
    profile = {
        "window_size": 20,
        "min_segment_fraction": 0.2,
        "max_segment_fraction": 0.3,
        "spike_scale": 3.0,
        "anomaly_visibility_boost": 1.5,
        "family_intensity": {},
    }

    gallery_batches = _build_family_gallery_batches(
        profile=profile,
        clean_batch=clean_batch,
        seed=11,
    )

    assert [item["family_name"] for item in gallery_batches] == list(
        REDLAMP_ANOMALY_FAMILIES
    )
    assert len(gallery_batches) == 11

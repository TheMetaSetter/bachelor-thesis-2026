from __future__ import annotations

import torch

from scripts.compare_synthetic_profiles import (
    _ensure_demo_batch_has_enough_channels,
    _build_sample_plot_annotation,
    _injected_point_indices,
    _resolve_visualization_seed,
    _select_most_visible_channels,
    _select_most_visible_sample_channel,
)


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
            }
        ],
        "synthetic_anomaly_mask": torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0]]),
    }

    annotation = _build_sample_plot_annotation(
        profile_name="Visible profile",
        batch=batch,
        sample_index=0,
    )

    assert annotation["title"] == "Visible profile | anomaly=spike | segment=[3, 7)"
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

from __future__ import annotations

import torch

from scripts.visualize_synthetic_anomalies import (
    save_synthetic_anomaly_visualization,
    plot_synthetic_anomaly_mask,
    select_channels_to_plot,
    select_visible_affected_channels,
)
import matplotlib.pyplot as plt
from src.data.augment import SyntheticAnomalyInjector


def test_select_channels_to_plot_prioritizes_affected_channels() -> None:
    clean_window = torch.tensor(
        [
            [0.0, 2.0, 0.0, 1.0],
            [0.2, 2.0, 0.1, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [1.03, 2.0, 0.0, 1.0],
            [1.05, 2.0, 0.2, 1.08],
            [0.7, 2.0, 0.0, 1.0],
            [0.6, 2.0, 0.0, 1.0],
            [0.5, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    metadata = {"start_index": 2, "end_index": 5}
    selected_channels = select_channels_to_plot(
        clean_window=clean_window,
        metadata=metadata,
        num_channels=7,
        affected_channels=[1, 0, 3, 2],
    )

    assert selected_channels == [3, 0, 1, 2, 4]


def test_select_channels_to_plot_fills_with_context_channels() -> None:
    clean_window = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.15, 0.3],
            [0.4, 0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    metadata = {"start_index": 1, "end_index": 4}
    selected_channels = select_channels_to_plot(
        clean_window=clean_window,
        metadata=metadata,
        num_channels=4,
        affected_channels=[2],
    )

    assert selected_channels == [2, 0, 1, 3]


def test_select_visible_affected_channels_filters_by_delta_range() -> None:
    clean_window = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [1.0, 0.1, 0.2],
            [1.05, 0.1, 0.22],
            [1.08, 0.104, 0.25],
        ],
        dtype=torch.float32,
    )
    metadata = {"start_index": 2, "end_index": 5}

    visible_channels = select_visible_affected_channels(
        clean_window=clean_window,
        metadata=metadata,
        num_channels=3,
        affected_channels=[0, 1, 2],
    )

    assert visible_channels == [0, 2]


def test_synthetic_anomaly_visualization_writes_artifact(tmp_path) -> None:
    clean_batch = {
        "x": torch.randn(2, 20, 3),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=0.2,
        max_segment_fraction=0.3,
        spike_scale=3.0,
        anomaly_families=("flip",),
        train_balance_classes=False,
    )
    augmented_batch = injector.augment_batch(clean_batch)

    output_path = save_synthetic_anomaly_visualization(
        clean_batch=clean_batch,
        augmented_batch=augmented_batch,
        output_path=tmp_path / "synthetic_anomaly.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert augmented_batch["augmentation_metadata"][0]["anomaly_family"] == "flip"


def test_plot_synthetic_anomaly_mask_uses_time_aligned_extent() -> None:
    anomaly_mask = torch.tensor([0, 0, 1, 1, 0], dtype=torch.long)
    figure, axis = plt.subplots()

    plot_synthetic_anomaly_mask(axis, anomaly_mask)

    assert tuple(axis.images[0].get_extent()) == (0, 5, 0, 1)
    plt.close(figure)

from __future__ import annotations

import torch

from scripts.visualize_synthetic_anomalies import save_synthetic_anomaly_visualization
from src.data.augment import SyntheticAnomalyInjector


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
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        anomaly_families=("flip",),
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

from __future__ import annotations

import numpy as np
import pytest

from scripts.visualize_smd_train_test_entity import (
    labels_to_spans,
    load_smd_entity_series,
    save_smd_train_test_visualization,
)


def test_labels_to_spans_extracts_half_open_intervals() -> None:
    labels = np.asarray([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int64)

    assert labels_to_spans(labels) == [(1, 3), (4, 5), (6, 8)]


def test_load_smd_entity_series_requires_matching_test_and_label_lengths(
    tmp_path,
) -> None:
    dataset_root = tmp_path / "ServerMachineDataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir(parents=True)
    (dataset_root / "test_label").mkdir(parents=True)

    np.savetxt(
        dataset_root / "train" / "machine-x.txt",
        np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        delimiter=",",
    )
    np.savetxt(
        dataset_root / "test" / "machine-x.txt",
        np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        delimiter=",",
    )
    np.savetxt(
        dataset_root / "test_label" / "machine-x.txt",
        np.asarray([0, 1, 0], dtype=np.int64),
        delimiter=",",
        fmt="%d",
    )

    with pytest.raises(ValueError, match="test sequence length must match test_label length"):
        load_smd_entity_series(dataset_root=dataset_root, entity_id="machine-x")


def test_save_smd_train_test_visualization_writes_png(tmp_path) -> None:
    output_path = tmp_path / "entity_plot.png"

    resolved_path = save_smd_train_test_visualization(
        train_values=np.random.randn(12, 3).astype(np.float32),
        test_values=np.random.randn(10, 3).astype(np.float32),
        test_labels=np.asarray([0, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int64),
        entity_id="machine-x",
        output_path=output_path,
    )

    assert resolved_path.exists()
    assert resolved_path.stat().st_size > 0

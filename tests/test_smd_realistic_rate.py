from __future__ import annotations

from pathlib import Path

from src.data.datasets.smd import compute_smd_test_window_anomaly_rate


def _write_series(path: Path, values: list[list[float]]) -> None:
    path.write_text("\n".join(",".join(str(value) for value in row) for row in values), encoding="utf-8")


def _write_labels(path: Path, labels: list[int]) -> None:
    path.write_text("\n".join(str(value) for value in labels), encoding="utf-8")


def test_compute_smd_test_window_anomaly_rate_supports_scope_and_all_entities(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "ServerMachineDataset"
    (root_dir / "train").mkdir(parents=True)
    (root_dir / "test").mkdir(parents=True)
    (root_dir / "test_label").mkdir(parents=True)

    # machine-a: no anomalous points in test labels.
    _write_series(
        root_dir / "train" / "machine-a.txt",
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    _write_series(
        root_dir / "test" / "machine-a.txt",
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    _write_labels(root_dir / "test_label" / "machine-a.txt", [0, 0, 0, 0])

    # machine-b: one anomalous point in the middle.
    _write_series(
        root_dir / "train" / "machine-b.txt",
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    _write_series(
        root_dir / "test" / "machine-b.txt",
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    )
    _write_labels(root_dir / "test_label" / "machine-b.txt", [0, 1, 0, 0])

    # window_size=2, stride=1 => each entity has 3 windows.
    # machine-a anomalous windows: 0/3, machine-b anomalous windows: 2/3.
    same_scope_rate = compute_smd_test_window_anomaly_rate(
        root_dir=root_dir,
        window_size=2,
        stride=1,
        entity_ids=["machine-a"],
        use_all_entities=False,
    )
    all_entities_rate = compute_smd_test_window_anomaly_rate(
        root_dir=root_dir,
        window_size=2,
        stride=1,
        entity_ids=["machine-a"],
        use_all_entities=True,
    )

    assert same_scope_rate == 0.0
    assert all_entities_rate == (2 / 6)

from __future__ import annotations

from pathlib import Path

from scripts.rank_smd_train_test_normal_drift import (
    build_smd_normal_drift_ranking,
)


def _write_smd_entity_files(
    dataset_root: Path,
    *,
    entity_id: str,
    train_rows: list[str],
    test_rows: list[str],
    label_rows: list[str],
) -> None:
    (dataset_root / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "test").mkdir(parents=True, exist_ok=True)
    (dataset_root / "test_label").mkdir(parents=True, exist_ok=True)
    (dataset_root / "train" / f"{entity_id}.txt").write_text(
        "\n".join(train_rows),
        encoding="utf-8",
    )
    (dataset_root / "test" / f"{entity_id}.txt").write_text(
        "\n".join(test_rows),
        encoding="utf-8",
    )
    (dataset_root / "test_label" / f"{entity_id}.txt").write_text(
        "\n".join(label_rows),
        encoding="utf-8",
    )


def test_build_smd_normal_drift_ranking_returns_one_row_per_entity(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "ServerMachineDataset"
    _write_smd_entity_files(
        dataset_root,
        entity_id="machine-a",
        train_rows=["0.0,0.0", "0.0,0.0", "0.0,0.0"],
        test_rows=["10.0,10.0", "10.0,10.0", "10.0,10.0"],
        label_rows=["0", "0", "0"],
    )
    _write_smd_entity_files(
        dataset_root,
        entity_id="machine-b",
        train_rows=["1.0,1.0", "1.0,1.0", "1.0,1.0"],
        test_rows=["1.0,1.0", "50.0,50.0", "1.0,1.0"],
        label_rows=["0", "1", "0"],
    )

    ranking_rows = build_smd_normal_drift_ranking(
        root_dir=tmp_path,
        bins=4,
        smoothing=1.0e-9,
    )

    assert [row.entity_id for row in ranking_rows] == ["machine-a", "machine-b"]
    assert ranking_rows[0].num_test_normal_points == 3
    assert ranking_rows[1].num_test_normal_points == 2
    assert (
        ranking_rows[0].mean_kl_test_to_train >= ranking_rows[1].mean_kl_test_to_train
    )


def test_build_smd_normal_drift_ranking_ignores_anomalous_test_rows(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "ServerMachineDataset"
    _write_smd_entity_files(
        dataset_root,
        entity_id="machine-a",
        train_rows=["0.0,0.0", "0.0,0.0", "0.0,0.0"],
        test_rows=["0.0,0.0", "100.0,100.0", "0.0,0.0"],
        label_rows=["0", "1", "0"],
    )

    ranking_rows = build_smd_normal_drift_ranking(
        root_dir=tmp_path,
        bins=4,
        smoothing=1.0e-9,
    )

    assert len(ranking_rows) == 1
    assert ranking_rows[0].num_test_normal_points == 2
    assert ranking_rows[0].mean_kl_test_to_train == 0.0

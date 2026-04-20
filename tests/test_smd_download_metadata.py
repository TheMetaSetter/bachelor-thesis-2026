from __future__ import annotations

from pathlib import Path

import pytest

from src.data.download import (
    get_smd_dataset_root,
    is_smd_dataset_present,
    list_smd_download_plan,
)


def test_smd_download_plan_points_to_canonical_dataset_layout() -> None:
    download_plan = list_smd_download_plan("data")
    planned_paths = {record.local_path for record in download_plan}

    assert str(get_smd_dataset_root("data") / "train") in planned_paths
    assert str(get_smd_dataset_root("data") / "test") in planned_paths
    assert str(get_smd_dataset_root("data") / "test_label") in planned_paths


def test_smd_dataset_presence_check_recognizes_local_workspace_dataset() -> None:
    assert is_smd_dataset_present("data/ServerMachineDataset") is True


def test_smd_dataset_root_prefers_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMD_ROOT_DIR", "/tmp/custom-smd-root")

    resolved_root = get_smd_dataset_root("data/ServerMachineDataset")

    assert resolved_root == Path("/tmp/custom-smd-root/ServerMachineDataset")

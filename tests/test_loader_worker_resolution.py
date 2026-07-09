from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.data.datasets.smd import SMDDatasetParser
from src.data.loaders import _resolve_data_loader_num_workers, _resolve_smd_root_dir


def test_resolve_data_loader_num_workers_supports_explicit_integer() -> None:
    resolved_num_workers = _resolve_data_loader_num_workers({"num_workers": 6})

    assert resolved_num_workers == 6


def test_resolve_data_loader_num_workers_uses_auto_cpu_count_with_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 2)

    resolved_num_workers = _resolve_data_loader_num_workers(
        {
            "num_workers": "auto",
            "min_num_workers": 4,
        }
    )

    assert resolved_num_workers == 4


def test_resolve_data_loader_num_workers_uses_auto_cpu_count_when_above_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 12)

    resolved_num_workers = _resolve_data_loader_num_workers(
        {
            "num_workers": "auto",
            "min_num_workers": 4,
        }
    )

    assert resolved_num_workers == 12


def test_resolve_data_loader_num_workers_rejects_unknown_string_value() -> None:
    with pytest.raises(ValueError, match="data.num_workers"):
        _resolve_data_loader_num_workers({"num_workers": "many"})


def test_resolve_smd_root_dir_prefers_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SMD_ROOT_DIR", "/tmp/kaggle-smd")

    resolved_root_dir = _resolve_smd_root_dir({"root_dir": "data/ServerMachineDataset"})

    assert Path(resolved_root_dir) == Path("/tmp/kaggle-smd/ServerMachineDataset")


def test_smd_parser_resolves_repo_relative_data_root_from_other_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    parser = SMDDatasetParser(
        root_dir="data/ServerMachineDataset",
        validation_split_ratio=0.2,
        entity_ids=["machine-1-6"],
    )
    parsed_sequences = parser.parse()

    assert len(parsed_sequences["train"]) == 1
    assert parsed_sequences["train"][0]["meta"]["entity_id"] == "machine-1-6"

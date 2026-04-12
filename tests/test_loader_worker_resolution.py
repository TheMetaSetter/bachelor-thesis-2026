from __future__ import annotations

import os
from pathlib import Path

import pytest

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

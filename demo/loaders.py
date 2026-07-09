from __future__ import annotations

"""Artifact loaders for the thesis demo."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.core.config import load_experiment_config
from src.core.registry import build_dataset
from src.core.runtime_components import register_evaluation_runtime_components


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _to_numpy(array_like: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=dtype)


def _resolve_output_dir(report: dict[str, Any], report_path: Path) -> Path:
    artifact_paths = report.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        thresholds_path = artifact_paths.get("thresholds")
        if thresholds_path is not None:
            return Path(thresholds_path).resolve().parents[1]
    return report_path.resolve().parents[1]


def _resolve_data_config_path(report: dict[str, Any]) -> Path:
    experiment_config_path = report.get("experiment_config_path")
    if isinstance(experiment_config_path, str) and experiment_config_path:
        experiment_config = load_experiment_config(experiment_config_path)
        data_config_path = experiment_config.get("data_config_path")
        if isinstance(data_config_path, str) and data_config_path:
            return Path(data_config_path)
        data_config = experiment_config.get("data")
        if isinstance(data_config, dict):
            nested_path = data_config.get("data_config_path")
            if isinstance(nested_path, str) and nested_path:
                return Path(nested_path)

    benchmark_config = report.get("benchmark_config")
    if isinstance(benchmark_config, dict):
        data_config_path = benchmark_config.get("data_config_path")
        if isinstance(data_config_path, str) and data_config_path:
            return Path(data_config_path)
        data_config = benchmark_config.get("data")
        if isinstance(data_config, dict):
            nested_path = data_config.get("data_config_path")
            if isinstance(nested_path, str) and nested_path:
                return Path(nested_path)

    raise ValueError("Report does not provide a data config path")


def _load_test_sequence(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    data_config = _load_yaml(_resolve_data_config_path(report))
    register_evaluation_runtime_components()
    data_bundle = build_dataset(data_config["dataset_name"], data_config)
    test_sequences = data_bundle.get("scaled_sequences", {}).get("test")
    if not test_sequences:
        raise ValueError(f"Report {report_path} does not resolve a test sequence")
    if len(test_sequences) != 1:
        raise ValueError("Demo expects one test sequence per report")
    return test_sequences[0]


def load_report_payload(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    return _load_json(path)


def load_sequence_values(sequence: dict[str, Any]) -> np.ndarray:
    return _to_numpy(sequence["x"], dtype=np.float64)


def load_sequence_labels(sequence: dict[str, Any]) -> np.ndarray:
    labels = sequence.get("point_labels")
    if labels is None:
        raise ValueError("Demo requires point_labels for replay")
    return _to_numpy(labels, dtype=np.int64).reshape(-1)


def load_demo_test_sequence(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    report = load_report_payload(path)
    return _load_test_sequence(report, path)


def resolve_output_root(report_path: str | Path) -> Path:
    path = Path(report_path)
    report = load_report_payload(path)
    return _resolve_output_dir(report, path)

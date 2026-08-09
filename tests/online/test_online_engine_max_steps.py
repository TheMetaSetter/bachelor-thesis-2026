from __future__ import annotations

from typing import Any, Iterator
from pathlib import Path

import torch
import pytest

from src.engine.online_tta.online_engine import _run_online_sequence
from src.engine.online_tta import online_engine_run as online_engine_run_module
from src.engine.online_tta.verification_buffer import VerificationBuffer
from scripts.experiments.run_online_adaptation import (
    _resolve_max_online_steps as resolve_online_adaptation_max_steps,
)


def _fake_batch_stream() -> Iterator[dict[str, Any]]:
    for step in range(5):
        yield {
            "x": torch.zeros(1, 20, 38),
            "view_a": torch.zeros(1, 20, 38),
            "view_b": torch.zeros(1, 20, 38),
            "point_labels": torch.zeros(1, 20, dtype=torch.long),
            "mask": torch.ones(1, 20, 38),
            "timestamps": torch.arange(20).unsqueeze(0),
            "meta": [
                {
                    "dataset_name": "smd",
                    "entity_id": "machine-1-6",
                    "split": "test",
                    "stream_step": step,
                }
            ],
        }


def test_run_online_sequence_honors_max_online_steps(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    monkeypatch.setattr(
        "src.engine.online_tta.online_engine._build_online_stream",
        lambda **kwargs: _fake_batch_stream(),
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine._process_online_window",
        lambda **kwargs: (
            None,
            {"online/step": 1},
            {"step": 1, "entity_ids": ["machine-1-6"]},
        ),
    )

    metric_history, records = _run_online_sequence(
        model=model,
        optimizer=optimizer,
        sequence={"meta": {"entity_id": "machine-1-6"}},
        online_variant="A0",
        threshold_value=0.0,
        protocol_config={
            "window_size": 20,
            "online_ewma_current_weight": 0.9,
            "online_ewma_previous_weight": 0.1,
        },
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device="cpu",
        verification_buffer=VerificationBuffer(max_size=8, non_overlap_gap=0),
        max_online_steps=2,
    )

    assert len(metric_history) == 2
    assert len(records) == 2


def test_online_event_callback_receives_an_isolated_record(monkeypatch) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    record = {
        "causal_window": {"absolute_indices": [0, 1]},
        "window_point_predictions": [0, 1],
    }

    monkeypatch.setattr(
        "src.engine.online_tta.online_engine._build_online_stream",
        lambda **kwargs: _fake_batch_stream(),
    )
    monkeypatch.setattr(
        "src.engine.online_tta.online_engine._process_online_window",
        lambda **kwargs: (None, {"online/step": 1}, record),
    )

    def mutate_and_fail(callback_record: dict[str, Any]) -> None:
        callback_record["causal_window"]["absolute_indices"].clear()
        raise RuntimeError("demo is unavailable")

    _, records = _run_online_sequence(
        model=model,
        optimizer=optimizer,
        sequence={"meta": {"entity_id": "machine-1-6"}},
        online_variant="A0",
        threshold_value=0.0,
        protocol_config={
            "window_size": 20,
            "online_ewma_current_weight": 0.9,
            "online_ewma_previous_weight": 0.1,
        },
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device="cpu",
        verification_buffer=VerificationBuffer(max_size=8, non_overlap_gap=0),
        max_online_steps=1,
        event_callback=mutate_and_fail,
    )

    assert records == [record]
    assert record["causal_window"]["absolute_indices"] == [0, 1]


def test_run_online_sequence_rejects_batched_causal_windows(monkeypatch) -> None:
    def _batched_stream() -> Iterator[dict[str, Any]]:
        yield {
            "x": torch.zeros(2, 20, 38),
            "view_a": torch.zeros(2, 20, 38),
            "view_b": torch.zeros(2, 20, 38),
            "point_labels": torch.zeros(2, 20, dtype=torch.long),
            "mask": torch.ones(2, 20, 38),
            "timestamps": torch.arange(20).repeat(2, 1),
            "meta": [
                {
                    "dataset_name": "smd",
                    "entity_id": "machine-1-6",
                    "split": "test",
                    "stream_step": 0,
                },
                {
                    "dataset_name": "smd",
                    "entity_id": "machine-1-6",
                    "split": "test",
                    "stream_step": 1,
                },
            ],
        }

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    monkeypatch.setattr(
        "src.engine.online_tta.online_engine._build_online_stream",
        lambda **kwargs: _batched_stream(),
    )

    with pytest.raises(ValueError, match="exactly one causal window"):
        _run_online_sequence(
            model=model,
            optimizer=optimizer,
            sequence={"meta": {"entity_id": "machine-1-6"}},
            online_variant="A0",
            threshold_value=0.0,
            protocol_config={
                "window_size": 20,
                "online_ewma_current_weight": 0.9,
                "online_ewma_previous_weight": 0.1,
            },
            batch_size=2,
            view_noise_std=0.0,
            view_dropout_probability=0.0,
            device="cpu",
            verification_buffer=VerificationBuffer(max_size=8, non_overlap_gap=0),
            max_online_steps=1,
        )


def test_resolve_max_online_steps_treats_none_as_unbounded() -> None:
    assert online_engine_run_module._resolve_max_online_steps(None) is None
    assert online_engine_run_module._resolve_max_online_steps(16) == 16
    assert resolve_online_adaptation_max_steps(None) is None
    assert resolve_online_adaptation_max_steps(16) == 16


def test_build_runtime_online_context_keeps_none_max_online_steps(monkeypatch) -> None:
    class _DummyScaler:
        def state_dict(self) -> dict[str, Any]:
            return {"scale": 1.0}

    class _DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def to(self, *args: Any, **kwargs: Any) -> "_DummyModel":
            return self

        def set_point_score_calibration(self, calibration) -> None:
            self.point_score_calibration = calibration

    class _DummyCheckpointManager:
        def __init__(self, checkpoint_dir: Path) -> None:
            self.checkpoint_dir = checkpoint_dir

    monkeypatch.setattr(
        online_engine_run_module,
        "build_dataset",
        lambda dataset_name, data_config: {
            "scaled_sequences": {
                "val": [{"meta": {"entity_id": "machine-1-6"}}],
                "test": [
                    {"meta": {"entity_id": "machine-1-6"}, "x": torch.zeros(1, 1, 1)}
                ],
            },
            "scaler": _DummyScaler(),
        },
    )
    monkeypatch.setattr(
        online_engine_run_module,
        "_build_model_from_experiment_config",
        lambda experiment_config: _DummyModel(),
    )
    monkeypatch.setattr(
        online_engine_run_module,
        "assert_only_projector_is_trainable",
        lambda model: None,
    )
    monkeypatch.setattr(
        online_engine_run_module,
        "CheckpointManager",
        _DummyCheckpointManager,
    )
    monkeypatch.setattr(
        online_engine_run_module,
        "resolve_threshold_artifact",
        lambda _: Path("/tmp/thresholds.json"),
    )
    monkeypatch.setattr(
        online_engine_run_module, "sha256_file", lambda _: "checkpoint-sha"
    )
    monkeypatch.setattr(
        online_engine_run_module,
        "load_threshold_artifact",
        lambda _: {
            "entity_id": "machine-1-6",
            "variant_name": "O0",
            "seed": 6,
            "window_size": 20,
            "checkpoint_sha256": "checkpoint-sha",
            "ewma_current_weight": 0.9,
            "ewma_previous_weight": 0.1,
            "point_score_transform": "shifted-and-scaled logistic sigmoid",
            "point_score_c": 0.2,
            "point_score_tau": 1.0,
            "point_score_tau_estimator": "mad_based_robust_scale",
            "point_score_mad_normalizer": 0.6745,
            "thresholds": {"online_ewma_point": {"value": 0.5}},
        },
    )
    monkeypatch.setattr(
        online_engine_run_module,
        "build_online_runtime_state",
        lambda **kwargs: {"entity_id": "machine-1-6"},
    )

    context = online_engine_run_module._build_runtime_online_context(
        experiment_config={
            "data": {
                "dataset_name": "smd",
                "batch_size": 1,
                "window_size": 20,
            },
            "task": {
                "view_noise_std": 0.0,
                "view_dropout_probability": 0.0,
                "max_online_steps": None,
                "reference_checkpoint_path": "/tmp/reference.pt",
                "threshold_artifact_path": "/tmp/thresholds.json",
                "offline_variant": "O0",
                "entity_id": "machine-1-6",
                "seed": 6,
            },
            "device": "cpu",
            "checkpoint_dir": "/tmp/checkpoints",
            "output_dir": "/tmp/outputs",
        },
        protocol_config={
            "window_size": 20,
            "online_ewma_current_weight": 0.9,
            "online_ewma_previous_weight": 0.1,
        },
        online_variant="A0",
    )

    assert context["max_online_steps"] is None
    assert [key for key in context if key.endswith("_buffer")] == [
        "verification_buffer"
    ]

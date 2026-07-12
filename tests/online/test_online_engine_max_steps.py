from __future__ import annotations

from typing import Any, Iterator

import torch
import pytest

from src.engine.online_tta.online_engine import _run_online_sequence
from src.engine.online_tta.ttl_buffer import TTLBuffer
from src.engine.online_tta.verification_buffer import VerificationBuffer


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
        ttl_buffer=TTLBuffer(ttl_steps=20),
        max_online_steps=2,
    )

    assert len(metric_history) == 2
    assert len(records) == 2


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
            ttl_buffer=TTLBuffer(ttl_steps=20),
            max_online_steps=1,
        )

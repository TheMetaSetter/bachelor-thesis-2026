from __future__ import annotations

import torch

from src.core.contracts import validate_online_batch
from src.engine.online_tta.triage import classify_online_window
from src.engine.online_tta.verification_adapter import build_entry_batch


def _one_window_batch() -> dict[str, object]:
    return {
        "x": torch.zeros(1, 20, 3),
        "point_labels": None,
        "mask": None,
        "timestamps": torch.arange(20).unsqueeze(0),
        "absolute_indices": torch.arange(20).unsqueeze(0),
        "meta": [{"entity_id": "machine-1-6", "start_index": 0, "end_index": 20}],
    }


def test_full_spec_online_batch_requires_only_one_window() -> None:
    batch = _one_window_batch()
    validate_online_batch(batch)
    assert "view_a" not in batch
    assert "view_b" not in batch


def test_verification_batch_is_label_free_and_single_window() -> None:
    batch = build_entry_batch(
        {
            "window": torch.zeros(20, 3),
            "entity_id": "machine-1-6",
            "window_start": 4,
            "window_end": 24,
            "stream_step": 24,
        },
        "cpu",
    )
    validate_online_batch(batch)
    assert batch["point_labels"] is None
    assert "view_a" not in batch and "view_b" not in batch


def test_full_spec_triage_has_exact_four_regions() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }
    decisions = {
        classify_online_window(0.2, 0.9, thresholds),
        classify_online_window(0.3, 0.4, thresholds),
        classify_online_window(0.3, 0.8, thresholds),
        classify_online_window(0.3, 0.9, thresholds),
    }
    assert decisions == {"normal", "hard_old_normality", "gray_zone", "strong_anomaly"}

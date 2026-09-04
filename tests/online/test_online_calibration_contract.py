from __future__ import annotations

import numpy as np
import torch

from src.data.scalers import SequenceStandardScaler
from src.engine.online_tta.online_calibration import (
    collect_nonoverlap_offline_scores,
    collect_stride1_online_scores,
)
from src.engine.online_tta.online_engine_window_metrics import (
    _extract_online_window_scores,
)


class _CalibrationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_source_calls = 0
        self.forward_calls = 0

    def _build_output(self, batch, *, base: float) -> dict[str, object]:
        start_index = int(batch["meta"][0]["start_index"])
        point_scores = torch.tensor([[base + start_index, base + start_index + 1.0]])
        recon = batch["x"].clone()
        latent_window_score = torch.tensor([base + start_index + 2.0])
        return {
            "point_scores": point_scores,
            "recon": recon,
            "aux": {"latent_window_score": latent_window_score},
        }

    def forward_source(self, batch):
        self.forward_source_calls += 1
        return self._build_output(batch, base=0.0)

    def forward(self, batch):
        self.forward_calls += 1
        return self._build_output(batch, base=10.0)


def _sequence(length: int = 6) -> dict[str, object]:
    x = torch.arange(length * 2, dtype=torch.float32).reshape(length, 2)
    return {
        "x": x,
        "point_labels": torch.zeros(length, dtype=torch.long),
        "mask": torch.ones(length, 2),
        "timestamps": torch.arange(length),
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-1-6",
            "split": "val",
            "num_channels": 2,
            "sequence_length": length,
            "source_sequence_length": length,
        },
    }


def test_collect_nonoverlap_offline_scores_respects_window_stride() -> None:
    model = _CalibrationModel()
    scores = collect_nonoverlap_offline_scores(
        model=model,
        clean_validation_sequences=[_sequence()],
        window_size=2,
        device="cpu",
    )
    assert model.forward_source_calls == 3
    assert model.forward_calls == 0
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_collect_stride1_online_scores_keeps_overlapping_window_points() -> None:
    model = _CalibrationModel()
    scores = collect_stride1_online_scores(
        model=model,
        clean_validation_sequences=[_sequence()],
        window_size=2,
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device="cpu",
        current_weight=0.9,
        previous_weight=0.1,
    )
    assert model.forward_source_calls == 0
    assert model.forward_calls > 0
    assert scores["point"] == [
        10.0,
        11.0,
        11.0,
        12.0,
        12.0,
        13.0,
        13.0,
        14.0,
        14.0,
        15.0,
    ]
    assert len(scores["ewma"]) == len(scores["point"])
    assert all(np.isfinite(np.asarray(scores["input_window"], dtype=float)))
    assert all(np.isfinite(np.asarray(scores["latent_window"], dtype=float)))


def test_collect_stride1_online_scores_uses_a0_source_path() -> None:
    model = _CalibrationModel()
    model.online_variant = "A0"

    collect_stride1_online_scores(
        model=model,
        clean_validation_sequences=[_sequence()],
        window_size=2,
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device="cpu",
        current_weight=0.9,
        previous_weight=0.1,
    )

    assert model.forward_source_calls > 0
    assert model.forward_calls == 0


def test_collect_stride1_online_scores_uses_raw_input_mse_when_scaler_is_given() -> None:
    model = _CalibrationModel()
    scaler = SequenceStandardScaler()
    scaler.fit([_sequence()])

    scores = collect_stride1_online_scores(
        model=model,
        clean_validation_sequences=[_sequence()],
        window_size=2,
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device="cpu",
        current_weight=0.9,
        previous_weight=0.1,
        scaler=scaler,
    )

    assert scores["point"] == [0.0] * 10
    assert scores["input_window"] == [0.0] * 5


def test_online_window_score_extraction_uses_raw_input_mse() -> None:
    scaler = SequenceStandardScaler()
    scaler.fit([_sequence()])
    batch = {"x": torch.zeros(1, 2, 2)}
    outputs = {
        "recon": torch.ones(1, 2, 2),
        "point_scores": torch.full((1, 2), 99.0),
        "window_scores": torch.tensor([7.0]),
        "aux": {"latent_window_score": torch.tensor([8.0])},
    }

    raw_point, normalized_point, raw_window, normalized_window, latent = (
        _extract_online_window_scores(outputs, batch, scaler=scaler)
    )

    expected_raw_mse = scaler.feature_std.square().mean()
    assert torch.allclose(raw_point, torch.full((2,), expected_raw_mse))
    assert torch.allclose(normalized_point, torch.tensor([1.0, 1.0]))
    assert raw_window == float(expected_raw_mse)
    assert normalized_window == 1.0
    assert latent == 8.0

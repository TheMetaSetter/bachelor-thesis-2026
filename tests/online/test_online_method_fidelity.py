from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from src.baselines.online import CANDIStreamingBaseline, M2N2StreamingBaseline
from src.models.online_redlamp_reconstruction import (
    RedLampReconstructionModel,
    load_redlamp_reconstruction_checkpoint,
)
from src.models.simple_window_cnn_autoencoder import SimpleWindowCnnAutoencoder


def _sequence(seed: int, length: int = 40) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    return {
        "x": rng.normal(size=(length, 3)).astype(np.float64),
        "point_labels": np.zeros(length, dtype=np.int64),
        "meta": {"entity_id": "machine-1-6"},
    }


def _encoder_checkpoint(tmp_path: Path) -> Path:
    model = SimpleWindowCnnAutoencoder(
        input_dim=3,
        latent_dim=128,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3,
        dropout=0.1,
    )
    checkpoint = tmp_path / "redlamp_encoder.pt"
    torch.save(
        {"model_state_dict": model.encoder.state_dict(), "epoch": 100}, checkpoint
    )
    return checkpoint


def _protocol() -> dict[str, float]:
    return {
        "online_ewma_current_weight": 0.9,
        "online_ewma_previous_weight": 0.1,
    }


def test_encoder_checkpoint_contract_initializes_adapter_head(tmp_path: Path) -> None:
    checkpoint = _encoder_checkpoint(tmp_path)
    model = RedLampReconstructionModel(
        input_dim=3,
        window_size=8,
        latent_dim=128,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3,
        dropout=0.1,
    )
    identity = load_redlamp_reconstruction_checkpoint(
        model=model, checkpoint_path=checkpoint
    )

    assert identity.checkpoint_role == "pretrained_encoder"
    assert identity.checkpoint_contract == "reference_adapter_redlamp_encoder"
    assert identity.checkpoint_sha256
    assert tuple(model(torch.zeros(1, 8, 3)).shape) == (1, 8, 3)


def test_legacy_main_variant_is_rejected_for_adaptive_baselines(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="reference_adapter_redlamp_encoder"):
        M2N2StreamingBaseline(
            train_sequence=_sequence(0)["x"],
            window_size=8,
            online_variant="main",
            pretrained_encoder_checkpoint=_encoder_checkpoint(tmp_path),
        )


def test_m2n2_uses_detrender_mask_and_optimizer_step(tmp_path: Path) -> None:
    train = _sequence(0)["x"]
    baseline = M2N2StreamingBaseline(
        train_sequence=train,
        window_size=8,
        pretrained_encoder_checkpoint=_encoder_checkpoint(tmp_path),
        m2n2_gamma=0.9,
    )
    before = {
        key: value.detach().clone()
        for key, value in baseline.backbone_.decoder.state_dict().items()
    }
    window = torch.as_tensor(_sequence(1, 8)["x"][None], dtype=torch.float32)
    result = baseline._adapt_tensor(window, score=0.0, threshold=1.0e9)

    assert isinstance(baseline.optimizer_, torch.optim.SGD)
    assert result["did_update"] is True
    assert result["mask_count"] == 8
    assert np.isfinite(result["loss_total"])
    assert not torch.equal(
        before["0.weight"], baseline.backbone_.decoder.state_dict()["0.weight"]
    )
    assert float(baseline.detrender.mean.abs().sum()) > 0.0


def test_candi_pool_gate_and_sana_update_are_method_owned(tmp_path: Path) -> None:
    baseline = CANDIStreamingBaseline(
        train_sequence=_sequence(0)["x"],
        window_size=8,
        pretrained_encoder_checkpoint=_encoder_checkpoint(tmp_path),
        candi_use_fpm=False,
        candi_min_samples=3,
        sana_type="Linear",
    )
    window = torch.as_tensor(_sequence(1, 8)["x"][None], dtype=torch.float32)
    first = baseline._adapt_tensor(window, score=0.0, threshold=1.0e9)
    second = baseline._adapt_tensor(window, score=0.0, threshold=1.0e9)
    third = baseline._adapt_tensor(window, score=0.0, threshold=1.0e9)

    assert first["did_update"] is False
    assert second["did_update"] is False
    assert third["did_update"] is True
    assert third["loss_total"] is not None
    assert third["candidate_pool_moderate_size"] == 0


def test_stream_records_score_before_adaptation(tmp_path: Path) -> None:
    baseline = M2N2StreamingBaseline(
        train_sequence=_sequence(0)["x"],
        window_size=8,
        pretrained_encoder_checkpoint=_encoder_checkpoint(tmp_path),
    )
    baseline.calibrate(
        clean_validation_sequences=[_sequence(1)],
        protocol_config=_protocol(),
        device="cpu",
    )
    state = {"value": 0.0}

    def score(self: object, x: torch.Tensor) -> tuple[float, float]:
        del x
        return state["value"], 0.0

    def adapt(
        self: object, x: torch.Tensor, score: float, threshold: float
    ) -> dict[str, object]:
        del x, score, threshold
        state["value"] += 1.0
        return {"decision": "probe", "did_update": True, "loss_total": 0.0}

    baseline._score_tensor = MethodType(score, baseline)
    baseline._adapt_tensor = MethodType(adapt, baseline)
    _, records = baseline.run_sequence(
        sequence=_sequence(2, 11),
        threshold_value=10.0,
        protocol_config=_protocol(),
        device="cpu",
    )

    assert [record["raw_point_score"] for record in records] == [0.0, 1.0, 2.0, 3.0]


def test_batch_lifecycle_scores_all_windows_before_one_update(tmp_path: Path) -> None:
    baseline = M2N2StreamingBaseline(
        train_sequence=_sequence(0)["x"],
        window_size=8,
        adaptation_batch_size=2,
        pretrained_encoder_checkpoint=_encoder_checkpoint(tmp_path),
    )
    baseline.calibrate(
        clean_validation_sequences=[_sequence(1)],
        protocol_config=_protocol(),
        device="cpu",
    )
    events: list[tuple[str, int]] = []
    state = {"next_score": 0.0}

    def score_batch(self: object, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        del self
        batch_size = int(x.shape[0])
        events.append(("score", batch_size))
        scores = np.arange(state["next_score"], state["next_score"] + batch_size)
        state["next_score"] += batch_size
        return scores, np.zeros(batch_size, dtype=np.float64)

    def adapt_batch(
        self: object, x: torch.Tensor, score: np.ndarray, threshold: float
    ) -> dict[str, object]:
        del self, score, threshold
        events.append(("adapt", int(x.shape[0])))
        return {"decision": "probe", "did_update": True, "loss_total": 0.0}

    baseline._score_tensor_batch = MethodType(score_batch, baseline)
    baseline._adapt_tensor = MethodType(adapt_batch, baseline)
    _, records = baseline.run_sequence(
        sequence=_sequence(2, 13),
        threshold_value=100.0,
        protocol_config=_protocol(),
        device="cpu",
    )

    assert events == [
        ("score", 2),
        ("adapt", 2),
        ("score", 2),
        ("adapt", 2),
        ("score", 2),
        ("adapt", 2),
    ]
    assert [record["raw_point_score"] for record in records] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_candi_fpm_reads_raw_current_input(tmp_path: Path) -> None:
    baseline = CANDIStreamingBaseline(
        train_sequence=_sequence(0)["x"],
        window_size=8,
        candi_anomaly_ratio=5.0,
        sana_type="Linear",
        pretrained_encoder_checkpoint=_encoder_checkpoint(tmp_path),
    )
    validation = _sequence(1)
    windows = np.stack(
        [validation["x"][index : index + 8] for index in range(33)], axis=0
    ).astype(np.float32)
    validation_scores = np.arange(33, dtype=np.float64)
    baseline._calibration_complete(windows, validation_scores)

    observed: list[torch.Tensor] = []
    original_get_representations = baseline.backbone_.get_representations

    def capture(self: object, x: torch.Tensor) -> torch.Tensor:
        del self
        observed.append(x.detach().clone())
        return original_get_representations(x)

    baseline.backbone_.get_representations = MethodType(capture, baseline.backbone_)
    baseline._mahalanobis_similarity = lambda representation, references: True
    current_window = torch.ones(1, 8, 3)
    baseline._collect_candidates(
        current_window, np.asarray([100.0], dtype=np.float64), threshold=50.0
    )

    assert observed
    assert torch.equal(observed[-1], current_window)

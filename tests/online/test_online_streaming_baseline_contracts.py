from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.baselines.online import (
    CANDIStreamingBaseline,
    IForestStreamingBaseline,
    KMeansADStreamingBaseline,
    M2N2StreamingBaseline,
    StumpyChannelABStreamingBaseline,
)
from src.models.simple_window_cnn_autoencoder import SimpleWindowCnnAutoencoder
from src.protocols.online_stream_range import select_online_stream_sequence


def _build_sequence(entity_id: str, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    length = 32
    channels = 3
    return {
        "x": rng.normal(size=(length, channels)).astype(np.float64),
        "point_labels": np.zeros(length, dtype=np.int64),
        "mask": np.ones((length, channels), dtype=np.float64),
        "timestamps": np.arange(length, dtype=np.int64),
        "meta": {
            "dataset_name": "smd",
            "entity_id": entity_id,
            "split": "test",
            "sequence_length": length,
            "source_sequence_length": length,
        },
    }


def _protocol_config() -> dict[str, Any]:
    return {
        "window_size": 8,
        "offline_tail_policy": "end_align",
        "offline_threshold_split": "clean_validation",
        "offline_threshold_quantile": 0.9,
        "online_window_stride": 1,
        "online_threshold_split": "clean_validation",
        "online_threshold_quantile": 0.9,
        "online_ewma_current_weight": 0.9,
        "online_ewma_previous_weight": 0.1,
        "test_label_usage": "metrics_only",
        "point_adjustment": False,
    }


def _write_redlamp_fixture(tmp_path: Path) -> Path:
    model = SimpleWindowCnnAutoencoder(
        input_dim=3,
        latent_dim=128,
        hidden_channels=64,
        kernel_size=3,
        num_layers=3,
        dropout=0.1,
    )
    checkpoint_path = tmp_path / "redlamp_best.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": 100}, checkpoint_path
    )
    return checkpoint_path


def test_online_streaming_baselines_calibrate_and_run(tmp_path: Path) -> None:
    train_sequence = _build_sequence("machine-1-6", seed=0)["x"]
    clean_validation_sequences = [_build_sequence("machine-1-6", seed=1)]
    test_sequence = _build_sequence("machine-1-6", seed=2)
    protocol_config = _protocol_config()
    checkpoint_path = _write_redlamp_fixture(tmp_path)

    baselines = [
        CANDIStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="reference_adapter_redlamp_encoder",
            seed=7,
            pretrained_encoder_checkpoint=checkpoint_path,
            candi_use_fpm=False,
        ),
        M2N2StreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="reference_adapter_redlamp_encoder",
            seed=7,
            pretrained_encoder_checkpoint=checkpoint_path,
        ),
        StumpyChannelABStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="main",
            seed=7,
        ),
        KMeansADStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="main",
            seed=7,
            n_clusters=2,
        ),
        IForestStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="main",
            seed=7,
            n_estimators=10,
        ),
    ]

    for baseline in baselines:
        expected_variant = baseline.online_variant
        calibration = baseline.calibrate(
            clean_validation_sequences=clean_validation_sequences,
            protocol_config=protocol_config,
            device="cpu",
        )
        metric_history, records = baseline.run_sequence(
            sequence=test_sequence,
            threshold_value=float(calibration["threshold_value"]),
            protocol_config=protocol_config,
            device="cpu",
        )
        expected_threshold_source = (
            "clean_validation_stride1_raw_window"
            if baseline.method_name in {"candi", "m2n2"}
            else "clean_validation_stride1_ewma"
        )
        assert calibration["threshold_source"] == expected_threshold_source
        if baseline.method_name in {"candi", "m2n2"}:
            assert calibration["threshold_artifact"]["checkpoint_sha256"]
        assert metric_history
        assert records
        assert records[0]["online_variant"] == expected_variant
        if baseline.method_name in {"candi", "m2n2"}:
            metadata = calibration["method_metadata"]
            assert metadata["encoder_family"] == "cnn_simple"
            assert metadata["encoder_dim"] == 128
            assert metadata["cnn_num_layers"] == 3
            assert metadata["cnn_kernel_size"] == 3
            assert metadata["cnn_hidden_channels"] == 64
        for metric in metric_history:
            assert "online/verification_buffer_size" in metric
            assert "online/ttl_buffer_size" not in metric
            assert "online/raw_point_score" in metric
            assert "online/prediction" in metric
            assert "online/did_update" in metric


def test_online_streaming_baselines_emit_entity_global_indices(tmp_path: Path) -> None:
    train_sequence = _build_sequence("machine-1-6", seed=3)["x"]
    clean_validation_sequences = [_build_sequence("machine-1-6", seed=4)]
    test_sequence = select_online_stream_sequence(
        _build_sequence("machine-1-6", seed=5),
        absolute_start_index=10,
        absolute_end_index=32,
    )
    protocol_config = _protocol_config()
    checkpoint_path = _write_redlamp_fixture(tmp_path)

    baselines = [
        CANDIStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            pretrained_encoder_checkpoint=checkpoint_path,
            candi_use_fpm=False,
        ),
        M2N2StreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            pretrained_encoder_checkpoint=checkpoint_path,
        ),
        StumpyChannelABStreamingBaseline(train_sequence=train_sequence, window_size=8),
        KMeansADStreamingBaseline(
            train_sequence=train_sequence, window_size=8, n_clusters=2
        ),
        IForestStreamingBaseline(
            train_sequence=train_sequence, window_size=8, n_estimators=10
        ),
    ]

    for baseline in baselines:
        calibration = baseline.calibrate(
            clean_validation_sequences=clean_validation_sequences,
            protocol_config=protocol_config,
            device="cpu",
        )
        _, records = baseline.run_sequence(
            sequence=test_sequence,
            threshold_value=float(calibration["threshold_value"]),
            protocol_config=protocol_config,
            device="cpu",
        )
        assert records[0]["point_index"] == 17
        assert records[0]["window_start_index"] == 10
        assert records[0]["window_end_index"] == 18

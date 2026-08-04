from __future__ import annotations

from pathlib import Path

import pytest

from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    load_threshold_artifact,
    write_threshold_artifact,
)
from src.engine.online_tta.checkpoint_resolution import resolve_threshold_artifact


def test_threshold_artifact_round_trips_json(tmp_path) -> None:
    artifact = build_threshold_artifact(
        method_name="THESIS",
        variant_name="O0-A0",
        entity_id="machine-1-6",
        seed=6,
        window_size=20,
        offline_point_threshold=1.5,
        online_ewma_point_threshold=2.5,
        quantile=0.99,
        ewma_current_weight=0.9,
        ewma_previous_weight=0.1,
        created_by="pytest",
        config_path="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        checkpoint_sha256="checkpoint-sha",
        input_window_threshold=3.0,
        latent_window_low_threshold=4.0,
        latent_window_high_threshold=5.0,
    )
    output_path = tmp_path / "thresholds.json"

    write_threshold_artifact(artifact, output_path)
    loaded = load_threshold_artifact(output_path)

    assert loaded == artifact
    assert loaded["schema_version"] == 4
    assert loaded["stochastic_inference"] is True
    assert loaded["monte_carlo_samples"] == 10
    assert loaded["variance_correction"] == 1
    assert loaded["return_mc_samples"] is False
    assert loaded["sample_retention_policy"] == "none"
    assert loaded["thresholds"]["offline_point"]["source_split"] == "clean_validation"
    assert loaded["thresholds"]["online_ewma_point"]["score_rule"] == (
        "stride1_causal_window_vector_ewma"
    )
    assert loaded["provenance"]["calibration_split"] == "clean_validation"
    assert loaded["provenance"]["score_reduction"] == "mean"
    assert loaded["provenance"]["resolved_config_sha256"] is None
    assert loaded["offline_stride"] == loaded["window_size"]
    assert loaded["online_stride"] == 1


def test_threshold_artifact_keeps_independent_window_thresholds() -> None:
    artifact = build_threshold_artifact(
        method_name="THESIS",
        variant_name="A2",
        entity_id="machine-1-6",
        seed=6,
        window_size=20,
        offline_point_threshold=1.0,
        online_ewma_point_threshold=2.0,
        input_window_threshold=3.0,
        latent_window_low_threshold=4.0,
        latent_window_high_threshold=5.0,
        quantile=0.99,
        ewma_current_weight=0.9,
        ewma_previous_weight=0.1,
        created_by="pytest",
        config_path="test.yaml",
        checkpoint_sha256="checkpoint-sha",
    )
    thresholds = artifact["thresholds"]
    assert thresholds["input_window"]["value"] == 3.0
    assert thresholds["latent_window_low"]["quantile"] == 0.95
    assert thresholds["latent_window_high"]["value"] == 5.0
    assert artifact["offline_point_threshold_nonoverlap"] == 1.0
    assert artifact["online_point_threshold_ewma"] == 2.0


def test_threshold_artifact_rejects_invalid_stride_contract() -> None:
    artifact = build_threshold_artifact(
        method_name="THESIS",
        variant_name="A2",
        entity_id="machine-1-6",
        seed=6,
        window_size=20,
        offline_point_threshold=1.0,
        online_ewma_point_threshold=2.0,
        quantile=0.99,
        ewma_current_weight=0.9,
        ewma_previous_weight=0.1,
        created_by="pytest",
        config_path="test.yaml",
        offline_stride=10,
        checkpoint_sha256="checkpoint-sha",
        input_window_threshold=3.0,
        latent_window_low_threshold=4.0,
        latent_window_high_threshold=5.0,
    )
    with pytest.raises(ValueError, match="offline_stride must match window_size"):
        write_threshold_artifact(artifact, Path("/tmp/threshold.json"))


def test_online_config_requires_an_explicit_threshold_artifact_path(tmp_path: Path) -> None:
    artifact_path = tmp_path / "thresholds.json"
    artifact_path.write_text("{}", encoding="utf-8")
    resolved = resolve_threshold_artifact(
        {"task": {"threshold_artifact_path": str(artifact_path)}}
    )

    assert resolved == artifact_path
    with pytest.raises(ValueError, match="threshold_artifact_path"):
        resolve_threshold_artifact({"task": {}})

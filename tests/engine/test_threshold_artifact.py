from __future__ import annotations

from pathlib import Path

import pytest

from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    load_threshold_artifact,
    write_threshold_artifact,
)


def test_threshold_artifact_preserves_checkpoint_and_config_provenance(
    tmp_path: Path,
) -> None:
    artifact = build_threshold_artifact(
        method_name="THESIS",
        variant_name="A2",
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
        resolved_config_sha256="config-sha",
        return_mc_samples=False,
        sample_retention_policy="retain_for_eda",
        input_window_threshold=3.0,
        latent_window_low_threshold=4.0,
        latent_window_high_threshold=5.0,
        point_score_c=1.0,
        point_score_tau=0.5,
    )
    output_path = tmp_path / "thresholds.json"
    write_threshold_artifact(artifact, output_path)
    loaded = load_threshold_artifact(output_path)

    assert loaded["checkpoint_sha256"] == "checkpoint-sha"
    assert loaded["resolved_config_sha256"] == "config-sha"
    assert loaded["sample_retention_policy"] == "retain_for_eda"
    assert loaded["provenance"]["checkpoint_sha256"] == "checkpoint-sha"
    assert loaded["provenance"]["resolved_config_sha256"] == "config-sha"
    assert loaded["point_score_transform"] == "shifted-and-scaled logistic sigmoid"
    assert loaded["point_score_c"] == 1.0
    assert loaded["point_score_tau"] == 0.5


def test_raw_threshold_artifact_uses_schema_five_and_identity_transform(
    tmp_path: Path,
) -> None:
    artifact = build_threshold_artifact(
        method_name="THESIS",
        variant_name="A2",
        entity_id="machine-1-6",
        seed=6,
        window_size=20,
        offline_point_threshold=1.5,
        online_ewma_point_threshold=2.5,
        quantile=0.99,
        ewma_current_weight=0.9,
        ewma_previous_weight=0.1,
        created_by="pytest",
        config_path="raw.yaml",
        checkpoint_sha256="checkpoint-sha",
        resolved_config_sha256="config-sha",
        score_space="raw_input",
        input_window_threshold=3.0,
        latent_window_low_threshold=4.0,
        latent_window_high_threshold=5.0,
    )

    assert artifact["schema_version"] == 5
    assert artifact["score_space"] == "raw_input"
    assert artifact["point_score_transform"] == "identity"
    assert artifact["point_score_definition"] == "raw_input_point_mse"
    assert artifact["window_score_definition"] == "raw_input_window_mse"
    output_path = tmp_path / "raw-thresholds.json"
    write_threshold_artifact(artifact, output_path)
    assert load_threshold_artifact(output_path) == artifact


def test_raw_threshold_artifact_rejects_sigmoid_identity() -> None:
    artifact = build_threshold_artifact(
        method_name="THESIS",
        variant_name="A2",
        entity_id="machine-1-6",
        seed=6,
        window_size=20,
        offline_point_threshold=1.5,
        online_ewma_point_threshold=2.5,
        quantile=0.99,
        ewma_current_weight=0.9,
        ewma_previous_weight=0.1,
        created_by="pytest",
        config_path="raw.yaml",
        checkpoint_sha256="checkpoint-sha",
        score_space="raw_input",
        input_window_threshold=3.0,
        latent_window_low_threshold=4.0,
        latent_window_high_threshold=5.0,
    )
    artifact["point_score_transform"] = "shifted-and-scaled logistic sigmoid"

    with pytest.raises(ValueError, match="point_score_transform"):
        write_threshold_artifact(artifact, Path("/tmp/raw-thresholds.json"))

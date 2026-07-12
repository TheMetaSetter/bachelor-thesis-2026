from __future__ import annotations

from pathlib import Path

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
    )
    output_path = tmp_path / "thresholds.json"
    write_threshold_artifact(artifact, output_path)
    loaded = load_threshold_artifact(output_path)

    assert loaded["checkpoint_sha256"] == "checkpoint-sha"
    assert loaded["resolved_config_sha256"] == "config-sha"
    assert loaded["sample_retention_policy"] == "retain_for_eda"
    assert loaded["provenance"]["checkpoint_sha256"] == "checkpoint-sha"
    assert loaded["provenance"]["resolved_config_sha256"] == "config-sha"

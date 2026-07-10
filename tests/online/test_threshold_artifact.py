from __future__ import annotations

from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    load_threshold_artifact,
    write_threshold_artifact,
)


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
    )
    output_path = tmp_path / "thresholds.json"

    write_threshold_artifact(artifact, output_path)
    loaded = load_threshold_artifact(output_path)

    assert loaded == artifact
    assert loaded["thresholds"]["offline_point"]["source_split"] == "clean_validation"
    assert loaded["thresholds"]["online_ewma_point"]["score_rule"] == (
        "stride1_causal_endpoint_ewma"
    )

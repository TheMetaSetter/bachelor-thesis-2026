from __future__ import annotations

import numpy as np
import pytest

from scripts.ops.recalibrate_thesis_threshold_artifacts_v4 import (
    StageBInventoryEntry,
    _validate_v3_identity,
    build_v4_threshold_artifact,
    discover_stage_b_inventory,
    preflight_inventory,
)
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    validate_threshold_artifact,
    write_threshold_artifact,
)


def _v3_artifact() -> dict:
    artifact = build_threshold_artifact(
        method_name="M2N2",
        variant_name="O0",
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
    )
    artifact["method_name"] = "THESIS"
    artifact["provenance"]["threshold_method"] = "THESIS"
    return artifact


def test_build_v4_artifact_recalibrates_vector_ewma_and_triage_thresholds() -> None:
    artifact_v4, audit = build_v4_threshold_artifact(
        artifact_v3=_v3_artifact(),
        checkpoint_sha256="checkpoint-sha",
        calibration_scores={
            "ewma": [0.0, 1.0, 2.0, 3.0],
            "input_window": [1.0, 2.0, 3.0, 4.0],
            "latent_window": [2.0, 4.0, 6.0, 8.0],
        },
        protocol_config={
            "offline_threshold_quantile": 0.99,
            "online_threshold_quantile": 0.99,
            "online_ewma_current_weight": 0.9,
            "online_ewma_previous_weight": 0.1,
            "B_window_quantile": 0.99,
            "A_low_quantile": 0.75,
            "A_high_quantile": 0.99,
        },
        experiment_config_path=__file__,
    )

    validate_threshold_artifact(artifact_v4)
    thresholds = artifact_v4["thresholds"]
    assert artifact_v4["schema_version"] == 4
    assert artifact_v4["checkpoint_sha256"] == "checkpoint-sha"
    assert thresholds["online_ewma_point"]["score_rule"] == (
        "stride1_causal_window_vector_ewma"
    )
    assert thresholds["online_ewma_point"]["value"] == pytest.approx(
        np.quantile([0.0, 1.0, 2.0, 3.0], 0.99)
    )
    assert thresholds["input_window"]["value"] == pytest.approx(
        np.quantile([1.0, 2.0, 3.0, 4.0], 0.99)
    )
    assert thresholds["latent_window_low"]["quantile"] == 0.75
    assert thresholds["latent_window_low"]["value"] == pytest.approx(
        np.quantile([2.0, 4.0, 6.0, 8.0], 0.75)
    )
    assert audit["clean_validation_window_count"] == 4


def test_v3_identity_rejects_wrong_offline_variant(tmp_path) -> None:
    entry = StageBInventoryEntry(
        experiment_config_path=tmp_path / "config.yaml",
        offline_variant="O1",
        entity_id="machine-1-6",
        seed=6,
        threshold_artifact_v3_path=tmp_path / "thresholds.json",
        stage_b_best_checkpoint_path=tmp_path / "best.pt",
        threshold_artifact_v4_path=tmp_path / "thresholds_v4_recalibrated.json",
        audit_path=tmp_path / "audit.json",
    )

    with pytest.raises(ValueError, match="variant_name"):
        _validate_v3_identity(_v3_artifact(), entry)


def test_discovery_requires_the_18_official_thesis_main_configs(tmp_path) -> None:
    for offline_variant in ("O0", "O1"):
        for entity_id in ("machine_1_6", "machine_3_4", "machine_3_9"):
            for seed in (6, 8, 36):
                config_path = tmp_path / (
                    f"smd__thesis__offline__{offline_variant}__{entity_id}"
                    f"__w20__seed{seed}__main.yaml"
                )
                config_path.write_text(
                    f"output_dir: outputs/benchmark/smd/thesis/{offline_variant}/"
                    f"{entity_id}/seed{seed}\n",
                    encoding="utf-8",
                )

    entries = discover_stage_b_inventory(tmp_path, "thresholds_v4_recalibrated.json")

    assert len(entries) == 18
    assert entries[0].threshold_artifact_v3_path.name == "thresholds.json"
    assert entries[0].threshold_artifact_v4_path.name == "thresholds_v4_recalibrated.json"


def test_preflight_refuses_to_overwrite_an_existing_v4_artifact(tmp_path) -> None:
    entry = StageBInventoryEntry(
        experiment_config_path=tmp_path / "config.yaml",
        offline_variant="O0",
        entity_id="machine-1-6",
        seed=6,
        threshold_artifact_v3_path=tmp_path / "thresholds.json",
        stage_b_best_checkpoint_path=tmp_path / "best.pt",
        threshold_artifact_v4_path=tmp_path / "thresholds_v4_recalibrated.json",
        audit_path=tmp_path / "audit.json",
    )
    write_threshold_artifact(_v3_artifact(), entry.threshold_artifact_v3_path)
    entry.stage_b_best_checkpoint_path.write_bytes(b"checkpoint")
    entry.threshold_artifact_v4_path.write_text("already exists\n", encoding="utf-8")

    failures = preflight_inventory([entry])

    assert failures[0]["reason"] == "V4 output already exists; refusing to overwrite"

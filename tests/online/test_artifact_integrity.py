from __future__ import annotations

from src.core.artifact_integrity import (
    build_artifact_manifest,
    build_retention_bundle_manifest,
    verify_artifact_manifest,
    verify_retention_bundle_manifest,
    write_artifact_manifest,
    write_retention_bundle_manifest,
)


def test_artifact_manifest_verifies_identity_and_detects_content_change(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "online_final.pt"
    records_path = tmp_path / "online_records.json"
    checkpoint_path.write_bytes(b"checkpoint-v1")
    records_path.write_text('{"records": []}\n', encoding="utf-8")
    identity = {"entity_id": "machine-1-6", "online_variant": "A2"}

    manifest = build_artifact_manifest(
        {"checkpoint": checkpoint_path, "records": records_path},
        identity,
        provenance={"created_by": "pytest", "config_path": "test.yaml"},
    )
    manifest_path = write_artifact_manifest(tmp_path / "manifest.json", manifest)

    assert manifest_path.is_file()
    assert verify_artifact_manifest(manifest, expected_identity=identity)
    assert verify_artifact_manifest(
        manifest,
        expected_identity=identity,
        expected_provenance={"created_by": "pytest", "config_path": "test.yaml"},
    )
    assert not verify_artifact_manifest(
        manifest, expected_identity={"entity_id": "machine-1-6", "online_variant": "A0"}
    )

    records_path.write_text('{"records": ["changed"]}\n', encoding="utf-8")
    assert not verify_artifact_manifest(manifest, expected_identity=identity)


def test_retention_bundle_manifest_records_policy_and_hashes(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"bundle": "retention"}\n', encoding="utf-8")
    records_path = tmp_path / "records.json"
    records_path.write_text('{"records": [1]}\n', encoding="utf-8")
    manifest = build_retention_bundle_manifest(
        {"summary": summary_path, "records": records_path},
        {"entity_id": "machine-1-6", "online_variant": "A2"},
        provenance={
            "checkpoint_sha256": "abc123",
            "resolved_config_sha256": "def456",
        },
        retention_policy="retain_for_eda",
        compression="none",
        export_scope="entity",
    )
    manifest_path = write_retention_bundle_manifest(
        tmp_path / "retention_manifest.json",
        manifest,
    )

    assert manifest_path.is_file()
    assert manifest["bundle_type"] == "retention_bundle"
    assert manifest["retention_policy"] == "retain_for_eda"
    assert manifest["compression"] == "none"
    assert verify_retention_bundle_manifest(
        manifest,
        expected_identity={"entity_id": "machine-1-6", "online_variant": "A2"},
        expected_provenance={
            "checkpoint_sha256": "abc123",
            "resolved_config_sha256": "def456",
        },
        retention_policy="retain_for_eda",
        compression="none",
        export_scope="entity",
    )

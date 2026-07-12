from __future__ import annotations

from src.core.artifact_integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
    write_artifact_manifest,
)


def test_artifact_manifest_verifies_identity_and_detects_content_change(tmp_path) -> None:
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

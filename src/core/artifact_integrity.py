"""Checksum manifests for benchmark artifacts that must survive resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one existing regular file."""
    resolved_path = Path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"artifact file does not exist: {resolved_path}")
    digest = hashlib.sha256()
    with resolved_path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(
    artifact_paths: dict[str, str | Path],
    identity: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic checksum manifest for named artifact files."""
    if not isinstance(identity, dict) or not identity:
        raise ValueError("manifest identity must be a non-empty mapping")
    artifacts = {
        name: {"path": str(Path(path)), "sha256": sha256_file(path)}
        for name, path in sorted(artifact_paths.items())
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "identity": dict(identity),
        "artifacts": artifacts,
    }
    if provenance is not None:
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("manifest provenance must be a non-empty mapping")
        manifest["provenance"] = dict(provenance)
    return manifest


def build_retention_bundle_manifest(
    artifact_paths: dict[str, str | Path],
    identity: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    *,
    retention_policy: str = "retain_for_eda",
    compression: str = "none",
    export_scope: str = "entity",
) -> dict[str, Any]:
    """Build a manifest for a retention bundle used by later EDA."""
    if retention_policy not in {"retain_for_eda", "summary_only"}:
        raise ValueError(
            "retention_policy must be one of: retain_for_eda, summary_only"
        )
    if compression not in {"none", "gzip"}:
        raise ValueError("compression must be one of: none, gzip")
    if export_scope not in {"entity", "run"}:
        raise ValueError("export_scope must be one of: entity, run")
    manifest = build_artifact_manifest(artifact_paths, identity, provenance)
    manifest["bundle_type"] = "retention_bundle"
    manifest["retention_policy"] = retention_policy
    manifest["compression"] = compression
    manifest["export_scope"] = export_scope
    manifest["bundle_schema_version"] = 1
    return manifest


def verify_artifact_manifest(
    manifest: dict[str, Any],
    expected_identity: dict[str, Any] | None = None,
    expected_provenance: dict[str, Any] | None = None,
) -> bool:
    """Return whether identity and all named artifact checksums still match."""
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        return False
    if expected_provenance is not None and manifest.get("provenance") != expected_provenance:
        return False
    if manifest.get("schema_version") != 1:
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            return False
        try:
            if not isinstance(artifact.get("path"), str) or not isinstance(
                artifact.get("sha256"), str
            ):
                return False
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                return False
        except (FileNotFoundError, KeyError):
            return False
    return True


def verify_retention_bundle_manifest(
    manifest: dict[str, Any],
    expected_identity: dict[str, Any] | None = None,
    expected_provenance: dict[str, Any] | None = None,
    *,
    retention_policy: str | None = None,
    compression: str | None = None,
    export_scope: str | None = None,
) -> bool:
    """Return whether a retention bundle manifest still matches its files."""
    if manifest.get("bundle_type") != "retention_bundle":
        return False
    if int(manifest.get("bundle_schema_version", 0)) != 1:
        return False
    if retention_policy is not None and manifest.get("retention_policy") != retention_policy:
        return False
    if compression is not None and manifest.get("compression") != compression:
        return False
    if export_scope is not None and manifest.get("export_scope") != export_scope:
        return False
    return verify_artifact_manifest(
        manifest,
        expected_identity=expected_identity,
        expected_provenance=expected_provenance,
    )


def write_artifact_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write one checksum manifest without mutating its artifact entries."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def write_retention_bundle_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write one retention-bundle manifest without mutating its entries."""
    return write_artifact_manifest(path, manifest)

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
    artifacts = {
        name: {"path": str(Path(path)), "sha256": sha256_file(path)}
        for name, path in sorted(artifact_paths.items())
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "identity": identity,
        "artifacts": artifacts,
    }
    if provenance is not None:
        manifest["provenance"] = provenance
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
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            return False
        try:
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                return False
        except (FileNotFoundError, KeyError):
            return False
    return True


def write_artifact_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write one checksum manifest without mutating its artifact entries."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path

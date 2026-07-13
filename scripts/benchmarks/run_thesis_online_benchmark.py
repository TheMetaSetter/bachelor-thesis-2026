from __future__ import annotations

"""THESIS online benchmark wrapper.

₍^. .^₎⟆ Online benchmark boundary

offline checkpoint + protocol thresholds
  -> A0/A1/A2 online runner
  -> normalized online records
  -> benchmark report

The first implementation slice exposes the report boundary for A0. The full
streaming scorer/adaptation policy can then grow behind this stable contract.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.core.config import load_experiment_config
from src.core.artifact_integrity import (
    build_artifact_manifest,
    build_retention_bundle_manifest,
    sha256_file,
    verify_artifact_manifest,
    write_artifact_manifest,
    write_retention_bundle_manifest,
)
from src.engine.online_tta.checkpoint_resolution import resolve_stage_b_checkpoint
from src.engine.online_tta.online_engine import run_thesis_online_tta_experiment
from src.engine.checkpoint import CheckpointManager
from src.protocols.smd_benchmark_protocol import validate_protocol_config


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_yaml_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_experiment_config_with_compatibility(
    experiment_config_path: str,
) -> dict[str, Any]:
    raw_config = _load_yaml_config(experiment_config_path)
    has_config_references = all(
        key in raw_config
        for key in ("data_config_path", "model_config_path", "task_config_path")
    )
    if not has_config_references:
        return raw_config
    try:
        return load_experiment_config(experiment_config_path)
    except (FileNotFoundError, ValueError):
        return raw_config


def _normalize_online_records(
    records: list[dict[str, Any]],
    online_variant: str,
) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        normalized_record = dict(record)
        normalized_record.setdefault("online_variant", online_variant)
        if online_variant == "A0":
            normalized_record["did_update"] = False
        else:
            normalized_record.setdefault("did_update", True)
        normalized_records.append(normalized_record)
    return normalized_records


def _write_report(
    output_dir: Path,
    online_variant: str,
    report: dict[str, Any],
) -> Path:
    report_dir = output_dir / "benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"thesis_online_{online_variant}_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return report_path


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(path)


def _resolve_retention_policy(experiment_config: dict[str, Any]) -> str:
    evaluation_config = dict(experiment_config.get("evaluation", {}))
    return str(evaluation_config.get("retention_policy", "retain_for_eda"))


def _load_runtime_state_snapshot(checkpoint_path: str | None) -> dict[str, Any] | None:
    if not checkpoint_path:
        return None
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        return None
    checkpoint_payload = torch.load(checkpoint_file, map_location="cpu")
    extra_state = checkpoint_payload.get("extra_state", {})
    runtime_state = extra_state.get("online_runtime_state")
    if runtime_state is not None:
        return runtime_state
    return {
        "online_variant": extra_state.get("online_variant"),
        "threshold_artifact": extra_state.get("threshold_artifact"),
        "stream_cursor": extra_state.get("stream_cursor"),
        "previous_ewma_score": extra_state.get("previous_ewma_score"),
        "signature_history": extra_state.get("signature_history", []),
        "recurrent_signatures": extra_state.get("recurrent_signatures", []),
        "verification_entries": extra_state.get("verification_buffer_entries", []),
        "verification_history": extra_state.get("verification_history", []),
        "hard_old_intervals": extra_state.get("hard_old_guard_intervals", []),
    }


def _export_online_retention_bundle(
    *,
    output_dir: Path,
    experiment_config: dict[str, Any],
    online_outputs: dict[str, Any],
    online_variant: str,
    retention_policy: str,
) -> dict[str, str]:
    threshold_artifact = dict(online_outputs["threshold_artifact"])
    entity_id = str(threshold_artifact["entity_id"])
    retention_root = output_dir / "retention" / entity_id / online_variant
    retention_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = online_outputs.get("final_checkpoint_path")
    checkpoint_sha256 = None
    if checkpoint_path and Path(checkpoint_path).is_file():
        checkpoint_sha256 = sha256_file(checkpoint_path)
    resolved_config_sha256 = CheckpointManager._stable_json_digest(experiment_config)
    runtime_state = _load_runtime_state_snapshot(checkpoint_path)
    summary_payload = {
        "bundle_type": "online_thesis_retention",
        "bundle_schema_version": 1,
        "entity_id": entity_id,
        "online_variant": online_variant,
        "retention_policy": retention_policy,
        "compression": "none",
        "experiment_name": experiment_config.get("experiment_name"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "threshold_artifact_sha256": CheckpointManager._stable_json_digest(
            threshold_artifact
        ),
        "metric_history_length": len(online_outputs.get("metric_history", [])),
        "record_length": len(online_outputs.get("records", [])),
        "runtime_state_present": runtime_state is not None,
        "inspection_ready": retention_policy == "retain_for_eda",
    }
    bundle_paths: dict[str, str] = {
        "summary": _write_json(retention_root / "retention_summary.json", summary_payload)
    }
    if retention_policy == "retain_for_eda":
        bundle_paths["metrics"] = _write_json(
            retention_root / "online_metrics.json",
            online_outputs.get("metric_history", []),
        )
        bundle_paths["records"] = _write_json(
            retention_root / "online_records.json",
            online_outputs.get("records", []),
        )
        bundle_paths["threshold_artifact"] = _write_json(
            retention_root / "threshold_artifact.json",
            threshold_artifact,
        )
        if runtime_state is not None:
            bundle_paths["runtime_state"] = _write_json(
                retention_root / "online_runtime_state.json",
                runtime_state,
            )
    manifest = build_retention_bundle_manifest(
        bundle_paths,
        identity={
            "entity_id": entity_id,
            "experiment_name": str(experiment_config.get("experiment_name")),
            "online_variant": online_variant,
        },
        provenance={
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "threshold_artifact_sha256": summary_payload["threshold_artifact_sha256"],
        },
        retention_policy=retention_policy,
        compression="none",
        export_scope="entity",
    )
    manifest_path = write_retention_bundle_manifest(
        retention_root / "retention_bundle_manifest.json",
        manifest,
    )
    bundle_paths["manifest"] = str(manifest_path)
    return bundle_paths


def run_thesis_online_benchmark(
    *,
    experiment_config_path: str,
    protocol_config_path: str,
    online_variant: str,
    dry_run: bool,
) -> dict[str, Any]:
    if online_variant not in {"A0", "A1", "A2"}:
        raise ValueError("online_variant must be one of: A0, A1, A2")

    experiment_config = _load_experiment_config_with_compatibility(
        experiment_config_path
    )
    protocol_config = _load_yaml_config(protocol_config_path)
    retention_policy = _resolve_retention_policy(experiment_config)
    validate_protocol_config(protocol_config)
    resolved_checkpoint_path = resolve_stage_b_checkpoint(experiment_config)
    experiment_config["task"]["reference_checkpoint_path"] = str(
        resolved_checkpoint_path
    )

    online_outputs = run_thesis_online_tta_experiment(
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        dry_run=dry_run,
    )
    online_outputs = dict(online_outputs)
    online_outputs["records"] = _normalize_online_records(
        list(online_outputs.get("records", [])),
        online_variant,
    )

    report = {
        "benchmark_status": "dry_run" if dry_run else "completed",
        "created_at_utc": _utc_now_iso(),
        "experiment_config_path": experiment_config_path,
        "online_variant": online_variant,
        "protocol_config_path": protocol_config_path,
        "protocol": protocol_config,
        "online_execution": online_outputs,
        "retention_policy": retention_policy,
    }
    report_path = _write_report(
        Path(str(experiment_config["output_dir"])),
        online_variant,
        report,
    )
    retention_artifact_paths: dict[str, str] = {}
    if not dry_run:
        retention_artifact_paths = _export_online_retention_bundle(
            output_dir=Path(str(experiment_config["output_dir"])),
            experiment_config=experiment_config,
            online_outputs=online_outputs,
            online_variant=online_variant,
            retention_policy=retention_policy,
        )
    report_identity = {
        "experiment_name": str(experiment_config["experiment_name"]),
        "online_variant": online_variant,
        "protocol_config_path": str(protocol_config_path),
    }
    report_manifest = build_artifact_manifest(
        {"benchmark_report": report_path}, report_identity
    )
    report_manifest_path = write_artifact_manifest(
        report_path.with_name(f"{report_path.stem}_integrity_manifest.json"),
        report_manifest,
    )
    report["report_path"] = str(report_path)
    report["retention_artifact_paths"] = retention_artifact_paths
    report["report_artifact_integrity_status"] = (
        "verified"
        if verify_artifact_manifest(report_manifest, report_identity)
        else "failed"
    )
    report["report_artifact_manifest_path"] = str(report_manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
    )
    parser.add_argument("--online-variant", choices=["A0", "A1", "A2"], default="A0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_thesis_online_benchmark(
        experiment_config_path=args.experiment_config,
        protocol_config_path=args.protocol_config,
        online_variant=args.online_variant,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

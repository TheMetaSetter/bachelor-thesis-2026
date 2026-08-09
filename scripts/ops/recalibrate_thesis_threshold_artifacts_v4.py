"""Create safe THESIS V4 threshold artifacts from the 18 Stage-B checkpoints.

The script never overwrites a V3 artifact. It recreates clean-validation
windows with the checkpoint scaler, then scores them through the frozen A0
online path that defines the latent-window score used at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks.run_thesis_offline_benchmark import (
    validate_protocol_config,
)
from scripts.ops.threshold_artifact_v4_online_scoring import (
    StageBInventoryEntry,
    entry_as_report_value,
    load_a0_scoring_config,
)
from src.core.artifact_integrity import sha256_file
from src.core.config import load_experiment_config
from src.core.registry import build_dataset
from src.core.runtime_components import register_online_runtime_components
from src.core.seed import seed_everything
from src.data.loaders import rebuild_dataset_bundle_with_scaler_state
from src.engine.online_tta.online_calibration import (
    collect_nonoverlap_offline_scores,
    collect_stride1_online_scores,
)
from src.engine.online_tta.online_engine_shared import (
    _build_model_from_experiment_config,
)
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    load_threshold_artifact,
    write_threshold_artifact,
)
from src.protocols.point_score_calibration import (
    PointScoreCalibration,
    fit_mad_logistic_calibration,
    transform_point_scores,
)


DEFAULT_EXPERIMENT_CONFIG_DIRECTORY = Path(
    "configs/experiment/offline_benchmark/thesis"
)
DEFAULT_PROTOCOL_CONFIG_PATH = Path(
    "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
)
DEFAULT_V4_FILENAME = "thresholds_v4_recalibrated.json"
DEFAULT_AUDIT_FILENAME = "thresholds_v4_recalibration_audit.json"
DEFAULT_REPORT_PATH = Path(
    "outputs/benchmark/smd/thesis/threshold_artifact_v4_recalibration_report.json"
)
EXPECTED_OFFLINE_VARIANTS = ("O0", "O1")
EXPECTED_ENTITY_IDS = ("machine-1-6", "machine-3-4", "machine-3-9")
EXPECTED_SEEDS = (6, 8, 36)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML mapping is required: {path}")
    return payload


def _identity_from_config_filename(config_path: Path) -> tuple[str, str, int]:
    name_parts = config_path.stem.split("__")
    if len(name_parts) != 8 or name_parts[:3] != ["smd", "thesis", "offline"]:
        raise ValueError(
            f"Unexpected THESIS offline config filename: {config_path.name}"
        )
    offline_variant = name_parts[3]
    entity_id = name_parts[4].replace("_", "-")
    seed_text = name_parts[6]
    if not seed_text.startswith("seed"):
        raise ValueError(f"Config filename has no seed: {config_path.name}")
    return offline_variant, entity_id, int(seed_text.removeprefix("seed"))


def _expected_identities() -> set[tuple[str, str, int]]:
    return {
        (offline_variant, entity_id, seed)
        for offline_variant in EXPECTED_OFFLINE_VARIANTS
        for entity_id in EXPECTED_ENTITY_IDS
        for seed in EXPECTED_SEEDS
    }


def discover_stage_b_inventory(
    experiment_config_directory: Path,
    v4_filename: str,
) -> list[StageBInventoryEntry]:
    config_paths = sorted(experiment_config_directory.glob("*__main.yaml"))
    entries: list[StageBInventoryEntry] = []
    for config_path in config_paths:
        offline_variant, entity_id, seed = _identity_from_config_filename(config_path)
        root_config = _load_yaml_mapping(config_path)
        output_dir = Path(str(root_config["output_dir"]))
        threshold_dir = output_dir / "thresholds"
        entries.append(
            StageBInventoryEntry(
                experiment_config_path=config_path,
                offline_variant=offline_variant,
                entity_id=entity_id,
                seed=seed,
                threshold_artifact_v3_path=threshold_dir / "thresholds.json",
                stage_b_best_checkpoint_path=(
                    output_dir
                    / "two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
                ),
                threshold_artifact_v4_path=threshold_dir / v4_filename,
                audit_path=threshold_dir / DEFAULT_AUDIT_FILENAME,
            )
        )
    found_identities = {
        (entry.offline_variant, entry.entity_id, entry.seed) for entry in entries
    }
    if found_identities != _expected_identities():
        raise ValueError(
            "THESIS main-config inventory must contain exactly O0/O1, "
            "machine-1-6/machine-3-4/machine-3-9, and seeds 6/8/36"
        )
    return entries


def _validate_v3_identity(
    artifact_v3: dict[str, Any], entry: StageBInventoryEntry
) -> None:
    required_identity = {
        "method_name": "THESIS",
        "entity_id": entry.entity_id,
        "seed": entry.seed,
    }
    if int(artifact_v3["schema_version"]) != 3:
        raise ValueError("threshold artifact is not schema version 3")
    for field_name, expected_value in required_identity.items():
        if artifact_v3[field_name] != expected_value:
            raise ValueError(
                f"threshold artifact {field_name} does not match inventory: "
                f"{artifact_v3[field_name]!r} != {expected_value!r}"
            )
    artifact_variant = str(artifact_v3["variant_name"])
    if artifact_variant == entry.offline_variant:
        return
    if artifact_variant == "O0" and entry.offline_variant == "O1":
        return
    raise ValueError(
        "threshold artifact variant_name has no approved recovery rule: "
        f"{artifact_variant!r} != {entry.offline_variant!r}"
    )


def _validate_checkpoint_identity(
    checkpoint_payload: dict[str, Any],
    entry: StageBInventoryEntry,
    window_size: int,
) -> None:
    entity_token = entry.entity_id.replace("-", "_")
    expected_name = (
        f"smd__thesis__offline__{entry.offline_variant}__{entity_token}"
        f"__w{window_size}__seed{entry.seed}__main__stage_b_fusion_finetuning"
    )
    expected_output_dir = (
        "outputs/benchmark/smd/thesis/"
        f"{entry.offline_variant}/{entity_token}/seed{entry.seed}/two_stage/"
        "stage_b_fusion_finetuning"
    )
    checkpoint_config = checkpoint_payload.get("config")
    checkpoint_metadata = checkpoint_payload.get("checkpoint_metadata")
    if not isinstance(checkpoint_config, dict) or not isinstance(
        checkpoint_metadata, dict
    ):
        raise ValueError("Stage-B checkpoint has no config or checkpoint_metadata")
    if checkpoint_config.get("experiment_name") != expected_name:
        raise ValueError(
            "Stage-B checkpoint config experiment_name does not match inventory"
        )
    if checkpoint_metadata.get("experiment_name") != expected_name:
        raise ValueError(
            "Stage-B checkpoint metadata experiment_name does not match inventory"
        )
    if checkpoint_config.get("output_dir") != expected_output_dir:
        raise ValueError(
            "Stage-B checkpoint config output_dir does not match inventory"
        )


def preflight_inventory(entries: list[StageBInventoryEntry]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for entry in entries:
        try:
            if not entry.stage_b_best_checkpoint_path.is_file():
                raise FileNotFoundError("Stage-B best checkpoint is missing")
            artifact_v3 = load_threshold_artifact(entry.threshold_artifact_v3_path)
            _validate_v3_identity(artifact_v3, entry)
            if entry.threshold_artifact_v4_path.exists():
                raise FileExistsError("V4 output already exists; refusing to overwrite")
        except (FileNotFoundError, ValueError, FileExistsError) as error:
            failures.append(
                {
                    "experiment_config_path": str(entry.experiment_config_path),
                    "reason": str(error),
                }
            )
    return failures


def _require_quantile(protocol_config: dict[str, Any], field_name: str) -> float:
    value = float(protocol_config[field_name])
    if not 0.0 < value < 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def _finite_quantile(values: list[float], quantile: float, name: str) -> float:
    score_array = np.asarray(values, dtype=float)
    if score_array.size == 0 or not np.isfinite(score_array).all():
        raise ValueError(f"{name} must contain finite calibration scores")
    return float(np.quantile(score_array, quantile))


def _collect_clean_validation_scores(
    entry: StageBInventoryEntry,
    protocol_config: dict[str, Any],
) -> dict[str, list[float]]:
    experiment_config = load_experiment_config(entry.experiment_config_path)
    data_config = experiment_config["data"]
    window_size = int(protocol_config["window_size"])
    if int(data_config["window_size"]) != window_size:
        raise ValueError("data window_size does not match protocol window_size")

    checkpoint_payload = torch.load(
        entry.stage_b_best_checkpoint_path, map_location="cpu"
    )
    _validate_checkpoint_identity(
        checkpoint_payload,
        entry,
        window_size=window_size,
    )
    register_online_runtime_components()
    online_config = load_a0_scoring_config(entry, window_size)
    model = _build_model_from_experiment_config(online_config)
    data_bundle = build_dataset(data_config["dataset_name"], data_config)
    scaler_state = checkpoint_payload.get("scaler_state_dict")
    if scaler_state is not None and "raw_sequences" in data_bundle:
        data_bundle = rebuild_dataset_bundle_with_scaler_state(
            data_bundle=data_bundle,
            data_config=data_config,
            scaler_state_dict=scaler_state,
        )

    device = str(online_config["device"])
    model.to(device)
    seed_everything(entry.seed)
    clean_validation_sequences = data_bundle.get("scaled_sequences", {}).get("val", [])
    offline_raw_scores = collect_nonoverlap_offline_scores(
        model=model,
        clean_validation_sequences=clean_validation_sequences,
        window_size=window_size,
        device=device,
    )
    calibration = fit_mad_logistic_calibration(offline_raw_scores)
    model.set_point_score_calibration(calibration)
    online_scores = collect_stride1_online_scores(
        model=model,
        clean_validation_sequences=clean_validation_sequences,
        window_size=window_size,
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device=device,
        current_weight=float(protocol_config["online_ewma_current_weight"]),
        previous_weight=float(protocol_config["online_ewma_previous_weight"]),
    )
    online_scores["offline_raw"] = offline_raw_scores
    online_scores["calibration"] = calibration
    online_scores["offline"] = transform_point_scores(
        np.asarray(offline_raw_scores, dtype=float), calibration
    ).tolist()
    return online_scores


def _calibrate_threshold_values(
    calibration_scores: dict[str, list[float]], protocol_config: dict[str, Any]
) -> dict[str, float]:
    offline_quantile = _require_quantile(protocol_config, "offline_threshold_quantile")
    input_quantile = _require_quantile(protocol_config, "B_window_quantile")
    latent_low_quantile = _require_quantile(protocol_config, "A_low_quantile")
    latent_high_quantile = _require_quantile(protocol_config, "A_high_quantile")
    online_ewma_quantile = _require_quantile(
        protocol_config, "online_threshold_quantile"
    )
    input_threshold = _finite_quantile(
        calibration_scores["input_window"], input_quantile, "input_window"
    )
    latent_low_threshold = _finite_quantile(
        calibration_scores["latent_window"], latent_low_quantile, "latent_window"
    )
    latent_high_threshold = _finite_quantile(
        calibration_scores["latent_window"], latent_high_quantile, "latent_window"
    )
    online_ewma_threshold = _finite_quantile(
        calibration_scores["ewma"], online_ewma_quantile, "online_ewma"
    )
    offline_threshold = _finite_quantile(
        calibration_scores["offline"], offline_quantile, "offline_point"
    )
    return {
        "offline_point_quantile": offline_quantile,
        "offline_point_threshold": offline_threshold,
        "input_window_quantile": input_quantile,
        "input_window_threshold": input_threshold,
        "latent_window_low_quantile": latent_low_quantile,
        "latent_window_low_threshold": latent_low_threshold,
        "latent_window_high_quantile": latent_high_quantile,
        "latent_window_high_threshold": latent_high_threshold,
        "online_ewma_point_quantile": online_ewma_quantile,
        "online_ewma_point_threshold": online_ewma_threshold,
    }


def _v4_artifact_fields(
    artifact_v3: dict[str, Any],
    checkpoint_sha256: str,
    experiment_config_path: Path,
    protocol_config: dict[str, Any],
    entry: StageBInventoryEntry,
) -> dict[str, Any]:
    return {
        "method_name": "THESIS",
        "variant_name": entry.offline_variant,
        "entity_id": entry.entity_id,
        "seed": entry.seed,
        "window_size": int(artifact_v3["window_size"]),
        "offline_point_threshold": float(
            artifact_v3["offline_point_threshold_nonoverlap"]
        ),
        "quantile": float(artifact_v3["thresholds"]["offline_point"]["quantile"]),
        "ewma_current_weight": float(protocol_config["online_ewma_current_weight"]),
        "ewma_previous_weight": float(protocol_config["online_ewma_previous_weight"]),
        "created_by": "scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py",
        "config_path": str(experiment_config_path),
        "calibration_split": str(artifact_v3["calibration_split"]),
        "stochastic_inference": bool(artifact_v3["stochastic_inference"]),
        "monte_carlo_samples": int(artifact_v3["monte_carlo_samples"]),
        "continuous_temperature": float(artifact_v3["continuous_temperature"]),
        "discrete_temperature": float(artifact_v3["discrete_temperature"]),
        "score_reduction": str(artifact_v3["score_reduction"]),
        "variance_correction": artifact_v3["variance_correction"],
        "numeric_precision": str(artifact_v3["numeric_precision"]),
        "return_mc_samples": bool(artifact_v3["return_mc_samples"]),
        "sample_retention_policy": str(artifact_v3["sample_retention_policy"]),
        "offline_stride": int(artifact_v3["offline_stride"]),
        "online_stride": int(artifact_v3["online_stride"]),
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": artifact_v3.get("resolved_config_sha256"),
    }


def build_v4_threshold_artifact(
    *,
    artifact_v3: dict[str, Any],
    checkpoint_sha256: str,
    calibration_scores: dict[str, list[float]],
    protocol_config: dict[str, Any],
    experiment_config_path: Path,
    entry: StageBInventoryEntry,
) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold_values = _calibrate_threshold_values(calibration_scores, protocol_config)
    calibration = calibration_scores.get("calibration")
    if not isinstance(calibration, PointScoreCalibration):
        raise ValueError("recalibration requires fitted point-score calibration")
    artifact_fields = _v4_artifact_fields(
        artifact_v3,
        checkpoint_sha256,
        experiment_config_path,
        protocol_config,
        entry,
    )
    artifact_fields.update(
        {
            "offline_point_threshold": threshold_values["offline_point_threshold"],
            "point_score_c": calibration.center,
            "point_score_tau": calibration.tau,
        }
    )
    artifact_v4 = build_threshold_artifact(
        **artifact_fields,
        online_ewma_point_threshold=threshold_values["online_ewma_point_threshold"],
        input_window_threshold=threshold_values["input_window_threshold"],
        latent_window_low_threshold=threshold_values["latent_window_low_threshold"],
        latent_window_high_threshold=threshold_values["latent_window_high_threshold"],
        latent_window_low_quantile=threshold_values["latent_window_low_quantile"],
    )
    variant_name_resolution = (
        "checkpoint_and_config_verified_recovery"
        if artifact_v3["variant_name"] != entry.offline_variant
        else "artifact_v3_identity_match"
    )
    artifact_v4["provenance"]["artifact_v3_variant_name"] = str(
        artifact_v3["variant_name"]
    )
    artifact_v4["provenance"]["variant_name_resolution"] = variant_name_resolution
    audit = {
        "clean_validation_window_count": len(calibration_scores["input_window"]),
        "offline_point_count": len(calibration_scores["offline_raw"]),
        "point_score_calibration": calibration.to_artifact_fields(),
        "checkpoint_sha256": checkpoint_sha256,
        "artifact_v3_variant_name": str(artifact_v3["variant_name"]),
        "artifact_v4_variant_name": entry.offline_variant,
        "variant_name_resolution": variant_name_resolution,
        **threshold_values,
    }
    return artifact_v4, audit


def recalibrate_entry(
    entry: StageBInventoryEntry, protocol_config: dict[str, Any]
) -> dict[str, Any]:
    artifact_v3 = load_threshold_artifact(entry.threshold_artifact_v3_path)
    _validate_v3_identity(artifact_v3, entry)
    calibration_scores = _collect_clean_validation_scores(entry, protocol_config)
    artifact_v4, audit = build_v4_threshold_artifact(
        artifact_v3=artifact_v3,
        checkpoint_sha256=sha256_file(entry.stage_b_best_checkpoint_path),
        calibration_scores=calibration_scores,
        protocol_config=protocol_config,
        experiment_config_path=entry.experiment_config_path,
        entry=entry,
    )
    write_threshold_artifact(artifact_v4, entry.threshold_artifact_v4_path)
    entry.audit_path.write_text(
        json.dumps(
            {
                "artifact_v3_path": str(entry.threshold_artifact_v3_path),
                "artifact_v4_path": str(entry.threshold_artifact_v4_path),
                "stage_b_best_checkpoint_path": str(entry.stage_b_best_checkpoint_path),
                "identity": {
                    "offline_variant": entry.offline_variant,
                    "entity_id": entry.entity_id,
                    "seed": entry.seed,
                },
                "calibration": audit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "created",
        "entry": entry_as_report_value(entry),
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "clean_validation_window_count": audit["clean_validation_window_count"],
    }


def run_recalibration(
    *,
    experiment_config_directory: Path,
    protocol_config_path: Path,
    v4_filename: str,
    report_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    entries = discover_stage_b_inventory(experiment_config_directory, v4_filename)
    failures = preflight_inventory(entries)
    report: dict[str, Any] = {
        "created_by": "scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py",
        "dry_run": dry_run,
        "inventory_entry_count": len(entries),
        "preflight_failures": failures,
        "results": [],
    }
    if failures or dry_run:
        return report

    protocol_config = _load_yaml_mapping(protocol_config_path)
    validate_protocol_config(protocol_config)
    for entry in entries:
        try:
            report["results"].append(recalibrate_entry(entry, protocol_config))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            report["results"].append(
                {
                    "status": "failed",
                    "entry": entry_as_report_value(entry),
                    "reason": str(error),
                }
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config-directory",
        type=Path,
        default=DEFAULT_EXPERIMENT_CONFIG_DIRECTORY,
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=DEFAULT_PROTOCOL_CONFIG_PATH,
    )
    parser.add_argument("--v4-filename", default=DEFAULT_V4_FILENAME)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_recalibration(
        experiment_config_directory=args.experiment_config_directory,
        protocol_config_path=args.protocol_config,
        v4_filename=args.v4_filename,
        report_path=args.report_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    has_entry_failures = any(
        result["status"] == "failed" for result in report["results"]
    )
    return 1 if report["preflight_failures"] or has_entry_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.core.uq_summary import (
    build_uq_summary_payload,
    validate_uq_summary_payload,
    write_uq_summary_json,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_existing_path(candidate: str | None, base_dir: Path) -> Path | None:
    if candidate is None:
        return None
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    direct_path = base_dir / candidate_path
    if direct_path.exists():
        return direct_path
    repo_root = Path(__file__).resolve().parents[2]
    repo_path = repo_root / candidate_path
    if repo_path.exists():
        return repo_path
    return direct_path


def _infer_identity(output_dir: Path) -> dict[str, str | int]:
    variant_name = "unknown"
    entity_id = "unknown"
    seed = 0
    parts = list(output_dir.parts)
    for index, part in enumerate(parts):
        if re.fullmatch(r"seed\d+", part):
            seed = int(part.removeprefix("seed"))
            if index > 0:
                entity_id = parts[index - 1]
            if index > 1:
                variant_name = parts[index - 2]
            break
    return {
        "variant_name": variant_name,
        "entity_id": entity_id,
        "seed": seed,
    }


def _infer_entity_id_from_traces(split_inputs: dict[str, dict[str, Any]]) -> str | None:
    for split_name in ("test", "synthetic_validation", "clean_validation"):
        traces = list(split_inputs.get(split_name, {}).get("traces") or [])
        if not traces:
            continue
        first_trace = traces[0]
        entity_ids = first_trace.get("entity_ids") or []
        if entity_ids:
            return str(entity_ids[0])
    return None


def _load_npz_scores(path: Path) -> np.ndarray:
    if not path.exists():
        return np.asarray([], dtype=float)
    payload = np.load(path)
    return np.asarray(payload["point_scores"], dtype=float)


def _load_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(_load_json(path))


def _build_run_scalar_logs(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {
            "query/continuous_temperature": None,
            "query/discrete_temperature": None,
            "query/num_samples_train": None,
            "query/num_samples_eval": None,
            "query/continuous_weight_entropy_mean": None,
            "query/discrete_topk_weight_entropy_mean": None,
        }
    model_config = dict(_load_yaml(config_path).get("model", {}))
    return {
        "query/continuous_temperature": model_config.get("continuous_temperature"),
        "query/discrete_temperature": model_config.get("discrete_temperature"),
        "query/num_samples_train": model_config.get("monte_carlo_samples"),
        "query/num_samples_eval": model_config.get("monte_carlo_samples"),
        "query/continuous_weight_entropy_mean": model_config.get(
            "continuous_weight_entropy_mean"
        ),
        "query/discrete_topk_weight_entropy_mean": model_config.get(
            "discrete_topk_weight_entropy_mean"
        ),
    }


def _default_stage_b_config_path(output_dir: Path) -> Path:
    return (
        output_dir
        / "two_stage"
        / "generated_configs"
        / "02_stage_b_fusion_finetuning.yaml"
    )


def _load_run_context(
    benchmark_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = (
        benchmark_output_dir / "benchmark" / "thesis_offline_benchmark_report.json"
    )
    manifest_path = benchmark_output_dir / "two_stage" / "two_stage_manifest.json"
    report: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if report_path.exists():
        report = _load_json(report_path)
        manifest = dict(report.get("two_stage_manifest") or {})
    elif manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        raise FileNotFoundError(
            f"missing benchmark report and two-stage manifest under: {benchmark_output_dir}"
        )
    return report, manifest


def backfill_uq_summary(
    *,
    benchmark_output_dir: Path,
    write_retention_copy: bool = True,
) -> dict[str, Any]:
    report, manifest = _load_run_context(benchmark_output_dir)
    identity = _infer_identity(benchmark_output_dir)
    experiment_name = str(
        manifest.get("experiment_name")
        or report.get("experiment_name")
        or benchmark_output_dir.name
    )
    stage_entries = list(manifest.get("training_stages") or [])
    stage_b_entry = stage_entries[1] if len(stage_entries) > 1 else {}
    checkpoint_path = _resolve_existing_path(
        str(
            manifest.get("evaluation", {}).get("checkpoint_path")
            or stage_b_entry.get("best_checkpoint_path")
            or (
                benchmark_output_dir
                / "two_stage"
                / "stage_b_fusion_finetuning"
                / "checkpoints"
                / "best.pt"
            )
        ),
        benchmark_output_dir,
    )
    experiment_config_path = _resolve_existing_path(
        str(
            stage_b_entry.get("config_path")
            or (
                benchmark_output_dir
                / "two_stage"
                / "generated_configs"
                / "02_stage_b_fusion_finetuning.yaml"
            )
        ),
        benchmark_output_dir,
    )
    protocol_config_path = _resolve_existing_path(
        str(
            report.get("protocol_config_path")
            or "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml"
        ),
        benchmark_output_dir,
    )
    split_inputs = {}
    for split_name in ("clean_validation", "synthetic_validation", "test"):
        split_inputs[split_name] = {
            "point_scores": _load_npz_scores(
                benchmark_output_dir / "scores" / f"{split_name}_point_scores.npz"
            ),
            "traces": _load_traces(
                benchmark_output_dir / "traces" / f"{split_name}_traces.json"
            ),
        }
    trace_entity_id = _infer_entity_id_from_traces(split_inputs)
    if trace_entity_id is not None:
        identity["entity_id"] = trace_entity_id
    payload = build_uq_summary_payload(
        benchmark_kind="offline",
        experiment_name=experiment_name,
        method_name="THESIS",
        variant_name=str(identity["variant_name"]),
        entity_id=str(identity["entity_id"]),
        seed=int(identity["seed"]),
        stage_name=str(stage_b_entry.get("stage_name") or "stage_b_fusion_finetuning"),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else "",
        checkpoint_sha256=None,
        experiment_config_path=str(experiment_config_path)
        if experiment_config_path
        else "",
        protocol_config_path=str(protocol_config_path) if protocol_config_path else "",
        output_dir=str(benchmark_output_dir),
        run_scalar_logs=_build_run_scalar_logs(experiment_config_path),
        split_inputs=split_inputs,
    )
    if checkpoint_path and checkpoint_path.exists():
        from src.core.artifact_integrity import sha256_file

        payload["run"]["checkpoint_sha256"] = sha256_file(checkpoint_path)
        validate_uq_summary_payload(payload)
    uq_summary_path = benchmark_output_dir / "metrics" / "uq_summary.json"
    write_uq_summary_json(uq_summary_path, payload)
    retention_summary_path: Path | None = None
    if write_retention_copy:
        retention_root = (
            benchmark_output_dir / "retention" / str(identity["entity_id"]) / "offline"
        )
        retention_summary_path = retention_root / "uq_summary.json"
        write_uq_summary_json(retention_summary_path, payload)
    return {
        "uq_summary_path": str(uq_summary_path),
        "retention_summary_path": None
        if retention_summary_path is None
        else str(retention_summary_path),
        "identity": identity,
        "experiment_name": experiment_name,
        "split_names": list(split_inputs.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-output-dir", required=True)
    parser.add_argument("--no-retention-copy", action="store_true")
    args = parser.parse_args()
    result = backfill_uq_summary(
        benchmark_output_dir=Path(args.benchmark_output_dir),
        write_retention_copy=not args.no_retention_copy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

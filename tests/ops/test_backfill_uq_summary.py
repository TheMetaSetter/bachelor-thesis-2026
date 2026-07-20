from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.ops.backfill_uq_summary import backfill_uq_summary
from src.core.uq_summary import validate_uq_summary_payload


def _write_npz(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        point_scores=np.asarray(values, dtype=float),
        point_labels=np.asarray([0] * len(values), dtype=np.int64),
        covered_point_mask=np.asarray([True] * len(values), dtype=bool),
    )


def _write_trace(path: Path, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "entity_ids": ["machine-3-9"],
                    "sample_retention_policy": "retain_for_eda",
                    "window_score_history": [score, score + 0.1],
                    "uncertainty_history": {
                        "point_anomaly_score_variance": [score, score + 0.2],
                        "window_anomaly_score_variance": [score + 0.3],
                        "classification_variance_mean": [score + 0.4],
                    },
                    "mc_sample_histories": {
                        "point_score_samples": [score],
                        "window_score_samples": [score],
                        "reconstruction_samples": [score],
                        "classification_probability_samples": [score],
                    },
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_backfill_uq_summary_writes_metrics_and_retention(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "benchmark_smoke" / "smd" / "thesis" / "O0" / "machine_3_9" / "seed8"
    report_path = output_dir / "benchmark" / "thesis_offline_benchmark_report.json"
    stage_b_config_path = (
        output_dir / "two_stage" / "generated_configs" / "02_stage_b_fusion_finetuning.yaml"
    )
    checkpoint_path = (
        output_dir / "two_stage" / "stage_b_fusion_finetuning" / "checkpoints" / "best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    stage_b_config_path.parent.mkdir(parents=True, exist_ok=True)
    stage_b_config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "smd__thesis__offline__O0__machine_3_9__w20__seed8__smoke__stage_b_fusion_finetuning",
                "model": {
                    "continuous_temperature": 0.9,
                    "discrete_temperature": 0.9,
                    "monte_carlo_samples": 10,
                    "continuous_weight_entropy_mean": 0.0,
                    "discrete_topk_weight_entropy_mean": 0.0,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "protocol_config_path": "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
                "two_stage_manifest": {
                    "experiment_name": "smd__thesis__offline__O0__machine_3_9__w20__seed8__smoke",
                    "evaluation": {"checkpoint_path": str(checkpoint_path)},
                    "training_stages": [
                        {
                            "stage_name": "stage_a_multitask_pretraining",
                            "config_path": str(
                                output_dir
                                / "two_stage"
                                / "generated_configs"
                                / "01_stage_a_multitask_pretraining.yaml"
                            ),
                        },
                        {
                            "stage_name": "stage_b_fusion_finetuning",
                            "config_path": str(stage_b_config_path),
                            "best_checkpoint_path": str(checkpoint_path),
                        },
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_npz(output_dir / "scores" / "clean_validation_point_scores.npz", [0.1, 0.2, 0.3])
    _write_npz(output_dir / "scores" / "synthetic_validation_point_scores.npz", [0.4, 0.5])
    _write_npz(output_dir / "scores" / "test_point_scores.npz", [0.6, 0.7])
    _write_trace(output_dir / "traces" / "clean_validation_traces.json", 0.15)
    _write_trace(output_dir / "traces" / "synthetic_validation_traces.json", 0.25)
    _write_trace(output_dir / "traces" / "test_traces.json", 0.35)

    result = backfill_uq_summary(benchmark_output_dir=output_dir, write_retention_copy=True)

    metrics_path = output_dir / "metrics" / "uq_summary.json"
    retention_path = output_dir / "retention" / "machine-3-9" / "offline" / "uq_summary.json"
    assert metrics_path.exists()
    assert retention_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_uq_summary_payload(payload)
    assert payload["run"]["variant_name"] == "O0"
    assert payload["run"]["entity_id"] == "machine-3-9"
    assert payload["splits"]["clean_validation"]["num_traces"] == 1
    assert result["retention_summary_path"] == str(retention_path)


def test_backfill_uq_summary_uses_manifest_when_report_is_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "benchmark_smoke" / "smd" / "thesis" / "O0" / "machine_3_9" / "seed8"
    manifest_path = output_dir / "two_stage" / "two_stage_manifest.json"
    stage_b_config_path = (
        output_dir / "two_stage" / "generated_configs" / "02_stage_b_fusion_finetuning.yaml"
    )
    checkpoint_path = (
        output_dir / "two_stage" / "stage_b_fusion_finetuning" / "checkpoints" / "best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    stage_b_config_path.parent.mkdir(parents=True, exist_ok=True)
    stage_b_config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "smd__thesis__offline__O0__machine_3_9__w20__seed8__smoke__stage_b_fusion_finetuning",
                "model": {
                    "continuous_temperature": 0.9,
                    "discrete_temperature": 0.9,
                    "monte_carlo_samples": 10,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_name": "smd__thesis__offline__O0__machine_3_9__w20__seed8__smoke",
                "evaluation": {"checkpoint_path": str(checkpoint_path)},
                "training_stages": [
                    {
                        "stage_name": "stage_a_multitask_pretraining",
                        "config_path": str(
                            output_dir
                            / "two_stage"
                            / "generated_configs"
                            / "01_stage_a_multitask_pretraining.yaml"
                        ),
                    },
                    {
                        "stage_name": "stage_b_fusion_finetuning",
                        "config_path": str(stage_b_config_path),
                        "best_checkpoint_path": str(checkpoint_path),
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_npz(output_dir / "scores" / "clean_validation_point_scores.npz", [0.1, 0.2])
    _write_trace(output_dir / "traces" / "clean_validation_traces.json", 0.15)

    result = backfill_uq_summary(benchmark_output_dir=output_dir, write_retention_copy=False)

    assert (output_dir / "metrics" / "uq_summary.json").exists()
    assert result["retention_summary_path"] is None


def test_backfill_uq_summary_rejects_missing_report_and_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        backfill_uq_summary(benchmark_output_dir=tmp_path / "missing")

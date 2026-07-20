from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.ops.backfill_all_uq_summaries import backfill_all_uq_summaries
from scripts.ops.prune_raw_trace_artifacts import prune_raw_trace_artifacts


def _write_npz(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        point_scores=np.asarray(values, dtype=float),
        point_labels=np.asarray([0] * len(values), dtype=np.int64),
        covered_point_mask=np.asarray([True] * len(values), dtype=bool),
    )


def _write_trace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "entity_ids": ["machine-3-9"],
                    "sample_retention_policy": "retain_for_eda",
                    "window_score_history": [0.1, 0.2],
                    "uncertainty_history": {
                        "point_anomaly_score_variance": [0.1, 0.2],
                    },
                    "mc_sample_histories": {},
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, checkpoint_path: Path, stage_b_config_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "protocol_config_path": "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
                "two_stage_manifest": {
                    "experiment_name": "smd__thesis__offline__O0__machine_3_9__w20__seed8__smoke",
                    "evaluation": {"checkpoint_path": str(checkpoint_path)},
                    "training_stages": [
                        {
                            "stage_name": "stage_a_multitask_pretraining",
                            "config_path": "stage_a.yaml",
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


def test_backfill_all_uq_summaries_processes_reports(tmp_path: Path) -> None:
    root_dir = tmp_path / "outputs"
    benchmark_dir = root_dir / "benchmark_smoke" / "smd" / "thesis" / "O0" / "machine_3_9" / "seed8"
    checkpoint_path = benchmark_dir / "two_stage" / "stage_b_fusion_finetuning" / "checkpoints" / "best.pt"
    stage_b_config_path = benchmark_dir / "two_stage" / "generated_configs" / "02_stage_b_fusion_finetuning.yaml"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"ckpt")
    stage_b_config_path.parent.mkdir(parents=True, exist_ok=True)
    stage_b_config_path.write_text(
        "model:\n  continuous_temperature: 0.9\n  discrete_temperature: 0.9\n  monte_carlo_samples: 10\n",
        encoding="utf-8",
    )
    _write_report(benchmark_dir / "benchmark" / "thesis_offline_benchmark_report.json", checkpoint_path, stage_b_config_path)
    _write_npz(benchmark_dir / "scores" / "clean_validation_point_scores.npz", [0.1, 0.2, 0.3])
    _write_npz(benchmark_dir / "scores" / "synthetic_validation_point_scores.npz", [0.4, 0.5])
    _write_npz(benchmark_dir / "scores" / "test_point_scores.npz", [0.6, 0.7])
    _write_trace(benchmark_dir / "traces" / "clean_validation_traces.json")
    _write_trace(benchmark_dir / "traces" / "synthetic_validation_traces.json")
    _write_trace(benchmark_dir / "traces" / "test_traces.json")

    result = backfill_all_uq_summaries(root_dir=root_dir, write_retention_copy=True)

    assert result["report_count"] == 1
    assert (benchmark_dir / "metrics" / "uq_summary.json").exists()
    assert (benchmark_dir / "retention" / "machine-3-9" / "offline" / "uq_summary.json").exists()


def test_prune_raw_trace_artifacts_requires_summary(tmp_path: Path) -> None:
    root_dir = tmp_path / "outputs"
    benchmark_dir = root_dir / "benchmark_smoke" / "smd" / "thesis" / "O0" / "machine_3_9" / "seed8"
    trace_path = benchmark_dir / "traces" / "test_traces.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("[]\n", encoding="utf-8")

    result = prune_raw_trace_artifacts(root_dir=root_dir, dry_run=True)
    assert result["compacted_count"] == 0
    assert trace_path.exists()


def test_prune_raw_trace_artifacts_compacts_in_place_when_summary_exists(
    tmp_path: Path,
) -> None:
    root_dir = tmp_path / "outputs"
    benchmark_dir = root_dir / "benchmark_smoke" / "smd" / "thesis" / "O0" / "machine_3_9" / "seed8"
    trace_path = benchmark_dir / "traces" / "test_traces.json"
    summary_path = benchmark_dir / "metrics" / "uq_summary.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}\n", encoding="utf-8")
    trace_path.write_text(
        json.dumps(
            [
                {
                    "batch_index": 1,
                    "entity_ids": ["machine-3-9"],
                    "point_score_history": [0.1, 0.2],
                    "window_score_history": [0.3],
                    "uncertainty_history": {
                        "point_anomaly_score_variance": [0.4],
                    },
                    "stochastic_query": {
                        "schema_version": 3,
                        "enabled": True,
                        "num_samples": 10,
                        "continuous_temperature": 0.9,
                        "discrete_temperature": 0.8,
                        "return_mc_samples": False,
                        "sample_retention_policy": "none",
                        "point_score_samples": [1.0, 2.0],
                    },
                    "mc_sample_histories": {
                        "point_score_samples": [1.0, 2.0],
                    },
                    "deterministic_geometry": {
                        "hidden_reconstruction": [1.0],
                    },
                }
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = prune_raw_trace_artifacts(root_dir=root_dir, dry_run=False)

    assert result["compacted_count"] == 1
    compacted_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert compacted_payload[0]["uncertainty_history"] == {
        "point_anomaly_score_variance": [0.4]
    }
    assert "mc_sample_histories" not in compacted_payload[0]
    assert "deterministic_geometry" not in compacted_payload[0]
    assert compacted_payload[0]["stochastic_query"] == {
        "schema_version": 3,
        "enabled": True,
        "num_samples": 10,
        "continuous_temperature": 0.9,
        "discrete_temperature": 0.8,
        "return_mc_samples": False,
        "sample_retention_policy": "none",
    }

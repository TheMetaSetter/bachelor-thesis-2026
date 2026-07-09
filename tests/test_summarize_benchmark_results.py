from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_benchmark_results import summarize_benchmark_results


def test_summarize_benchmark_results_normalizes_offline_and_online_reports(
    tmp_path: Path,
) -> None:
    offline_report_path = (
        tmp_path / "offline" / "benchmark" / "offline_benchmark_report.json"
    )
    offline_report_path.parent.mkdir(parents=True, exist_ok=True)
    offline_report_path.write_text(
        json.dumps(
            {
                "benchmark_status": "completed",
                "created_at_utc": "2026-07-10T00:00:00Z",
                "benchmark_config_path": "configs/experiment/offline_benchmark/stumpy.yaml",
                "protocol_config_path": "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
                "benchmark_config": {
                    "baseline_name": "stumpy",
                    "entity_id": "machine_1_6",
                    "seed": 6,
                },
                "protocol": {
                    "test_label_usage": "metrics_only",
                },
                "thresholds": {
                    "offline_point": {
                        "value": 0.5,
                        "source_split": "clean_validation",
                        "score_rule": "nonoverlap_tail_average",
                    },
                    "online_ewma_point": {
                        "value": 0.7,
                        "source_split": "clean_validation",
                        "score_rule": "stride1_causal_endpoint_ewma",
                    },
                },
                "offline_metrics": {"point_f1": 0.9},
                "runtime_seconds": 12.5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    online_report_path = (
        tmp_path / "online" / "benchmark" / "online_streaming_benchmark_report.json"
    )
    online_report_path.parent.mkdir(parents=True, exist_ok=True)
    online_report_path.write_text(
        json.dumps(
            {
                "benchmark_status": "completed",
                "created_at_utc": "2026-07-10T00:05:00Z",
                "benchmark_config_path": "configs/experiment/online_benchmark/candi/sample.yaml",
                "protocol_config_path": "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
                "benchmark_config": {
                    "baseline_name": "candi",
                    "online_variant": "A1",
                    "seed": 8,
                    "entity_id": "machine_3_4",
                },
                "protocol": {
                    "test_label_usage": "metrics_only",
                },
                "online_execution": {
                    "threshold_source": "clean_validation_stride1_ewma",
                    "threshold_value": 0.8,
                    "metric_history": [{"online/step": 1}, {"online/step": 2}],
                    "records": [
                        {"did_update": True},
                        {"did_update": False},
                    ],
                    "threshold_artifact": {
                        "thresholds": {
                            "offline_point": {
                                "score_rule": "nonoverlap_tail_average",
                            },
                            "online_ewma_point": {
                                "score_rule": "stride1_causal_endpoint_ewma",
                            },
                        }
                    },
                },
                "runtime_seconds": 7.25,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "summary" / "benchmark_summary.json"

    summary = summarize_benchmark_results(
        report_paths=[offline_report_path, online_report_path],
        output_path=output_path,
    )

    assert output_path.exists()
    assert (tmp_path / "summary" / "benchmark_summary.csv").exists()
    assert len(summary["rows"]) == 2

    offline_row = next(
        row for row in summary["rows"] if row["benchmark_type"] == "offline"
    )
    assert offline_row["method"] == "stumpy"
    assert offline_row["variant"] == "stumpy"
    assert offline_row["entity_id"] == "machine_1_6"
    assert offline_row["seed"] == 6
    assert offline_row["threshold_source"] == "clean_validation"
    assert offline_row["point_rule"] == "nonoverlap_tail_average"
    assert offline_row["smoothing_rule"] == "stride1_causal_endpoint_ewma"
    assert offline_row["test_label_usage"] == "metrics_only"
    assert offline_row["row_kind"] == "regular"
    assert offline_row["runtime_seconds"] == 12.5
    assert offline_row["metrics"]["point_f1"] == 0.9

    online_row = next(
        row for row in summary["rows"] if row["benchmark_type"] == "online"
    )
    assert online_row["method"] == "candi"
    assert online_row["variant"] == "A1"
    assert online_row["entity_id"] == "machine_3_4"
    assert online_row["seed"] == 8
    assert online_row["threshold_source"] == "clean_validation_stride1_ewma"
    assert online_row["point_rule"] == "nonoverlap_tail_average"
    assert online_row["smoothing_rule"] == "stride1_causal_endpoint_ewma"
    assert online_row["test_label_usage"] == "metrics_only"
    assert online_row["row_kind"] == "regular"
    assert online_row["runtime_seconds"] == 7.25
    assert online_row["metrics"]["num_records"] == 2
    assert online_row["metrics"]["num_updates"] == 1


def test_summarize_benchmark_results_marks_special_row_kinds(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "oracle" / "benchmark" / "oracle_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "benchmark_status": "completed",
                "benchmark_config": {
                    "baseline_name": "stumpy",
                    "entity_id": "machine_1_6",
                    "seed": 6,
                    "row_kind": "oracle",
                },
                "protocol": {"test_label_usage": "metrics_only"},
                "thresholds": {
                    "offline_point": {
                        "score_rule": "oracle_threshold",
                        "source_split": "clean_validation",
                    },
                    "online_ewma_point": {
                        "score_rule": "oracle_smoothing",
                        "source_split": "clean_validation",
                    },
                },
                "runtime_seconds": 1.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = summarize_benchmark_results(
        report_paths=[report_path],
        output_path=tmp_path / "summary" / "special_summary.json",
    )

    assert summary["rows"][0]["row_kind"] == "oracle"

from __future__ import annotations


def test_forensic_markdown_conclusion_is_dataset_aware_for_truncated_smd() -> None:
    from scripts.ops.forensic_audit_run import _build_forensic_markdown

    markdown = _build_forensic_markdown(
        experiment_config_path="configs/experiment/comparative/baseline/smd.yaml",
        experiment_config={
            "experiment_name": "smd-smoke",
            "data_config_path": "configs/data/smd.yaml",
        },
        report={
            "dataset_name": "smd",
            "warnings": [
                "Test split window coverage is truncated relative to the raw test timeline.",
                "Configured max_test_windows=1 truncates the evaluated test timeline.",
            ],
            "splits": {
                "test": {
                    "label_regime": "mixed",
                    "positive_ratio": 0.1,
                    "num_windows": 1,
                    "is_truncated": True,
                    "raw_num_points": 100,
                    "evaluated_num_points": 20,
                }
            },
        },
        observed_metrics={
            "precision": 0.0,
            "recall": 0.0,
            "threshold": 0.5,
        },
    )

    lower_markdown = markdown.lower()
    assert "truncated early-prefix evaluation" in lower_markdown
    assert "all-positive `anomaly_archive` test slice" not in markdown


def test_forensic_markdown_conclusion_mentions_all_positive_anomaly_archive() -> None:
    from scripts.ops.forensic_audit_run import _build_forensic_markdown

    markdown = _build_forensic_markdown(
        experiment_config_path="configs/experiment/scale/anomaly_archive.yaml",
        experiment_config={
            "experiment_name": "staffiii",
            "data_config_path": "configs/data/anomaly_archive_staffiii_full.yaml",
        },
        report={
            "dataset_name": "anomaly_archive",
            "warnings": [
                "Configured max_test_windows=32 truncates the evaluated test timeline. Treat this as a truncated smoke evaluation, not a full-timeline test.",
            ],
            "benchmark_comparability": "non_comparable",
            "splits": {
                "test": {
                    "label_regime": "mixed",
                    "positive_ratio": 0.1,
                    "num_windows": 21,
                    "is_truncated": True,
                    "raw_num_points": 258228,
                    "evaluated_num_points": 220,
                }
            },
        },
        observed_metrics={
            "precision": 0.2,
            "recall": 0.05,
            "threshold": 0.072061,
        },
    )

    lower_markdown = markdown.lower()
    assert "truncated early-prefix evaluation" in lower_markdown
    assert "all-positive `anomaly_archive` test slice" not in markdown

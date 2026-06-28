from __future__ import annotations

from pathlib import Path

import torch

from src.data.loaders import WindowDataset
from src.data.scalers import SequenceStandardScaler
from src.engine.evaluator import reconstruct_pointwise_records_from_window_payload


def _build_sequence(
    *,
    dataset_name: str,
    entity_id: str,
    split: str,
    values: list[float],
    labels: list[int],
) -> dict[str, object]:
    x = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    point_labels = torch.tensor(labels, dtype=torch.long)
    return {
        "x": x,
        "point_labels": point_labels,
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": dataset_name,
            "entity_id": entity_id,
            "split": split,
            "series_id": f"{dataset_name}:{split}:{entity_id}",
            "num_channels": 1,
            "sequence_length": len(values),
            "source_sequence_length": len(values),
        },
    }


def test_reconstructed_records_keep_evaluated_coverage_metadata() -> None:
    sequences_by_entity = {
        "machine-1": _build_sequence(
            dataset_name="smd",
            entity_id="machine-1",
            split="test",
            values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            labels=[0, 0, 0, 1, 1, 0],
        )
    }
    batch_payloads = [
        {
            "meta": [
                {
                    "entity_id": "machine-1",
                    "start_index": 0,
                    "end_index": 3,
                }
            ],
            "point_scores": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "point_labels": torch.tensor([[0, 0, 0]], dtype=torch.long),
        }
    ]

    reconstructed_records = reconstruct_pointwise_records_from_window_payload(
        sequences_by_entity=sequences_by_entity,
        batch_payloads=batch_payloads,
    )

    assert reconstructed_records[0]["evaluated_start_index"] == 0
    assert reconstructed_records[0]["evaluated_end_index"] == 3
    assert reconstructed_records[0]["evaluated_num_points"] == 3
    assert reconstructed_records[0]["raw_num_points"] == 6


def test_reconstructed_records_keep_uncovered_suffix_as_zero_scored_points() -> None:
    sequences_by_entity = {
        "machine-1": _build_sequence(
            dataset_name="smd",
            entity_id="machine-1",
            split="test",
            values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            labels=[0, 0, 0, 1, 1, 0],
        )
    }
    batch_payloads = [
        {
            "meta": [
                {
                    "entity_id": "machine-1",
                    "start_index": 0,
                    "end_index": 3,
                }
            ],
            "point_scores": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "point_labels": torch.tensor([[0, 0, 0]], dtype=torch.long),
        }
    ]

    reconstructed_records = reconstruct_pointwise_records_from_window_payload(
        sequences_by_entity=sequences_by_entity,
        batch_payloads=batch_payloads,
    )

    assert torch.allclose(
        reconstructed_records[0]["point_scores"],
        torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    assert torch.equal(
        reconstructed_records[0]["point_labels"],
        torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
    )
    assert reconstructed_records[0]["evaluated_num_points"] == 3
    assert reconstructed_records[0]["raw_num_points"] == 6


def test_build_dataset_protocol_audit_report_flags_truncated_smd_test_coverage() -> (
    None
):
    from src.analysis.evaluation_protocol_audit import (
        build_dataset_protocol_audit_report,
    )

    train_sequence = _build_sequence(
        dataset_name="smd",
        entity_id="machine-1",
        split="train",
        values=[1.0, 2.0, 3.0, 4.0],
        labels=[0, 0, 0, 0],
    )
    val_sequence = _build_sequence(
        dataset_name="smd",
        entity_id="machine-1",
        split="val",
        values=[5.0, 6.0, 7.0, 8.0],
        labels=[0, 0, 0, 0],
    )
    test_sequence = _build_sequence(
        dataset_name="smd",
        entity_id="machine-1",
        split="test",
        values=[0.0, 0.0, 0.0, 0.0, 3.0, 4.0],
        labels=[0, 0, 0, 0, 1, 1],
    )
    scaler = SequenceStandardScaler()
    scaler.fit([train_sequence])
    scaled_sequences = {
        "train": scaler.transform_sequences([train_sequence]),
        "val": scaler.transform_sequences([val_sequence]),
        "test": scaler.transform_sequences([test_sequence]),
    }
    datasets = {
        "train": WindowDataset(scaled_sequences["train"], window_size=3, stride=1),
        "val": WindowDataset(scaled_sequences["val"], window_size=3, stride=1),
        "test": WindowDataset(
            scaled_sequences["test"],
            window_size=3,
            stride=1,
            max_windows=1,
        ),
    }
    report = build_dataset_protocol_audit_report(
        data_bundle={
            "dataset_name": "smd",
            "parser": object(),
            "scaler": scaler,
            "raw_sequences": {
                "train": [train_sequence],
                "val": [val_sequence],
                "test": [test_sequence],
            },
            "scaled_sequences": scaled_sequences,
            "datasets": datasets,
            "loaders": {},
        },
        data_config={
            "dataset_name": "smd",
            "window_size": 3,
            "stride": 1,
            "max_test_windows": 1,
        },
    )

    entity_report = report["splits"]["test"]["entities"][0]
    assert report["splits"]["test"]["is_truncated"] is True
    assert report["splits"]["test"]["raw_num_points"] == 6
    assert report["splits"]["test"]["evaluated_num_points"] == 3
    assert entity_report["evaluated_num_points"] == 3
    assert entity_report["raw_num_points"] == 6
    assert "truncated" in " ".join(report["warnings"]).lower()
    assert "max_test_windows" in " ".join(report["warnings"])


def test_build_dataset_protocol_audit_report_marks_benchmark_comparable_anomaly_archive() -> None:
    from src.analysis.evaluation_protocol_audit import (
        build_dataset_protocol_audit_report,
        render_dataset_protocol_audit_markdown,
    )
    from src.data.datasets.anomaly_archive import AnomalyArchiveDatasetParser

    file_path = Path(
        "data/AnomalyArchive/219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt"
    )
    parser = AnomalyArchiveDatasetParser(
        file_path=file_path,
        validation_split_ratio=0.2,
    )
    raw_sequences = parser.parse()
    scaler = SequenceStandardScaler()
    scaler.fit(raw_sequences["train"])
    scaled_sequences = {
        split_name: scaler.transform_sequences(split_sequences)
        for split_name, split_sequences in raw_sequences.items()
    }
    datasets = {
        split_name: WindowDataset(
            scaled_sequences[split_name],
            window_size=20,
            stride=10,
        )
        for split_name in ("train", "val", "test")
    }

    report = build_dataset_protocol_audit_report(
        data_bundle={
            "dataset_name": "anomaly_archive",
            "parser": parser,
            "scaler": scaler,
            "raw_sequences": raw_sequences,
            "scaled_sequences": scaled_sequences,
            "datasets": datasets,
            "loaders": {},
        },
        data_config={
            "dataset_name": "anomaly_archive",
            "file_path": str(file_path),
            "window_size": 20,
            "stride": 10,
        },
    )
    markdown_report = render_dataset_protocol_audit_markdown(
        report,
        experiment_name="staffiii",
    )

    assert report["benchmark_comparability"] == "benchmark_comparable"
    assert report["splits"]["test"]["label_regime"] == "mixed"
    assert "Benchmark Protocol Status" in markdown_report
    assert "benchmark_comparable" in markdown_report
    assert "pre_vs_anomaly" not in markdown_report


def test_describe_metric_regime_implications_explains_all_positive_case() -> None:
    from src.analysis.evaluation_protocol_audit import (
        describe_metric_regime_implications,
    )

    implications = describe_metric_regime_implications(
        label_regime="all_one",
        threshold=0.072061,
        observed_metrics={
            "precision": 1.0,
            "recall": 0.05,
            "pr_auc": 1.0,
            "roc_auc": float("nan"),
            "vus_pr": float("nan"),
        },
    )

    assert implications["regime_name"] == "all_positive_test_labels"
    assert "all anomalous" in implications["summary"]
    implication_text = " ".join(implications["implications"]).lower()
    assert "precision can stay at 1.0" in implication_text
    assert "roc-auc becomes undefined" in implication_text
    assert "0.072061" in implications["threshold_note"]


def test_build_dataset_protocol_audit_report_accepts_partial_metric_bundle() -> None:
    from src.analysis.evaluation_protocol_audit import (
        build_dataset_protocol_audit_report,
    )
    from src.data.datasets.anomaly_archive import AnomalyArchiveDatasetParser

    file_path = Path(
        "data/AnomalyArchive/219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt"
    )
    parser = AnomalyArchiveDatasetParser(
        file_path=file_path,
        validation_split_ratio=0.2,
    )
    raw_sequences = parser.parse()
    scaler = SequenceStandardScaler()
    scaler.fit(raw_sequences["train"])
    scaled_sequences = {
        split_name: scaler.transform_sequences(split_sequences)
        for split_name, split_sequences in raw_sequences.items()
    }
    datasets = {
        split_name: WindowDataset(
            scaled_sequences[split_name],
            window_size=20,
            stride=10,
        )
        for split_name in ("train", "val", "test")
    }

    report = build_dataset_protocol_audit_report(
        data_bundle={
            "dataset_name": "anomaly_archive",
            "parser": parser,
            "scaler": scaler,
            "raw_sequences": raw_sequences,
            "scaled_sequences": scaled_sequences,
            "datasets": datasets,
            "loaders": {},
        },
        data_config={
            "dataset_name": "anomaly_archive",
            "file_path": str(file_path),
            "window_size": 20,
            "stride": 10,
        },
        evaluation_outputs={
            "metrics": {
                "threshold": 0.072061,
                "precision": 1.0,
                "recall": 0.05,
                "raw_num_points": 220,
                "evaluated_num_points": 220,
                "is_truncated_evaluation": 0.0,
            }
        },
    )

    assert report["evaluation"]["threshold"] == 0.072061
    assert report["evaluation"]["unique_label_count"] == -1
    assert report["evaluation"]["raw_num_points"] == 220
    assert report["evaluation"]["evaluated_num_points"] == 220
    assert report["evaluation"]["is_truncated_evaluation"] is False


def test_forensic_markdown_conclusion_is_dataset_aware_for_truncated_smd() -> None:
    from scripts.forensic_audit_run import _build_forensic_markdown

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
    from scripts.forensic_audit_run import _build_forensic_markdown

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

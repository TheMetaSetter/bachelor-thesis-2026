from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from scripts.analysis.kl_divergence_anomaly_archive import parse_arguments
from scripts.ops.forensic_audit_run import build_forensic_audit_log_path
from src.analysis.evaluation_protocol_audit import build_protocol_audit_log_path
from src.core.log_paths import dated_log_directory


def test_dated_log_directory_uses_iso_date() -> None:
    assert dated_log_directory("research", current_date=date(2026, 11, 3)) == Path(
        "documents/logs/2026-11-03/research"
    )


def test_protocol_audit_log_path_uses_iso_date() -> None:
    assert build_protocol_audit_log_path(
        experiment_name="evaluation-metrics-test",
        current_date=date(2026, 11, 3),
    ) == Path(
        "documents/logs/2026-11-03/research/"
        "evaluation-metrics-test__evaluation_protocol_audit.md"
    )


def test_forensic_audit_log_path_uses_iso_date() -> None:
    assert build_forensic_audit_log_path(
        experiment_name="smd-smoke",
        current_date=date(2026, 11, 3),
    ) == Path("documents/logs/2026-11-03/research/smd-smoke__forensic_audit.md")


def test_kl_analysis_default_output_uses_iso_date(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["kl_divergence_anomaly_archive.py"])

    assert parse_arguments().output_dir == (
        Path("documents/logs")
        / date.today().isoformat()
        / "research"
        / "anomaly_archive_kl"
    )


def test_analysis_and_visualization_outputs_use_iso_date() -> None:
    from scripts.analysis.export_post_anomaly_suffix_counts import (
        OUTPUT_DIR as suffix_dir,
    )
    from scripts.analysis.summarize_anomaly_span_lengths import (
        DEFAULT_OUTPUT_DIR as span_dir,
    )
    from scripts.analysis.summarize_anomaly_span_lengths_helpers import (
        DEFAULT_OUTPUT_DIR as span_helper_dir,
    )
    from scripts.visualization.visualize_smd_benchmark_train_test import (
        DEFAULT_OUTPUT_DIR as benchmark_plot_dir,
    )
    from scripts.visualize_smd_12_classes_w20 import OUTPUT_DIR as class_plot_dir

    log_root = Path(__file__).resolve().parents[2] / "documents" / "logs"
    date_folder = date.today().isoformat()
    assert suffix_dir == log_root / date_folder / "research"
    assert span_dir == log_root / date_folder / "research"
    assert span_helper_dir == log_root / date_folder / "research"
    assert benchmark_plot_dir == log_root / date_folder / "research"
    assert class_plot_dir == log_root / date_folder / "detail"

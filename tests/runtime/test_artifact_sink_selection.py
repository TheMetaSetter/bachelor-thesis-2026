from __future__ import annotations

from pathlib import Path

import pytest

from src.engine.artifact_sinks import (
    KaggleArtifactSink,
    WandbArtifactSink,
    build_artifact_sinks,
    build_output_artifact_sinks,
)
from src.engine.logger import ExperimentLogger


class _FakeLogger:
    def __init__(self) -> None:
        self.logged_files: list[str] = []

    def log_artifact_file(
        self, *, file_path, artifact_name, artifact_type, aliases=None, metadata=None
    ) -> None:
        self.logged_files.append(str(file_path))

    def log_artifact_directory(
        self,
        *,
        directory_path,
        artifact_name,
        artifact_type,
        aliases=None,
        metadata=None,
    ) -> None:
        self.logged_files.append(str(directory_path))


def test_build_artifact_sinks_returns_expected_sink_variants() -> None:
    fake_logger = _FakeLogger()
    wandb_sinks = build_artifact_sinks(
        {"use_wandb": True},
        experiment_logger=fake_logger,
        include_wandb_sink=True,
    )
    kaggle_sinks = build_artifact_sinks(
        {
            "mirror_best_checkpoint_to_kaggle": True,
            "kaggle_dataset_handle": "user/dataset",
        }
    )

    assert any(isinstance(sink, WandbArtifactSink) for sink in wandb_sinks)
    assert any(isinstance(sink, KaggleArtifactSink) for sink in kaggle_sinks)


def test_build_output_artifact_sinks_selects_kaggle_sink_only_for_output_policy() -> (
    None
):
    output_sinks = build_output_artifact_sinks(
        {
            "mirror_output_dir_to_kaggle": True,
            "kaggle_dataset_handle": "user/dataset",
        }
    )

    assert any(isinstance(sink, KaggleArtifactSink) for sink in output_sinks)


def test_experiment_logger_mirrors_output_directory_with_output_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = ExperimentLogger(
        tmp_path,
        experiment_config=None,
        logging_config=None,
        write_run_start_record=False,
        write_resolved_config=False,
    )
    mirrored_paths: list[tuple[str, dict[str, str] | None]] = []

    class _FakeSink:
        def save_file(self, path, metadata=None) -> None:
            return

        def save_directory(self, path, metadata=None) -> None:
            mirrored_paths.append((str(path), metadata))

    monkeypatch.setattr(
        "src.engine.logger.build_output_artifact_sinks",
        lambda *args, **kwargs: [_FakeSink()],
    )

    logger.mirror_output_directory(
        {"mirror_output_dir_to_kaggle": True, "kaggle_dataset_handle": "user/dataset"},
        metadata={"experiment_name": "demo"},
    )

    assert mirrored_paths == [(str(tmp_path), {"experiment_name": "demo"})]


def test_kaggle_artifact_sink_requires_kagglehub_when_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = KaggleArtifactSink(dataset_handle="user/dataset")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.delitem(__import__("sys").modules, "kagglehub", raising=False)

    with pytest.raises(ImportError, match="kagglehub"):
        sink.save_file(checkpoint_path)

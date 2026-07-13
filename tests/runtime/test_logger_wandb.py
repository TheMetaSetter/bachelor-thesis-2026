from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from src.engine.logger import ExperimentLogger


class _FakeArtifact:
    def __init__(self, name: str, type: str, metadata=None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[str] = []
        self.directories: list[str] = []

    def add_file(self, file_path: str, name: str | None = None) -> None:
        self.files.append(name or file_path)

    def add_dir(self, directory_path: str, name: str | None = None) -> None:
        self.directories.append(name or directory_path)


class _FakeRun:
    def __init__(self) -> None:
        self.logged_metrics: list[dict[str, object]] = []
        self.logged_artifacts: list[tuple[_FakeArtifact, list[str] | None]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, metrics: dict[str, object]) -> None:
        self.logged_metrics.append(metrics)

    def log_artifact(self, artifact: _FakeArtifact, aliases=None) -> None:
        self.logged_artifacts.append((artifact, aliases))

    def finish(self) -> None:
        self.finished = True


def test_experiment_logger_logs_metrics_and_artifacts_to_wandb(
    monkeypatch, tmp_path: Path
) -> None:
    fake_run = _FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: fake_run,
        Artifact=_FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    artifact_file = tmp_path / "artifact.json"
    artifact_file.write_text('{"ok": true}', encoding="utf-8")

    logger = ExperimentLogger(
        tmp_path / "outputs",
        experiment_config={
            "experiment_name": "logger-test",
            "task": {"task_name": "multitask_tsad"},
        },
        logging_config={
            "use_wandb": True,
            "wandb_project": "bachelor-thesis-2026",
            "wandb_mode": "offline",
            "wandb_job_type": "train",
        },
    )

    logger.log_metrics({"train/loss": 1.0})
    logger.log_summary({"run/checkpoint": "best.pt"})
    logger.log_artifact_file(
        file_path=artifact_file,
        artifact_name="logger-test-artifact",
        artifact_type="metrics",
        aliases=["latest"],
        metadata={"kind": "unit-test"},
    )
    logger.close()

    assert fake_run.logged_metrics
    assert fake_run.summary["run/checkpoint"] == "best.pt"
    assert fake_run.logged_artifacts[0][0].name == "logger-test-artifact"
    assert fake_run.logged_artifacts[0][1] == ["latest"]
    assert fake_run.finished is True


def test_experiment_logger_quiet_terminal_suppresses_console_noise(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    fake_run = _FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: fake_run,
        Artifact=_FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = ExperimentLogger(
        tmp_path / "outputs",
        experiment_config={
            "experiment_name": "logger-test",
            "task": {"task_name": "multitask_tsad"},
        },
        logging_config={
            "use_wandb": True,
            "wandb_project": "bachelor-thesis-2026",
            "wandb_mode": "offline",
            "wandb_job_type": "train",
            "quiet_terminal": True,
        },
        quiet_terminal=True,
    )

    logger.log_metrics({"train/loss": 1.0})
    logger.close()

    captured = capsys.readouterr()

    assert captured.out == ""
    assert fake_run.logged_metrics

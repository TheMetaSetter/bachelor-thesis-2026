from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.core.console import console_print


class ArtifactSink(Protocol):
    def save_file(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None: ...

    def save_directory(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None: ...


@dataclass
class NoOpArtifactSink:
    def save_file(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        return

    def save_directory(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        return


@dataclass
class WandbArtifactSink:
    experiment_logger: Any
    artifact_type: str = "checkpoint"

    def save_file(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        path_obj = Path(path)
        console_print(
            "WANDB",
            "Logging file artifact to W&B",
            path=path_obj,
            artifact_type=self.artifact_type,
        )
        self.experiment_logger.log_artifact_file(
            file_path=path_obj,
            artifact_name=path_obj.stem,
            artifact_type=self.artifact_type,
            aliases=["latest"],
            metadata=metadata,
        )

    def save_directory(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        path_obj = Path(path)
        console_print(
            "WANDB",
            "Logging directory artifact to W&B",
            path=path_obj,
            artifact_type=self.artifact_type,
        )
        self.experiment_logger.log_artifact_directory(
            directory_path=path_obj,
            artifact_name=path_obj.name,
            artifact_type=self.artifact_type,
            aliases=["latest"],
            metadata=metadata,
        )


@dataclass
class KaggleArtifactSink:
    dataset_handle: str
    version_notes: str = "Automated checkpoint update"

    def _upload_directory(self, directory_path: Path) -> None:
        console_print(
            "WANDB",
            "Uploading directory to Kaggle artifact dataset",
            directory_path=directory_path,
            dataset_handle=self.dataset_handle,
        )
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is not installed. Install it before enabling Kaggle artifact mirroring."
            ) from exc

        kagglehub.dataset_upload(
            self.dataset_handle,
            str(directory_path),
            version_notes=self.version_notes,
        )

    def save_file(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        self._upload_directory(Path(path).parent)

    def save_directory(
        self, path: str | Path, metadata: dict[str, Any] | None = None
    ) -> None:
        self._upload_directory(Path(path))


def _build_kaggle_sink(logging_config: dict[str, Any]) -> KaggleArtifactSink:
    console_print(
        "WANDB",
        "Building Kaggle artifact sink",
        dataset_handle=logging_config["kaggle_dataset_handle"],
    )
    return KaggleArtifactSink(
        dataset_handle=logging_config["kaggle_dataset_handle"],
        version_notes=logging_config.get(
            "kaggle_version_notes",
            "Automated checkpoint update",
        ),
    )


def build_artifact_sinks(
    logging_config: dict[str, Any] | None,
    *,
    experiment_logger: Any | None = None,
    include_wandb_sink: bool = False,
) -> list[ArtifactSink]:
    if not logging_config:
        console_print(
            "WANDB", "No logging config provided for checkpoint artifact sinks"
        )
        return []

    artifact_sinks: list[ArtifactSink] = []
    if (
        include_wandb_sink
        and logging_config.get("use_wandb", False)
        and experiment_logger is not None
    ):
        artifact_sinks.append(WandbArtifactSink(experiment_logger=experiment_logger))
    if logging_config.get("mirror_best_checkpoint_to_kaggle", False):
        artifact_sinks.append(_build_kaggle_sink(logging_config))
    console_print(
        "WANDB",
        "Built checkpoint artifact sinks",
        sink_types=[type(artifact_sink).__name__ for artifact_sink in artifact_sinks],
    )
    return artifact_sinks


def build_output_artifact_sinks(
    logging_config: dict[str, Any] | None,
    *,
    experiment_logger: Any | None = None,
    include_wandb_sink: bool = False,
) -> list[ArtifactSink]:
    if not logging_config:
        console_print("WANDB", "No logging config provided for output artifact sinks")
        return []

    artifact_sinks: list[ArtifactSink] = []
    if (
        include_wandb_sink
        and logging_config.get("use_wandb", False)
        and experiment_logger is not None
    ):
        artifact_sinks.append(
            WandbArtifactSink(
                experiment_logger=experiment_logger, artifact_type="run-output"
            )
        )
    if logging_config.get("mirror_output_dir_to_kaggle", False):
        artifact_sinks.append(_build_kaggle_sink(logging_config))
    console_print(
        "WANDB",
        "Built output artifact sinks",
        sink_types=[type(artifact_sink).__name__ for artifact_sink in artifact_sinks],
    )
    return artifact_sinks

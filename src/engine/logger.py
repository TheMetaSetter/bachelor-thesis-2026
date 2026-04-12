from __future__ import annotations
"""Experiment logging for metrics and resolved configs.

This logger is intentionally small: it writes a JSONL metrics stream, persists
the resolved experiment config, and optionally mirrors metrics to Weights &
Biases when that is explicitly enabled in config.
"""

import json
from pathlib import Path
from typing import Any

from src.core.console import console_print
from src.engine.artifact_sinks import build_artifact_sinks, build_output_artifact_sinks


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        experiment_config: dict[str, Any] | None = None,
        logging_config: dict[str, Any] | None = None,
        *,
        write_run_start_record: bool = True,
        write_resolved_config: bool = True,
    ) -> None:
        # Persisting the fully resolved config next to the metrics makes each run
        # easier to reproduce without re-deriving override composition by hand.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.resolved_config_path = self.output_dir / "resolved_experiment_config.json"
        self._wandb_run = None
        console_print(
            "WANDB",
            "Initializing experiment logger",
            output_dir=self.output_dir,
            metrics_path=self.metrics_path,
            resolved_config_path=self.resolved_config_path,
        )

        if experiment_config is not None and write_resolved_config:
            self.resolved_config_path.write_text(
                json.dumps(experiment_config, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            console_print(
                "WANDB",
                "Wrote resolved experiment config",
                path=self.resolved_config_path,
                experiment_name=experiment_config.get("experiment_name"),
            )
        if experiment_config is not None and write_run_start_record:
            run_start_record = {
                "event": "run_start",
                "experiment_name": experiment_config.get("experiment_name"),
                "task_name": experiment_config.get("task", {}).get("task_name"),
                "anomaly_families": experiment_config.get("task", {}).get("anomaly_families"),
            }
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run_start_record, sort_keys=True) + "\n")
            console_print(
                "WANDB",
                "Wrote run start record",
                metrics_path=self.metrics_path,
                run_start_record=run_start_record,
            )

        if logging_config and logging_config.get("use_wandb", False):
            # W&B remains opt-in so local experimentation keeps the same codepath
            # whether remote tracking is enabled or not.
            try:
                import wandb
            except ImportError as exc:
                raise ValueError("Weights & Biases logging was enabled but wandb is not installed") from exc

            self._wandb_run = wandb.init(
                project=logging_config["wandb_project"],
                entity=logging_config.get("wandb_entity"),
                mode=logging_config.get("wandb_mode", "offline"),
                dir=str(self.output_dir),
                config=experiment_config,
                tags=logging_config.get("wandb_tags"),
                name=logging_config.get("wandb_run_name", experiment_config.get("experiment_name") if experiment_config else None),
                job_type=logging_config.get("wandb_job_type"),
            )
            console_print(
                "WANDB",
                "Initialized W&B run",
                project=logging_config["wandb_project"],
                entity=logging_config.get("wandb_entity"),
                mode=logging_config.get("wandb_mode", "offline"),
                run_name=logging_config.get("wandb_run_name"),
                job_type=logging_config.get("wandb_job_type"),
            )
            if experiment_config is not None and write_run_start_record:
                self._wandb_run.log(
                    {
                        "event/run_start": 1,
                        "event/experiment_name": experiment_config.get("experiment_name"),
                        "event/task_name": experiment_config.get("task", {}).get("task_name"),
                    }
                )

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        serializable_metrics = json.dumps(metrics, sort_keys=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(serializable_metrics + "\n")
        console_print("WANDB", "Logged metrics to JSONL", metrics_path=self.metrics_path, metrics=metrics)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
            console_print("WANDB", "Logged metrics to W&B", metrics=metrics)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._wandb_run is None:
            console_print("WANDB", "Skipping W&B summary logging because no run is active", summary=summary)
            return
        for key, value in summary.items():
            self._wandb_run.summary[key] = value
        console_print("WANDB", "Updated W&B summary", summary=summary)

    def log_artifact_file(
        self,
        *,
        file_path: str | Path,
        artifact_name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._wandb_run is None:
            console_print("WANDB", "Skipping file artifact logging because no W&B run is active", file_path=file_path)
            return

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file does not exist: {path}")

        import wandb

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata,
        )
        artifact.add_file(str(path), name=path.name)
        self._wandb_run.log_artifact(artifact, aliases=aliases)
        console_print(
            "WANDB",
            "Logged file artifact to W&B",
            file_path=path,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            aliases=aliases,
        )

    def log_artifact_directory(
        self,
        *,
        directory_path: str | Path,
        artifact_name: str,
        artifact_type: str,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._wandb_run is None:
            console_print(
                "WANDB",
                "Skipping directory artifact logging because no W&B run is active",
                directory_path=directory_path,
            )
            return

        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact directory does not exist: {path}")

        import wandb

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata,
        )
        artifact.add_dir(str(path), name=path.name)
        self._wandb_run.log_artifact(artifact, aliases=aliases)
        console_print(
            "WANDB",
            "Logged directory artifact to W&B",
            directory_path=path,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            aliases=aliases,
        )

    def close(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            console_print("WANDB", "Closed W&B run")

    def build_artifact_sinks(self, logging_config: dict[str, Any] | None) -> list[Any]:
        console_print("WANDB", "Building artifact sinks for checkpoint logging")
        return build_artifact_sinks(
            logging_config,
            experiment_logger=self,
            include_wandb_sink=False,
        )

    def mirror_output_directory(
        self,
        logging_config: dict[str, Any] | None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        console_print(
            "WANDB",
            "Mirroring output directory through artifact sinks",
            output_dir=self.output_dir,
            metadata=metadata,
        )
        for artifact_sink in build_output_artifact_sinks(
            logging_config,
            experiment_logger=self,
            include_wandb_sink=False,
        ):
            artifact_sink.save_directory(self.output_dir, metadata=metadata)

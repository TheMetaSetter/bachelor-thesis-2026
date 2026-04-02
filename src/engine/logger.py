from __future__ import annotations
"""Experiment logging for metrics and resolved configs.

This logger is intentionally small: it writes a JSONL metrics stream, persists
the resolved experiment config, and optionally mirrors metrics to Weights &
Biases when that is explicitly enabled in config.
"""

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        output_dir: str | Path,
        experiment_config: dict[str, Any] | None = None,
        logging_config: dict[str, Any] | None = None,
    ) -> None:
        # Persisting the fully resolved config next to the metrics makes each run
        # easier to reproduce without re-deriving override composition by hand.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.resolved_config_path = self.output_dir / "resolved_experiment_config.json"
        self._wandb_run = None

        if experiment_config is not None:
            self.resolved_config_path.write_text(
                json.dumps(experiment_config, indent=2, sort_keys=True),
                encoding="utf-8",
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
                name=logging_config.get("wandb_run_name"),
            )

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        serializable_metrics = json.dumps(metrics, sort_keys=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(serializable_metrics + "\n")
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def close(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()

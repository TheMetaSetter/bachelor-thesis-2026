from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.core.console import console_print
from src.engine.artifact_sinks import ArtifactSink


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        artifact_sinks: list[ArtifactSink] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_sinks = artifact_sinks or []
        console_print(
            "CHECKPOINT",
            "Initialized checkpoint manager",
            checkpoint_dir=self.checkpoint_dir,
            num_artifact_sinks=len(self.artifact_sinks),
        )

    def save_checkpoint(
        self,
        checkpoint_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epoch: int,
        metric_history: list[dict[str, Any]],
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler_state,
            "config": config,
            "epoch": epoch,
            "metric_history": metric_history,
        }
        if scheduler is not None:
            checkpoint_payload["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state is not None:
            checkpoint_payload["extra_state"] = extra_state
        console_print(
            "CHECKPOINT",
            "Saving checkpoint",
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            epoch=epoch,
            metric_history_length=len(metric_history),
            has_scheduler_state=scheduler is not None,
            has_extra_state=extra_state is not None,
        )
        torch.save(checkpoint_payload, checkpoint_path)
        for artifact_sink in self.artifact_sinks:
            console_print(
                "CHECKPOINT",
                "Sending checkpoint to artifact sink",
                checkpoint_path=checkpoint_path,
                artifact_sink_type=type(artifact_sink).__name__,
            )
            artifact_sink.save_file(
                checkpoint_path,
                metadata={
                    "epoch": epoch,
                    "checkpoint_name": checkpoint_name,
                    "experiment_name": config.get("experiment_name"),
                },
            )
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        console_print(
            "CHECKPOINT",
            "Loading checkpoint",
            checkpoint_path=checkpoint_path,
            strict=strict,
        )
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_result = model.load_state_dict(
            loaded_checkpoint["model_state_dict"],
            strict=strict,
        )
        if not strict and hasattr(load_result, "missing_keys"):
            console_print(
                "CHECKPOINT",
                "Loaded checkpoint with compatibility mode",
                missing_keys=list(load_result.missing_keys),
                unexpected_keys=list(load_result.unexpected_keys),
            )
        if hasattr(model, "load_checkpoint_extra_state"):
            model.load_checkpoint_extra_state(loaded_checkpoint.get("extra_state"))
        if optimizer is not None:
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in loaded_checkpoint:
            scheduler.load_state_dict(loaded_checkpoint["scheduler_state_dict"])
        console_print(
            "CHECKPOINT",
            "Loaded checkpoint",
            checkpoint_path=checkpoint_path,
            epoch=loaded_checkpoint.get("epoch"),
            metric_history_length=len(loaded_checkpoint.get("metric_history", [])),
            has_scheduler_state="scheduler_state_dict" in loaded_checkpoint,
        )
        return loaded_checkpoint

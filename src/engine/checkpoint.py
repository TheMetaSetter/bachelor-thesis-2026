from __future__ import annotations

import hashlib
import json
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

    @staticmethod
    def _stable_json_digest(value: Any) -> str:
        serialized_value = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(serialized_value.encode("utf-8")).hexdigest()

    @classmethod
    def _build_checkpoint_metadata(
        cls,
        *,
        config: dict[str, Any],
        epoch: int,
        metric_history: list[dict[str, Any]],
        extra_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_config = dict(config.get("model", {}))
        task_config = dict(config.get("task", {}))
        metadata = {
            "schema_version": 3,
            "experiment_name": config.get("experiment_name"),
            "model_name": model_config.get("model_name"),
            "task_name": task_config.get("task_name"),
            "seed": config.get("seed"),
            "epoch": int(epoch),
            "metric_history_length": len(metric_history),
            "config_sha256": cls._stable_json_digest(config),
            "model_config_sha256": cls._stable_json_digest(model_config),
            "task_config_sha256": cls._stable_json_digest(task_config),
            "memory_label_source": model_config.get("discrete_memory_label_source"),
            "stochastic_inference": model_config.get("stochastic_inference"),
            "monte_carlo_samples": model_config.get("monte_carlo_samples"),
            "continuous_temperature": model_config.get("continuous_temperature"),
            "discrete_temperature": model_config.get("discrete_temperature"),
            "variance_correction": model_config.get("variance_correction"),
            "return_mc_samples": model_config.get("return_mc_samples"),
            "sample_retention_policy": model_config.get("sample_retention_policy"),
        }
        if extra_state is not None:
            metadata["extra_state_sha256"] = cls._stable_json_digest(extra_state)
        return metadata

    @classmethod
    def _validate_checkpoint_metadata(
        cls,
        checkpoint_metadata: dict[str, Any],
        *,
        config: dict[str, Any],
        epoch: int,
        metric_history: list[dict[str, Any]],
    ) -> None:
        required_keys = {
            "schema_version",
            "experiment_name",
            "model_name",
            "task_name",
            "seed",
            "epoch",
            "metric_history_length",
            "config_sha256",
            "model_config_sha256",
            "task_config_sha256",
            "memory_label_source",
            "stochastic_inference",
            "monte_carlo_samples",
            "continuous_temperature",
            "discrete_temperature",
            "variance_correction",
            "return_mc_samples",
            "sample_retention_policy",
        }
        missing_keys = sorted(required_keys - set(checkpoint_metadata))
        if missing_keys:
            raise ValueError(
                f"checkpoint_metadata is missing required keys: {missing_keys}"
            )
        if int(checkpoint_metadata["schema_version"]) != 3:
            raise ValueError("checkpoint_metadata schema_version must be 3")
        if checkpoint_metadata["epoch"] != int(epoch):
            raise ValueError("checkpoint_metadata epoch does not match payload epoch")
        if checkpoint_metadata["metric_history_length"] != len(metric_history):
            raise ValueError(
                "checkpoint_metadata metric_history_length does not match payload"
            )
        model_config = dict(config.get("model", {}))
        task_config = dict(config.get("task", {}))
        if checkpoint_metadata["experiment_name"] != config.get("experiment_name"):
            raise ValueError("checkpoint_metadata experiment_name does not match config")
        if checkpoint_metadata["model_name"] != model_config.get("model_name"):
            raise ValueError("checkpoint_metadata model_name does not match config")
        if checkpoint_metadata["task_name"] != task_config.get("task_name"):
            raise ValueError("checkpoint_metadata task_name does not match config")
        if checkpoint_metadata["memory_label_source"] != model_config.get(
            "discrete_memory_label_source"
        ):
            raise ValueError(
                "checkpoint_metadata memory_label_source does not match model config"
            )
        if model_config.get("stochastic_inference") is not None and (
            checkpoint_metadata["stochastic_inference"]
            != model_config.get("stochastic_inference")
        ):
            raise ValueError(
                "checkpoint_metadata stochastic_inference does not match model config"
            )
        for field_name in [
            "monte_carlo_samples",
            "continuous_temperature",
            "discrete_temperature",
            "variance_correction",
            "return_mc_samples",
            "sample_retention_policy",
        ]:
            model_value = model_config.get(field_name)
            if model_value is None:
                continue
            if checkpoint_metadata[field_name] != model_value:
                raise ValueError(
                    f"checkpoint_metadata {field_name} does not match model config"
                )
        if checkpoint_metadata["config_sha256"] != cls._stable_json_digest(config):
            raise ValueError("checkpoint_metadata config_sha256 does not match config")
        if checkpoint_metadata["model_config_sha256"] != cls._stable_json_digest(
            model_config
        ):
            raise ValueError(
                "checkpoint_metadata model_config_sha256 does not match model config"
            )
        if checkpoint_metadata["task_config_sha256"] != cls._stable_json_digest(
            task_config
        ):
            raise ValueError(
                "checkpoint_metadata task_config_sha256 does not match task config"
            )

    def _build_payload(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epoch: int,
        metric_history: list[dict[str, Any]],
        extra_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "scaler_state_dict": scaler_state,
            "config": config,
            "epoch": epoch,
            "metric_history": metric_history,
            "checkpoint_metadata": self._build_checkpoint_metadata(
                config=config,
                epoch=epoch,
                metric_history=metric_history,
                extra_state=extra_state,
            ),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        if extra_state is not None:
            payload["extra_state"] = extra_state
        return payload

    def _sync_artifacts(
        self,
        checkpoint_path: Path,
        checkpoint_name: str,
        config: dict[str, Any],
        epoch: int,
    ) -> None:
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

    def save_checkpoint(
        self,
        checkpoint_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epoch: int,
        metric_history: list[dict[str, Any]],
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_payload = self._build_payload(
            model, optimizer, scheduler, scaler_state, config, epoch,
            metric_history, extra_state,
        )
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
        self._sync_artifacts(checkpoint_path, checkpoint_name, config, epoch)
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
        checkpoint_metadata = loaded_checkpoint.get("checkpoint_metadata")
        if checkpoint_metadata is not None:
            self._validate_checkpoint_metadata(
                checkpoint_metadata,
                config=loaded_checkpoint["config"],
                epoch=int(loaded_checkpoint.get("epoch", 0)),
                metric_history=list(loaded_checkpoint.get("metric_history", [])),
            )
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
        if optimizer is not None and "optimizer_state_dict" in loaded_checkpoint:
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

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epoch: int,
        metric_history: list[dict[str, Any]],
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler_state,
                "config": config,
                "epoch": epoch,
                "metric_history": metric_history,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        return loaded_checkpoint


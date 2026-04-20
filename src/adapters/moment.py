from __future__ import annotations

from typing import Any

import torch

from src.adapters.base import BaseBatchAdapter
from src.data.api import point_labels_to_window_labels


class MomentWindowAdapter(BaseBatchAdapter):
    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-small",
        *,
        task_name: str = "embedding",
        context_length: int = 512,
        moment_model: torch.nn.Module | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.task_name = task_name
        self.context_length = int(context_length)
        self.device = None if device is None else torch.device(device)
        self._moment_model = moment_model

    def _load_model_if_needed(self) -> torch.nn.Module:
        if self._moment_model is not None:
            return self._moment_model
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError(
                "momentfm is not installed. Install it before loading a MOMENT model automatically."
            ) from exc
        moment_model = MOMENTPipeline.from_pretrained(
            self.model_name,
            model_kwargs={"task_name": self.task_name},
        )
        if hasattr(moment_model, "init"):
            moment_model.init()
        if self.device is not None:
            moment_model = moment_model.to(self.device)
        moment_model.eval()
        self._moment_model = moment_model
        return moment_model

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch_x = batch["x"]
        if batch_x.ndim != 3:
            raise ValueError(
                f"Expected batch['x'] with shape [B, L, D], got {tuple(batch_x.shape)}"
            )

        moment_x_enc = batch_x.transpose(1, 2).contiguous()
        observed_window_length = int(moment_x_enc.shape[-1])
        if observed_window_length > self.context_length:
            raise ValueError(
                f"Window length {observed_window_length} exceeds MOMENT context length {self.context_length}"
            )
        if observed_window_length < self.context_length:
            padding_length = self.context_length - observed_window_length
            moment_x_enc = torch.nn.functional.pad(
                moment_x_enc, (0, padding_length), value=0.0
            )

        input_mask = torch.zeros(
            (batch_x.shape[0], self.context_length),
            dtype=torch.float32,
            device=batch_x.device,
        )
        input_mask[:, :observed_window_length] = 1.0
        return {
            "x_enc": moment_x_enc,
            "input_mask": input_mask,
            "observed_window_length": observed_window_length,
            "meta": list(batch.get("meta", [])),
            "point_labels": batch.get("point_labels"),
        }

    def forward_prepared(self, prepared_batch: dict[str, Any]) -> Any:
        moment_model = self._load_model_if_needed()
        return moment_model(
            x_enc=prepared_batch["x_enc"],
            input_mask=prepared_batch["input_mask"],
        )

    def postprocess_outputs(
        self,
        model_outputs: Any,
        prepared_batch: dict[str, Any],
    ) -> dict[str, Any]:
        point_labels = prepared_batch.get("point_labels")
        window_labels = None
        if point_labels is not None:
            window_labels = point_labels_to_window_labels(point_labels).detach().cpu()
        return {
            "embeddings": model_outputs.embeddings,
            "window_labels": window_labels,
            "meta": prepared_batch["meta"],
            "observed_window_length": prepared_batch["observed_window_length"],
        }

    def embed_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared_batch = self.prepare_batch(batch)
        model_outputs = self.forward_prepared(prepared_batch)
        return self.postprocess_outputs(model_outputs, prepared_batch)

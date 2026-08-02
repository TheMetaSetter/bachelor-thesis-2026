"""Opt-in timing logs for the THESIS online TTA runtime."""

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

import torch


Result = TypeVar("Result")


class OnlineTtaTimingLogger:
    """Print each causal-window component duration only when enabled."""

    def __init__(self, *, enabled: bool, device: str) -> None:
        self.enabled = enabled
        self.device = device
        self._window_label = "window=unknown"

    def set_window(self, batch: dict[str, Any]) -> None:
        meta = batch["meta"][0]
        start_index = meta.get("start_index", "?")
        end_index = meta.get("end_index", "?")
        self._window_label = (
            f"entity={meta['entity_id']} window=[{start_index},{end_index})"
        )

    def measure(self, component: str, action: Callable[[], Result]) -> Result:
        if not self.enabled:
            return action()
        self._synchronize_cuda()
        start_time = perf_counter()
        result = action()
        self._synchronize_cuda()
        elapsed_ms = (perf_counter() - start_time) * 1_000
        print(
            f"[online-tta-timing] {self._window_label} "
            f"component={component} elapsed_ms={elapsed_ms:.3f}",
            flush=True,
        )
        return result

    def _synchronize_cuda(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)

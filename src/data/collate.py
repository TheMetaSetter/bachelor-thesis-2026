from __future__ import annotations

from typing import Any

import torch

from src.core.console import console_print, summarize_batch
from src.core.contracts import validate_batch, validate_window


def collate_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    console_print("DATA", "Collating windows into batch", num_windows=len(windows))
    for window in windows:
        validate_window(window)

    batch = {
        "x": torch.stack([window["x"] for window in windows], dim=0),
        "point_labels": None,
        "mask": None,
        "timestamps": None,
        "meta": [dict(window["meta"]) for window in windows],
    }

    if all(window["point_labels"] is not None for window in windows):
        batch["point_labels"] = torch.stack([window["point_labels"] for window in windows], dim=0)
    if all(window["mask"] is not None for window in windows):
        batch["mask"] = torch.stack([window["mask"] for window in windows], dim=0)
    if all(window["timestamps"] is not None for window in windows):
        batch["timestamps"] = torch.stack([window["timestamps"] for window in windows], dim=0)

    validate_batch(batch)
    console_print("DATA", "Built collated batch", **summarize_batch(batch))
    return batch

from __future__ import annotations

"""Shared demo state objects.

₍^. .^₎⟆ Demo flow

saved benchmark artifacts
  -> loader
  -> replay state
  -> plotting/app layer
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OfflineReplayState:
    report_path: Path
    method: str
    variant: str
    entity_id: str
    seed: int
    threshold: float
    threshold_source: str
    point_rule: str
    smoothing_rule: str
    raw_values: np.ndarray
    point_scores: np.ndarray
    point_labels: np.ndarray
    predicted_mask: np.ndarray
    covered_point_mask: np.ndarray
    threshold_artifact: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class OnlineReplayState:
    report_path: Path
    method: str
    variant: str
    entity_id: str
    seed: int
    threshold: float
    threshold_source: str
    point_rule: str
    smoothing_rule: str
    raw_values: np.ndarray
    score_indices: np.ndarray
    raw_point_scores: np.ndarray
    ewma_point_scores: np.ndarray
    predicted_mask: np.ndarray
    records: list[dict[str, Any]]
    threshold_artifact: dict[str, Any]
    metrics: dict[str, Any]

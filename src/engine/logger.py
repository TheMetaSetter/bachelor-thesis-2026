from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        serializable_metrics = json.dumps(metrics, sort_keys=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(serializable_metrics + "\n")


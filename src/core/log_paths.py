from __future__ import annotations

from datetime import date
from pathlib import Path


def dated_log_directory(
    task_type: str,
    *,
    current_date: date | None = None,
) -> Path:
    resolved_date = current_date or date.today()
    return Path("documents") / "logs" / resolved_date.isoformat() / task_type

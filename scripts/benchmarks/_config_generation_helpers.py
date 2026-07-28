from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def entity_token(entity_id: str) -> str:
    return entity_id.replace("-", "_")


def write_yaml_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

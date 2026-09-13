from __future__ import annotations

from pathlib import Path

import yaml

from src.core.artifact_naming import is_valid_wandb_smoke_run_name


def test_every_smoke_experiment_config_enables_online_wandb() -> None:
    config_roots = (Path("configs/experiment"), Path("scripts/configs/experiment"))
    smoke_config_paths = sorted(
        path
        for config_root in config_roots
        for path in config_root.rglob("*.yaml")
        if "smoke" in path.as_posix().lower()
        or "benchmark_smoke" in path.as_posix().lower()
    )

    assert smoke_config_paths

    for config_path in smoke_config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        logging_config = config.get("logging")

        assert logging_config is not None, config_path
        assert logging_config["use_wandb"] is True, config_path
        assert logging_config["wandb_project"] == "bachelor-thesis-2026", config_path
        assert logging_config["wandb_mode"] == "online", config_path
        assert is_valid_wandb_smoke_run_name(logging_config.get("wandb_run_name")), (
            config_path
        )

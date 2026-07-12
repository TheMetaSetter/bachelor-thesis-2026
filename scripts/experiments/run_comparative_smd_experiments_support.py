from __future__ import annotations

"""Sequential mixed-method launcher for comparative SMD experiments.

This script intentionally stays as a thin subprocess coordinator. It validates
resolved configs, derives the run family from config semantics, writes a durable
manifest, then delegates the actual work to existing entrypoints.
"""

#            ♪      ♫        ♬
#     .----------------------------------.
#    /  __________________________________\
#   /__/___________________________________\
#   |                                      |
#   |           GRAND PIANO                |
#   |______________________________________|
#   |  | |█| |█| | |█| |█| |█| | |█| |█| | |
#   |  | |█| |█| | |█| |█| |█| | |█| |█| | |
#   |  |_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_| |
#   |  | | | | | | | | | | | | | | | | | | |
#   |__|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
#      \                                  /
#       \________________________________/
#              ||               ||
#              ||               ||
#              ||               ||
#            __||__           __||__
#           /______\         /______\

#      ♪ "Every key hides a new idea." ♪

# (´▽`♡) IMPORTS - Essential Dependencies
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

import yaml

from src.core.config import load_experiment_config


# (๑•́ ω •̀๑) CONSTANTS - Runtime Configuration
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_BASELINE_MODEL_NAMES = {"redlamp_baseline"}


# ≧◡≦ UTILITY FUNCTIONS - Timestamps & Formatting
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～


def _utc_now_iso() -> str:  # (´◡`) Returns current UTC in ISO format
    """Generate ISO 8601 timestamp with Z suffix for UTC timezone."""
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


# ٩(◕‿◕｡)۶ CLI INTERFACE - Argument Parsing
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～


def parse_args() -> argparse.Namespace:  # └(★ω★)┘ Build CLI parser
    """Parse command-line arguments for experiment configuration.

    Supported flags:
      --config-paths: Paths to experiment config files (required)
      --smoke-config-paths: Quick test configs (optional)
      --report-dir: Output directory for reports
      --data-num-workers-override: Override thread count
      --dry-run: Show what would run without executing
      --preflight-only: Validate configs without running
      --skip-completed: Resume interrupted experiments
    """
    parser = argparse.ArgumentParser(
        description="Run mixed baseline and thesis comparative SMD experiments"
    )
    parser.add_argument("--config-paths", nargs="+", required=True)
    parser.add_argument("--smoke-config-paths", nargs="*", default=[])
    parser.add_argument(
        "--report-dir",
        default="outputs/comparative_smd_reports/default",
    )
    parser.add_argument(
        "--data-num-workers-override",
        type=int,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


# (´•ω•̥`) PATH RESOLUTION - Normalize File Paths
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Convert relative paths to absolute, resolve symlinks
# Impact: Ensures configs are found from any working directory


def normalize_config_path(config_path: str | Path) -> Path:  # ≧★≦ Make absolute
    """Convert config path to absolute canonical form.

    Args:
        config_path: Relative or absolute path to config file

    Returns:
        Resolved absolute Path object
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def resolve_dataset_root(
    resolved_experiment_config: dict[str, Any],
) -> Path:  # (๑♡⌓♡๑) Dataset paths
    """Extract and normalize dataset root path from config.

    Args:
        resolved_experiment_config: Resolved experiment configuration dict

    Returns:
        Absolute path to dataset directory
    """
    dataset_root = Path(str(resolved_experiment_config["data"]["root_dir"]))
    if not dataset_root.is_absolute():
        dataset_root = REPOSITORY_ROOT / dataset_root
    return dataset_root.resolve()


# (✿◠‿◠) VALIDATION LAYER - Integrity Checks
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Catch configuration errors early before execution
# Impact: Prevents silent failures and corrupted results


def validate_dataset_roots(
    resolved_experiment_configs: list[dict[str, Any]],
) -> None:  # (´∀｀)♡ Check datasets exist
    """Verify all dataset roots are accessible.

    Raises:
        FileNotFoundError: If any dataset directory doesn't exist
    """
    for resolved_experiment_config in resolved_experiment_configs:
        dataset_root = resolve_dataset_root(resolved_experiment_config)
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist for {resolved_experiment_config['experiment_name']}: {dataset_root}"
            )


def _normalize_artifact_path(
    path_like: str | Path,
) -> Path:  # (づ｡◕‿‿◕｡)づ Artifact paths
    """Convert artifact path to absolute form.

    Args:
        path_like: Artifact path (output dir, checkpoint, etc)

    Returns:
        Resolved absolute path
    """
    path = Path(path_like)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def validate_unique_artifact_paths(
    resolved_experiment_configs: list[dict[str, Any]],
) -> None:  # ≧★≦ No duplicate paths allowed
    """Check for path collisions across experiments.

    Ensures each experiment has unique output_dir and checkpoint_dir
    to prevent overwriting results.

    Raises:
        ValueError: If duplicate paths detected
    """
    seen_output_dirs: dict[Path, str] = {}
    seen_checkpoint_dirs: dict[Path, str] = {}
    for resolved_experiment_config in resolved_experiment_configs:
        experiment_name = str(resolved_experiment_config["experiment_name"])
        output_dir = Path(str(resolved_experiment_config["output_dir"]))
        checkpoint_dir = Path(str(resolved_experiment_config["checkpoint_dir"]))
        if output_dir in seen_output_dirs:
            raise ValueError(
                f"Duplicate output_dir detected: {output_dir} is shared by "
                f"{seen_output_dirs[output_dir]} and {experiment_name}"
            )
        if checkpoint_dir in seen_checkpoint_dirs:
            raise ValueError(
                f"Duplicate checkpoint_dir detected: {checkpoint_dir} is shared by "
                f"{seen_checkpoint_dirs[checkpoint_dir]} and {experiment_name}"
            )
        seen_output_dirs[output_dir] = experiment_name
        seen_checkpoint_dirs[checkpoint_dir] = experiment_name


# (✿◠‿◠) RUN FAMILY CLASSIFICATION - Type Detection
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Route configs to correct training pipeline
# Impact: One config → correct executor (two-stage, three-stage, baseline)


def resolve_stage_family(
    resolved_experiment_config: dict[str, Any],
) -> str:  # (´▽`♡) Classify run type
    """Determine which training pipeline to use.

    Returns:
        One of: 'thesis_two_stage', 'thesis_three_stage', 'baseline_single_stage'

    Raises:
        ValueError: If model type is unsupported
    """
    model_name = str(resolved_experiment_config["model"]["model_name"])
    if model_name == "thesis_multitask" and "two_stage" in resolved_experiment_config:
        return "thesis_two_stage"
    if model_name == "thesis_multitask" and "three_stage" in resolved_experiment_config:
        return "thesis_three_stage"
    if model_name in SUPPORTED_BASELINE_MODEL_NAMES:
        return "baseline_single_stage"
    raise ValueError(
        "Unsupported comparative run family for model "
        f"{model_name}. Expected thesis two-stage, thesis three-stage, or supported baseline."
    )


def _validate_single_entity_contract(
    resolved_experiment_config: dict[str, Any],
) -> str:  # (๑•́ ω •̀๑) One entity per run
    """Enforce requirement: exactly ONE entity_id per experiment.

    This constraint simplifies comparative analysis - each experiment
    focuses on a single system/entity.

    Returns:
        The single entity_id string

    Raises:
        ValueError: If entity_ids list is missing, empty, or has multiple entries
    """
    entity_ids = resolved_experiment_config["data"].get("entity_ids")
    if not isinstance(entity_ids, list) or len(entity_ids) != 1:
        raise ValueError(
            "Comparative SMD runs require exactly one entity_id per experiment config"
        )
    entity_id = entity_ids[0]
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("Comparative SMD entity_id must be a non-empty string")
    return entity_id


# ٩(◕‿◕｡)۶ COMMAND BUILDERS - Executable Construction
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Compose subprocess.run() commands for different run types
# Impact: Each pipeline gets correct training script + args


def _build_thesis_two_stage_commands(
    config_path: Path,
) -> list[list[str]]:  # ≧◡≦ Build two-stage pipeline
    """Create command for two-stage training (pretraining → fine-tuning).

    Returns:
        List of command lists, each ready for subprocess.run()
    """
    return [
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_two_stage_offline_pretraining.py"),
            "--experiment-config",
            str(config_path),
        ]
    ]


def _build_thesis_three_stage_commands(
    config_path: Path,
) -> list[list[str]]:  # (´◡`) Build three-stage pipeline
    """Create command for three-stage training.

    Returns:
        List of command lists, each ready for subprocess.run()
    """
    return [
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_three_stage_offline_pretraining.py"),
            "--experiment-config",
            str(config_path),
        ]
    ]


def _build_baseline_single_stage_commands(
    config_path: Path,
    resolved_experiment_config: dict[str, Any],
) -> list[list[str]]:  # └(★ω★)┘ Build baseline pipeline
    """Create commands for baseline: train THEN evaluate.

    Baseline runs two separate scripts in sequence:
      1. train.py - fits model on training data
      2. evaluate.py - tests on held-out test set

    Returns:
        Two command lists: [train_cmd, evaluate_cmd]
    """
    checkpoint_path = (
        Path(str(resolved_experiment_config["checkpoint_dir"])) / "best.pt"
    )
    return [
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "train.py"),
            "--experiment-config",
            str(config_path),
        ],
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate.py"),
            "--experiment-config",
            str(config_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
    ]


# (づ｡◕‿‿◕｡)づ RUN RECORD ASSEMBLY - Package Metadata
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Create a structured dict with all execution info
# Impact: Single source of truth for each run's configuration


def _build_run_record(
    *,
    config_path: Path,
    resolved_experiment_config: dict[str, Any],
    run_stage: str,
) -> dict[str, Any]:  # (๑♡⌓♡๑) Create structured execution record
    """Build a complete record for one experiment run.

    Combines config metadata, commands, and artifact paths into a
    single dictionary for easy tracking and debugging.

    Args:
        config_path: Path to config file
        resolved_experiment_config: Fully resolved config dict
        run_stage: Either 'smoke' or 'main'

    Returns:
        Dict with run_id, commands, paths, and metadata
    """
    stage_family = resolve_stage_family(resolved_experiment_config)
    entity_id = _validate_single_entity_contract(resolved_experiment_config)
    output_dir = Path(str(resolved_experiment_config["output_dir"]))
    if stage_family == "thesis_two_stage":
        commands = _build_thesis_two_stage_commands(config_path)
    elif stage_family == "thesis_three_stage":
        commands = _build_thesis_three_stage_commands(config_path)
    else:
        commands = _build_baseline_single_stage_commands(
            config_path,
            resolved_experiment_config,
        )

    return {
        "run_id": f"{run_stage}:{resolved_experiment_config['experiment_name']}",
        "run_stage": run_stage,
        "experiment_name": str(resolved_experiment_config["experiment_name"]),
        "stage_family": stage_family,
        "model_name": str(resolved_experiment_config["model"]["model_name"]),
        "device": str(resolved_experiment_config.get("device", "missing")),
        "entity_id": entity_id,
        "seed": int(resolved_experiment_config["seed"]),
        "config_path": str(config_path),
        "original_config_path": str(
            resolved_experiment_config.get("_original_config_path", config_path)
        ),
        "dataset_root": str(resolve_dataset_root(resolved_experiment_config)),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(resolved_experiment_config["checkpoint_dir"]),
        "evaluation_metrics_path": str(output_dir / "evaluation_metrics.json"),
        "evaluation_records_path": str(output_dir / "evaluation_records.json"),
        "evaluation_curves_path": str(output_dir / "evaluation_curves.json"),
        "data_num_workers": resolved_experiment_config["data"].get("num_workers"),
        "data_num_workers_override": resolved_experiment_config.get(
            "_data_num_workers_override"
        ),
        "commands": commands,
    }


# └(★ω★)┘ BATCH LOADING - Prepare Configurations
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Load all configs and create run records in bulk
# Impact: Validates all inputs before ANY experiment runs


def _load_run_records(
    config_paths: list[str | Path],
    run_stage: str,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:  # ≧★≦ Resolve configs into records
    """Load configs and convert to structured run records.

    This is the first validation gate - bad configs fail here,
    preventing partial/corrupted runs.

    Args:
        config_paths: Paths to YAML config files
        run_stage: 'smoke' for quick tests, 'main' for full runs

    Returns:
        Tuple of (run_records, resolved_configs) dicts
    """
    from scripts import run_comparative_smd_experiments as public_runner

    run_records: list[dict[str, Any]] = []
    resolved_experiment_configs: list[dict[str, Any]] = []
    for config_path in config_paths:
        normalized_config_path = normalize_config_path(config_path)
        resolved_experiment_config = public_runner.load_experiment_config(
            normalized_config_path
        )
        resolved_experiment_config["_original_config_path"] = str(
            normalized_config_path
        )
        resolved_experiment_configs.append(resolved_experiment_config)
        run_records.append(
            _build_run_record(
                config_path=normalized_config_path,
                resolved_experiment_config=resolved_experiment_config,
                run_stage=run_stage,
            )
        )
    return run_records, resolved_experiment_configs

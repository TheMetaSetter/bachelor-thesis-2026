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

sys.path.append(str(Path(__file__).parent.parent))

import yaml

from src.core.config import load_experiment_config


# (๑•́ ω •̀๑) CONSTANTS - Runtime Configuration
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
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


def resolve_dataset_root(resolved_experiment_config: dict[str, Any]) -> Path:  # (๑♡⌓♡๑) Dataset paths
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

def validate_dataset_roots(resolved_experiment_configs: list[dict[str, Any]]) -> None:  # (´∀｀)♡ Check datasets exist
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


def _normalize_artifact_path(path_like: str | Path) -> Path:  # (づ｡◕‿‿◕｡)づ Artifact paths
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

def resolve_stage_family(resolved_experiment_config: dict[str, Any]) -> str:  # (´▽`♡) Classify run type
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


def _validate_single_entity_contract(resolved_experiment_config: dict[str, Any]) -> str:  # (๑•́ ω •̀๑) One entity per run
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

def _build_thesis_two_stage_commands(config_path: Path) -> list[list[str]]:  # ≧◡≦ Build two-stage pipeline
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


def _build_thesis_three_stage_commands(config_path: Path) -> list[list[str]]:  # (´◡`) Build three-stage pipeline
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:  # ≧★≦ Resolve configs into records
    """Load configs and convert to structured run records.
    
    This is the first validation gate - bad configs fail here,
    preventing partial/corrupted runs.
    
    Args:
        config_paths: Paths to YAML config files
        run_stage: 'smoke' for quick tests, 'main' for full runs
    
    Returns:
        Tuple of (run_records, resolved_configs) dicts
    """
    run_records: list[dict[str, Any]] = []
    resolved_experiment_configs: list[dict[str, Any]] = []
    for config_path in config_paths:
        normalized_config_path = normalize_config_path(config_path)
        resolved_experiment_config = load_experiment_config(normalized_config_path)
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


# (´∀｀)♡ PARAMETER OVERRIDES - Config Customization
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Create config variants with modified parameters
# Impact: Enables quick tuning without editing source configs

def _materialize_worker_override_configs(
    *,
    run_records: list[dict[str, Any]],
    resolved_experiment_configs: list[dict[str, Any]],
    report_dir: Path,
    data_num_workers_override: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:  # (✿◠‿◠) Generate override variants
    """Create modified configs with custom num_workers setting.
    
    Useful for performance tuning - run same experiment with
    different thread counts without modifying original configs.
    
    Args:
        run_records: Original run records
        resolved_experiment_configs: Original configs
        report_dir: Where to save generated configs
        data_num_workers_override: New num_workers value
    
    Returns:
        Tuple of (modified_run_records, modified_configs)
    """
    generated_configs_dir = report_dir / "generated_configs"
    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    overridden_run_records: list[dict[str, Any]] = []
    overridden_configs: list[dict[str, Any]] = []

    for run_record, resolved_experiment_config in zip(
        run_records,
        resolved_experiment_configs,
        strict=True,
    ):
        overridden_config = json.loads(json.dumps(resolved_experiment_config))
        overridden_config = {
            key: value
            for key, value in overridden_config.items()
            if not str(key).startswith("_")
        }
        for reference_field in [
            "data_config_path",
            "model_config_path",
            "task_config_path",
        ]:
            if reference_field in overridden_config:
                overridden_config[reference_field] = str(
                    normalize_config_path(str(overridden_config[reference_field]))
                )
        overridden_config.setdefault("data_overrides", {})
        overridden_config["data_overrides"]["num_workers"] = int(
            data_num_workers_override
        )
        overridden_config.setdefault("data", {})
        overridden_config["data"]["num_workers"] = int(data_num_workers_override)
        generated_config_path = (
            generated_configs_dir / f"{run_record['run_id'].replace(':', '__')}.yaml"
        )
        generated_config_path.write_text(
            yaml.safe_dump(overridden_config, sort_keys=False),
            encoding="utf-8",
        )
        overridden_configs.append(overridden_config)
        overridden_runtime_config = dict(overridden_config)
        overridden_runtime_config["_original_config_path"] = str(
            run_record["config_path"]
        )
        overridden_runtime_config["_data_num_workers_override"] = int(
            data_num_workers_override
        )
        overridden_run_records.append(
            _build_run_record(
                config_path=generated_config_path,
                resolved_experiment_config=overridden_runtime_config,
                run_stage=str(run_record["run_stage"]),
            )
        )
    return overridden_run_records, overridden_configs


# (´◡`) PLAN ORCHESTRATION - Blueprint Construction
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～

# (´◡`) PLAN ORCHESTRATION - Blueprint Construction
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Assemble complete execution plan with all validations
# Impact: All errors caught before first subprocess spawns

def build_comparative_run_plan(
    *,
    config_paths: list[str | Path],
    smoke_config_paths: list[str | Path],
    report_dir: str | Path,
    data_num_workers_override: int | None = None,
) -> dict[str, Any]:  # ᐛᐛ Build complete execution manifest
    """Create durable execution plan from all configs.
    
    Performs these checks:
      1. Load and validate all configs
      2. Generate override configs if needed
      3. Check for path collisions
      4. Verify datasets exist
      5. Write manifest to disk
    
    Args:
        config_paths: Main experiment configs
        smoke_config_paths: Quick test configs (run first)
        report_dir: Output directory for reports
        data_num_workers_override: Override thread count (optional)
    
    Returns:
        Complete run plan with smoke_runs and main_runs
    """
    smoke_runs, smoke_configs = _load_run_records(smoke_config_paths, run_stage="smoke")
    main_runs, main_configs = _load_run_records(config_paths, run_stage="main")

    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = REPOSITORY_ROOT / resolved_report_dir
    resolved_report_dir.mkdir(parents=True, exist_ok=True)

    if data_num_workers_override is not None:
        smoke_runs, smoke_configs = _materialize_worker_override_configs(
            run_records=smoke_runs,
            resolved_experiment_configs=smoke_configs,
            report_dir=resolved_report_dir,
            data_num_workers_override=data_num_workers_override,
        )
        main_runs, main_configs = _materialize_worker_override_configs(
            run_records=main_runs,
            resolved_experiment_configs=main_configs,
            report_dir=resolved_report_dir,
            data_num_workers_override=data_num_workers_override,
        )

    all_runs = smoke_runs + main_runs
    resolved_experiment_configs = smoke_configs + main_configs
    # ✓ VALIDATION: Check for path collisions
    validate_unique_artifact_paths(resolved_experiment_configs)
    # ✓ VALIDATION: Verify all datasets accessible
    validate_dataset_roots(resolved_experiment_configs)

    run_plan = {
        "created_at_utc": _utc_now_iso(),
        "report_dir": str(resolved_report_dir),
        "manifest_path": str(resolved_report_dir / "comparative_manifest.json"),
        "execution_report_path": str(
            resolved_report_dir / "comparative_execution_report.json"
        ),
        "data_num_workers_override": data_num_workers_override,
        "smoke_runs": smoke_runs,
        "main_runs": main_runs,
    }
    Path(run_plan["manifest_path"]).write_text(
        json.dumps(run_plan, indent=2),
        encoding="utf-8",
    )
    return run_plan


# (๑•́ ω •̀๑) REPORTING - Write Results & Status
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Persist execution status to disk for recovery
# Impact: Enables resume on interruption, post-mortem analysis

def _write_execution_report(
    execution_report_path: str | Path,
    execution_report: dict[str, Any],
) -> None:  # ≧◡≦ Persist report to disk
    """Write execution status to JSON file.
    
    Records: timestamps, completed runs, failures, return codes
    Used for: resuming interrupted experiments, debugging
    """
    Path(execution_report_path).write_text(
        json.dumps(execution_report, indent=2),
        encoding="utf-8",
    )


def _iter_run_groups(
    run_plan: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:  # └(★ω★)┘ Partition runs into groups
    """Organize runs into execution groups.
    
    Returns:
        List of (group_name, runs_list) tuples
        Order: smoke tests FIRST, then main experiments
    """
    return [
        ("smoke", list(run_plan["smoke_runs"])),
        ("main", list(run_plan["main_runs"])),
    ]


def _load_existing_execution_report(
    execution_report_path: str | Path,
) -> dict[str, Any] | None:  # (´▽`♡) Retrieve previous execution report
    """Load report from previous run for resumption.
    
    Returns:
        Parsed report dict, or None if file doesn't exist
    """
    report_path = Path(execution_report_path)
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_has_required_artifacts(run_record: dict[str, Any]) -> bool:  # (๑♡⌓♡๑) Check artifacts exist
    """Verify all output files exist for a completed run.
    
    Checks for: checkpoint, metrics, records, curves, stage reports
    
    Returns:
        True if all expected files present, False otherwise
    """
    checkpoint_path = _normalize_artifact_path(run_record["checkpoint_dir"]) / "best.pt"
    metrics_path = _normalize_artifact_path(run_record["evaluation_metrics_path"])
    records_path = _normalize_artifact_path(run_record["evaluation_records_path"])
    curves_path = _normalize_artifact_path(run_record["evaluation_curves_path"])
    if str(run_record["stage_family"]) == "baseline_single_stage":
        return all(
            path.exists()
            for path in [checkpoint_path, metrics_path, records_path, curves_path]
        )

    if str(run_record["stage_family"]) == "thesis_two_stage":
        stage_execution_report_path = (
            _normalize_artifact_path(run_record["output_dir"])
            / "two_stage"
            / "two_stage_execution_report.json"
        )
    else:
        stage_execution_report_path = (
            _normalize_artifact_path(run_record["output_dir"])
            / "three_stage"
            / "three_stage_execution_report.json"
        )
    return all(
        path.exists()
        for path in [
            checkpoint_path,
            metrics_path,
            records_path,
            curves_path,
            stage_execution_report_path,
        ]
    )


# (๑♡⌓♡๑) EXECUTION ENGINE - Run & Track
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Execute all experiments sequentially with progress tracking
# Impact: Runs experiments, records completion, enables resume on failure

def execute_comparative_run_plan(
    run_plan: dict[str, Any],
    *,
    dry_run: bool,
    skip_completed: bool = False,
) -> dict[str, Any]:  # ٩(◕‿◕｡)۶ Execute experiments sequentially
    """Run all experiments with error handling and status tracking.
    
    Execution flow:
      1. Check for previous run (if resuming)
      2. Run smoke tests first (fail fast)
      3. Run main experiments
      4. Track completion status per run
      5. Write final report to disk
    
    Args:
        run_plan: Complete plan from build_comparative_run_plan()
        dry_run: If True, only show what would run
        skip_completed: If True, skip runs that succeeded before
    
    Returns:
        Execution report with timestamps and results
    
    Raises:
        subprocess.CalledProcessError: If any command fails
    """
    existing_execution_report = (
        _load_existing_execution_report(run_plan["execution_report_path"])
        if skip_completed and not dry_run
        else None
    )
    existing_completed_run_ids = set(
        existing_execution_report.get("completed_run_ids", [])
        if existing_execution_report is not None
        else []
    )
    execution_report = {
        "started_at_utc": _utc_now_iso(),
        "manifest_path": run_plan["manifest_path"],
        "execution_report_path": run_plan["execution_report_path"],
        "dry_run": dry_run,
        "completed_run_ids": [],
        "executed_run_ids": [],
        "skipped_run_ids": [],
        "skip_completed": skip_completed,
        "resumed_from_existing_report": existing_execution_report is not None,
        "status": "running",
    }

    if dry_run:
        # Dry run mode: just validate, don't execute
        execution_report["status"] = "dry_run"
        execution_report["finished_at_utc"] = _utc_now_iso()
        _write_execution_report(run_plan["execution_report_path"], execution_report)
        return execution_report

    try:
        # Execute runs grouped by type (smoke first, then main)
        for _, run_records in _iter_run_groups(run_plan):
            for run_record in run_records:
                # Check if already completed in previous run
                if (
                    skip_completed
                    and run_record["run_id"] in existing_completed_run_ids
                    and _run_has_required_artifacts(run_record)
                ):
                    execution_report["completed_run_ids"].append(run_record["run_id"])
                    execution_report["skipped_run_ids"].append(run_record["run_id"])
                    continue
                # Execute this run
                execution_report["executed_run_ids"].append(run_record["run_id"])
                for command in run_record["commands"]:
                    command_to_run = list(command)
                    # Add --skip-completed flag for multistage runs if resuming
                    if (
                        skip_completed
                        and str(run_record["stage_family"])
                        in {
                            "thesis_two_stage",
                            "thesis_three_stage",
                        }
                        and "--skip-completed" not in command_to_run
                    ):
                        command_to_run.append("--skip-completed")
                    subprocess.run(command_to_run, cwd=REPOSITORY_ROOT, check=True)
                execution_report["completed_run_ids"].append(run_record["run_id"])
    except subprocess.CalledProcessError as error:
        # Failure: record error and re-raise
        execution_report["status"] = "failed"
        execution_report["failed_at_utc"] = _utc_now_iso()
        if execution_report["executed_run_ids"]:
            execution_report["failed_run_id"] = execution_report["executed_run_ids"][-1]
        execution_report["failed_command"] = list(error.cmd)
        execution_report["failed_return_code"] = int(error.returncode)
        _write_execution_report(run_plan["execution_report_path"], execution_report)
        raise

    # Success: update status and write final report
    execution_report["status"] = "completed"
    execution_report["finished_at_utc"] = _utc_now_iso()
    _write_execution_report(run_plan["execution_report_path"], execution_report)
    return execution_report



def _print_run_records(run_records: list[dict[str, Any]], stage_name: str) -> None:  # ≧◡≦ Output metadata
    """Print run metadata to stdout for logging/debugging.
    
    Outputs JSON format for easy parsing by other tools.
    """
    for run_record in run_records:
        print(
            json.dumps(
                {
                    "stage_name": stage_name,
                    "run_id": run_record["run_id"],
                    "stage_family": run_record["stage_family"],
                    "entity_id": run_record["entity_id"],
                    "seed": run_record["seed"],
                    "commands": run_record["commands"],
                }
            )
        )


# (✿◠‿◠) ENTRY POINT - Main Workflow
# ～～～～～～～～～～～～～～～～～～～～～～～～～～～
# Purpose: Orchestrate full pipeline: plan → validate → execute
# Impact: Single entry point for batch experiment runs

def main() -> None:  # └(★ω★)┘ Orchestrate the entire pipeline
    """Main entry point for comparative experiment launcher.
    
    Flow:
      1. Parse CLI arguments
      2. Build complete execution plan with all validations
      3. Print run manifest (for inspection/logging)
      4. Execute all experiments (smoke first, then main)
    """
    args = parse_args()
    run_plan = build_comparative_run_plan(
        config_paths=list(args.config_paths),
        smoke_config_paths=list(args.smoke_config_paths),
        report_dir=args.report_dir,
        data_num_workers_override=args.data_num_workers_override,
    )
    _print_run_records(run_plan["smoke_runs"], stage_name="smoke")
    _print_run_records(run_plan["main_runs"], stage_name="main")

    if args.preflight_only:
        return
    execute_comparative_run_plan(
        run_plan,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()

from scripts.experiments._internal import (
    run_comparative_smd_experiments_support as comparative_support,
)
from scripts.experiments._internal.run_comparative_smd_experiments_support import *  # noqa: F401,F403

_build_baseline_single_stage_commands = (
    comparative_support._build_baseline_single_stage_commands
)
_build_run_record = comparative_support._build_run_record
_build_thesis_three_stage_commands = (
    comparative_support._build_thesis_three_stage_commands
)
_build_thesis_two_stage_commands = comparative_support._build_thesis_two_stage_commands
_load_run_records = comparative_support._load_run_records
_normalize_artifact_path = comparative_support._normalize_artifact_path
_utc_now_iso = comparative_support._utc_now_iso
_validate_single_entity_contract = (
    comparative_support._validate_single_entity_contract
)



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
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:  # (✿◠‿◠) Generate override variants
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


def _run_has_required_artifacts(
    run_record: dict[str, Any],
) -> bool:  # (๑♡⌓♡๑) Check artifacts exist
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
    elif str(run_record["stage_family"]) == "legacy_thesis_three_stage":
        stage_execution_report_path = (
            _normalize_artifact_path(run_record["output_dir"])
            / "three_stage"
            / "three_stage_execution_report.json"
        )
    else:
        raise ValueError(f"Unsupported stage family: {run_record['stage_family']}")
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
                            "legacy_thesis_three_stage",
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


def _print_run_records(
    run_records: list[dict[str, Any]], stage_name: str
) -> None:  # ≧◡≦ Output metadata
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

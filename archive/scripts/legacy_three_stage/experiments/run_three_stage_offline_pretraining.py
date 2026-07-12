from scripts.experiments._internal.run_three_stage_offline_pretraining_shared import *  # noqa: F401,F403
from scripts.experiments._internal.run_three_stage_offline_pretraining_support import *  # noqa: F401,F403



def execute_three_stage_plan(
    manifest: dict[str, Any],
    *,
    dry_run: bool,
    skip_completed: bool = False,
) -> dict[str, Any]:
    command_plan = build_three_stage_execution_commands(manifest)
    manifest_root = _resolve_manifest_root_from_manifest(manifest)
    execution_report_path = manifest_root / "three_stage_execution_report.json"
    planned_stage_names = [
        stage_record["phase_name"] for stage_record in manifest["training_stages"]
    ] + ["evaluation"]
    completed_stage_names: list[str] = []
    executed_stage_names: list[str] = []
    skipped_stage_names: list[str] = []
    existing_execution_report: dict[str, Any] | None = None
    existing_completed_stage_names: set[str] = set()

    if skip_completed and not dry_run and execution_report_path.exists():
        existing_execution_report = json.loads(
            execution_report_path.read_text(encoding="utf-8")
        )
        existing_completed_stage_names = set(
            existing_execution_report.get("completed_stage_names", [])
        )

    execution_report: dict[str, Any] = {
        "experiment_name": manifest["experiment_name"],
        "dry_run": dry_run,
        "status": "dry_run" if dry_run else "completed",
        "started_at_utc": _utc_now_iso(),
        "optimizer_training_phase_names": list(
            manifest.get(
                "optimizer_training_phase_names", _optimizer_training_phase_names()
            )
        ),
        "optimizer_training_total_epochs": int(manifest["total_training_epochs"]),
        "statistical_procedure_names": list(
            manifest.get("statistical_procedure_names", STATISTICAL_PROCEDURE_NAMES)
        ),
        "planned_stage_names": planned_stage_names,
        "executed_stage_names": planned_stage_names
        if dry_run
        else executed_stage_names,
        "completed_stage_names": planned_stage_names
        if dry_run
        else completed_stage_names,
        "skipped_stage_names": skipped_stage_names,
        "skip_completed": skip_completed,
        "resumed_from_existing_report": existing_execution_report is not None,
        "manifest_path": str(manifest_root / "three_stage_manifest.json"),
        "execution_report_path": str(execution_report_path),
        "server_preflight_summary_path": str(
            manifest_root / "server_preflight_summary.json"
        ),
        "stage2_recovery_initialization_checkpoint_path": str(
            manifest_root / "initializations" / "stage2_recovery_init.pt"
        ),
        "evaluation_checkpoint_path": str(manifest["evaluation"]["checkpoint_path"]),
        "training_commands": command_plan["training"],
        "evaluation_command": command_plan["evaluation"],
    }

    try:
        if not dry_run:
            for stage_record, training_command in zip(
                manifest["training_stages"],
                command_plan["training"],
            ):
                phase_name = str(stage_record["phase_name"])
                stage_best_checkpoint_path = Path(
                    str(stage_record["best_checkpoint_path"])
                )
                if (
                    skip_completed
                    and phase_name in existing_completed_stage_names
                    and stage_best_checkpoint_path.exists()
                ):
                    completed_stage_names.append(phase_name)
                    skipped_stage_names.append(phase_name)
                    continue
                executed_stage_names.append(phase_name)
                if phase_name == "stage2_recovery":
                    _prepare_stage2_recovery_initialization_checkpoint(manifest)
                subprocess.run(training_command, cwd=REPOSITORY_ROOT, check=True)
                completed_stage_names.append(phase_name)

            evaluation_metrics_path = manifest_root.parent / "evaluation_metrics.json"
            evaluation_records_path = manifest_root.parent / "evaluation_records.json"
            evaluation_curves_path = manifest_root.parent / "evaluation_curves.json"
            if (
                skip_completed
                and "evaluation" in existing_completed_stage_names
                and evaluation_metrics_path.exists()
                and evaluation_records_path.exists()
                and evaluation_curves_path.exists()
            ):
                completed_stage_names.append("evaluation")
                skipped_stage_names.append("evaluation")
            else:
                executed_stage_names.append("evaluation")
                subprocess.run(
                    command_plan["evaluation"], cwd=REPOSITORY_ROOT, check=True
                )
                completed_stage_names.append("evaluation")

        execution_report["executed_stage_names"] = (
            planned_stage_names if dry_run else executed_stage_names
        )
        execution_report["completed_stage_names"] = (
            planned_stage_names if dry_run else completed_stage_names
        )
        execution_report["finished_at_utc"] = _utc_now_iso()
        execution_report_path.write_text(
            json.dumps(execution_report, indent=2),
            encoding="utf-8",
        )
        return execution_report
    except subprocess.CalledProcessError as error:
        execution_report["status"] = "failed"
        execution_report["executed_stage_names"] = list(executed_stage_names)
        execution_report["completed_stage_names"] = list(completed_stage_names)
        execution_report["failed_stage_name"] = (
            executed_stage_names[-1] if executed_stage_names else None
        )
        execution_report["failed_command"] = list(error.cmd)
        execution_report["failed_returncode"] = int(error.returncode)
        execution_report["failed_at_utc"] = _utc_now_iso()
        execution_report_path.write_text(
            json.dumps(execution_report, indent=2),
            encoding="utf-8",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight the finalized three-stage offline pre-training plan"
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to the three-stage experiment config",
    )
    parser.add_argument(
        "--print-plan-json",
        action="store_true",
        help="Print the resolved three-stage training plan as JSON",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config and print plan without execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest and command plan without running stage subprocesses",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip completed stages when a prior execution report and required artifacts exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.experiment_config)
    training_plan = build_three_stage_training_plan(experiment_config)
    manifest = materialize_three_stage_run_manifest(experiment_config)
    console_print(
        "THREE_STAGE",
        "Validated three-stage training plan",
        experiment_name=experiment_config["experiment_name"],
        total_training_epochs=compute_three_stage_total_training_epochs(
            experiment_config["three_stage"]
        ),
        phases=[phase["phase_name"] for phase in training_plan],
        manifest_path=_stage_manifest_root(experiment_config)
        / "three_stage_manifest.json",
    )
    if args.print_plan_json:
        print(json.dumps(manifest, indent=2))
    if not args.preflight_only:
        execution_report = execute_three_stage_plan(
            manifest,
            dry_run=args.dry_run,
            skip_completed=args.skip_completed,
        )
        console_print(
            "THREE_STAGE",
            "Completed three-stage execution orchestration",
            dry_run=execution_report["dry_run"],
            executed_stage_names=execution_report["executed_stage_names"],
            execution_report_path=_resolve_manifest_root_from_manifest(manifest)
            / "three_stage_execution_report.json",
        )


if __name__ == "__main__":
    main()

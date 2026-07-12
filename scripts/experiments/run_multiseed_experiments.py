from __future__ import annotations

"""Launch multiple offline training experiments with isolated configs.

This script stays intentionally small. It validates that each experiment config
resolves cleanly, that local artifact paths do not collide, and then delegates
actual training to the existing offline training entrypoint.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.core.console import console_print


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multiple offline seed experiments"
    )
    parser.add_argument(
        "--config-paths",
        nargs="+",
        required=True,
        help="One or more experiment config paths for the main runs.",
    )
    parser.add_argument(
        "--smoke-config-paths",
        nargs="*",
        default=[],
        help="Optional smoke config paths to run before the main stage.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="How to execute the main config list.",
    )
    parser.add_argument(
        "--max-concurrent-processes",
        type=int,
        default=1,
        help="Maximum number of concurrent training processes when execution mode is parallel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without starting training processes.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate configs, paths, and dataset roots without launching any processes.",
    )
    return parser.parse_args()


def normalize_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def load_resolved_experiment_configs(
    config_paths: list[str | Path],
) -> list[dict[str, Any]]:
    resolved_experiment_configs: list[dict[str, Any]] = []
    for config_path in config_paths:
        normalized_config_path = normalize_config_path(config_path)
        resolved_experiment_config = load_experiment_config(normalized_config_path)
        resolved_experiment_config["_config_path"] = normalized_config_path
        resolved_experiment_configs.append(resolved_experiment_config)
    return resolved_experiment_configs


def resolve_dataset_root(resolved_experiment_config: dict[str, Any]) -> Path:
    dataset_root = Path(resolved_experiment_config["data"]["root_dir"])
    if not dataset_root.is_absolute():
        dataset_root = REPOSITORY_ROOT / dataset_root
    return dataset_root.resolve()


def validate_dataset_roots(resolved_experiment_configs: list[dict[str, Any]]) -> None:
    for resolved_experiment_config in resolved_experiment_configs:
        dataset_root = resolve_dataset_root(resolved_experiment_config)
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist for {resolved_experiment_config['experiment_name']}: {dataset_root}"
            )


def validate_unique_artifact_paths(
    resolved_experiment_configs: list[dict[str, Any]],
) -> None:
    seen_output_dirs: dict[Path, str] = {}
    seen_checkpoint_dirs: dict[Path, str] = {}
    for resolved_experiment_config in resolved_experiment_configs:
        experiment_name = resolved_experiment_config["experiment_name"]
        output_dir = Path(resolved_experiment_config["output_dir"])
        checkpoint_dir = Path(resolved_experiment_config["checkpoint_dir"])
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


def build_train_command(config_path: str | Path) -> list[str]:
    normalized_config_path = normalize_config_path(config_path)
    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "train.py"),
        "--experiment-config",
        str(normalized_config_path),
    ]


def print_commands(commands: list[list[str]], stage_name: str) -> None:
    for command_index, command in enumerate(commands, start=1):
        console_print(
            "LAUNCH",
            "Prepared training command",
            stage_name=stage_name,
            command_index=command_index,
            command=" ".join(command),
        )


def run_commands_sequentially(commands: list[list[str]], dry_run: bool) -> None:
    print_commands(commands, stage_name="sequential")
    if dry_run:
        return
    for command in commands:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def terminate_running_processes(
    running_processes: list[tuple[list[str], subprocess.Popen[Any]]],
) -> None:
    for _, running_process in running_processes:
        if running_process.poll() is None:
            running_process.terminate()
    for _, running_process in running_processes:
        if running_process.poll() is None:
            running_process.wait(timeout=10)


def run_commands_in_parallel(
    commands: list[list[str]],
    max_concurrent_processes: int,
    dry_run: bool,
) -> None:
    if max_concurrent_processes <= 0:
        raise ValueError("max_concurrent_processes must be a positive integer")

    print_commands(commands, stage_name="parallel")
    if dry_run:
        return

    pending_commands = list(commands)
    running_processes: list[tuple[list[str], subprocess.Popen[Any]]] = []
    while pending_commands or running_processes:
        while pending_commands and len(running_processes) < max_concurrent_processes:
            command = pending_commands.pop(0)
            process = subprocess.Popen(command, cwd=REPOSITORY_ROOT)
            running_processes.append((command, process))

        next_running_processes: list[tuple[list[str], subprocess.Popen[Any]]] = []
        for command, process in running_processes:
            return_code = process.poll()
            if return_code is None:
                next_running_processes.append((command, process))
                continue
            if return_code != 0:
                other_running_processes = [
                    (other_command, other_process)
                    for other_command, other_process in running_processes
                    if other_process is not process
                ]
                terminate_running_processes(other_running_processes)
                raise RuntimeError(
                    f"Parallel command failed with exit code {return_code}: {' '.join(command)}"
                )

        running_processes = next_running_processes
        if pending_commands or running_processes:
            time.sleep(0.2)


def run_command_stage(
    *,
    config_paths: list[str | Path],
    execution_mode: str,
    max_concurrent_processes: int,
    dry_run: bool,
) -> None:
    commands = [build_train_command(config_path) for config_path in config_paths]
    if execution_mode == "sequential":
        run_commands_sequentially(commands, dry_run=dry_run)
        return
    run_commands_in_parallel(
        commands,
        max_concurrent_processes=max_concurrent_processes,
        dry_run=dry_run,
    )


def main() -> None:
    args = parse_args()
    all_config_paths = list(args.smoke_config_paths) + list(args.config_paths)
    resolved_experiment_configs = load_resolved_experiment_configs(all_config_paths)
    validate_unique_artifact_paths(resolved_experiment_configs)
    validate_dataset_roots(resolved_experiment_configs)
    console_print(
        "LAUNCH",
        "Completed experiment preflight validation",
        total_configs=len(resolved_experiment_configs),
        main_configs=len(args.config_paths),
        smoke_configs=len(args.smoke_config_paths),
        execution_mode=args.execution_mode,
        max_concurrent_processes=args.max_concurrent_processes,
        dry_run=args.dry_run,
        preflight_only=args.preflight_only,
    )

    if args.preflight_only:
        return

    if args.smoke_config_paths:
        console_print(
            "LAUNCH",
            "Running smoke stage before main experiments",
            smoke_configs=len(args.smoke_config_paths),
        )
        run_command_stage(
            config_paths=list(args.smoke_config_paths),
            execution_mode="sequential",
            max_concurrent_processes=1,
            dry_run=args.dry_run,
        )

    run_command_stage(
        config_paths=list(args.config_paths),
        execution_mode=args.execution_mode,
        max_concurrent_processes=args.max_concurrent_processes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

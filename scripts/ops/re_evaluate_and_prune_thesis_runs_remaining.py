from __future__ import annotations

"""Thin wrapper for the remaining thesis re-evaluate-and-prune batch.

This script exists so the user can run the 17 remaining combinations with a
short command, while the heavy lifting stays in
`scripts.ops.re_evaluate_and_prune_thesis_runs`.
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_REMAINING_RUN_ROOTS = (
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed36",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed36",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the remaining 17 thesis re-evaluate-and-prune jobs with a "
            "short wrapper command."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        dest="run_roots",
        help=(
            "Optional run root override. If omitted, the default remaining 17 "
            "runs are used."
        ),
    )
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        help="Protocol config for all reruns.",
    )
    parser.add_argument(
        "--python-bin",
        default=".venv/bin/python",
        help="Python interpreter to use for the rerun batch.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/tmux_logs",
        help="Directory where per-run logs are written.",
    )
    parser.add_argument(
        "--keep-compact-traces",
        action="store_true",
        help="Keep compacted traces after verification instead of deleting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running evaluation or pruning.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = _repo_root()
    python_bin = _resolve_path(args.python_bin, repo_root)
    protocol_config = _resolve_path(args.protocol_config, repo_root)
    log_dir = _resolve_path(args.log_dir, repo_root)
    run_roots = (
        [_resolve_path(run_root, repo_root) for run_root in args.run_roots]
        if args.run_roots
        else [
            _resolve_path(run_root, repo_root)
            for run_root in DEFAULT_REMAINING_RUN_ROOTS
        ]
    )

    command = [
        str(python_bin),
        "-m",
        "scripts.ops.re_evaluate_and_prune_thesis_runs",
        "--protocol-config",
        str(protocol_config),
        "--log-dir",
        str(log_dir),
    ]
    if args.keep_compact_traces:
        command.append("--keep-compact-traces")
    if args.dry_run:
        command.append("--dry-run")
    for run_root in run_roots:
        command.extend(["--run-root", str(run_root)])

    print(
        {
            "repo_root": str(repo_root),
            "run_count": len(run_roots),
            "command": command,
        }
    )
    return subprocess.call(command, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())

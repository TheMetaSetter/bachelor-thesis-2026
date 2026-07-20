from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iter_raw_trace_files(root_dir: Path) -> list[Path]:
    trace_files = sorted(root_dir.rglob("*traces.json"))
    return [path for path in trace_files if path.is_file()]


def _has_uq_summary(run_root: Path) -> bool:
    return (run_root / "metrics" / "uq_summary.json").exists()


def prune_raw_trace_artifacts(
    *,
    root_dir: Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    candidates = _iter_raw_trace_files(root_dir)
    deleted: list[str] = []
    skipped_missing_summary: list[str] = []
    skipped_non_matching: list[str] = []
    for path in candidates:
        if path.name == "uq_summary.json":
            skipped_non_matching.append(str(path))
            continue
        if "retention" in path.parts:
            run_root = path.parents[3]
        elif "traces" in path.parts:
            run_root = path.parents[1]
        else:
            skipped_non_matching.append(str(path))
            continue
        if not _has_uq_summary(run_root):
            skipped_missing_summary.append(str(path))
            continue
        if not dry_run:
            path.unlink()
        deleted.append(str(path))
    return {
        "root_dir": str(root_dir),
        "dry_run": dry_run,
        "candidates": len(candidates),
        "deleted_count": len(deleted),
        "deleted": deleted,
        "skipped_missing_summary": skipped_missing_summary,
        "skipped_non_matching": skipped_non_matching,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="outputs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    payload = prune_raw_trace_artifacts(
        root_dir=Path(args.root_dir),
        dry_run=not args.apply,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

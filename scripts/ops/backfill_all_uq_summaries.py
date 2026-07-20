from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.ops.backfill_uq_summary import backfill_uq_summary


def _find_report_paths(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("thesis_offline_benchmark_report.json"))


def backfill_all_uq_summaries(
    *,
    root_dir: Path,
    write_retention_copy: bool = True,
    skip_existing: bool = True,
) -> dict[str, Any]:
    report_paths = _find_report_paths(root_dir)
    results: list[dict[str, Any]] = []
    for report_path in report_paths:
        benchmark_output_dir = report_path.parent.parent
        uq_summary_path = benchmark_output_dir / "metrics" / "uq_summary.json"
        if skip_existing and uq_summary_path.exists():
            results.append(
                {
                    "benchmark_output_dir": str(benchmark_output_dir),
                    "status": "skipped_existing",
                    "uq_summary_path": str(uq_summary_path),
                }
            )
            continue
        result = backfill_uq_summary(
            benchmark_output_dir=benchmark_output_dir,
            write_retention_copy=write_retention_copy,
        )
        result["status"] = "backfilled"
        result["benchmark_output_dir"] = str(benchmark_output_dir)
        results.append(result)
    return {
        "root_dir": str(root_dir),
        "report_count": len(report_paths),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="outputs")
    parser.add_argument("--no-retention-copy", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    payload = backfill_all_uq_summaries(
        root_dir=Path(args.root_dir),
        write_retention_copy=not args.no_retention_copy,
        skip_existing=not args.overwrite_existing,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

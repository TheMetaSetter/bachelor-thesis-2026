from __future__ import annotations

"""Write the compact JSON emitted by build_remote_offline_report_data.py."""

import json
import sys
from pathlib import Path


OUTPUT_DIRECTORY = Path("outputs/reporting/offline_phase_tables")
OUTPUT_JSON = OUTPUT_DIRECTORY / "offline_report_data.json"
OUTPUT_MARKDOWN = OUTPUT_DIRECTORY / "offline_report_data.md"


def main() -> int:
    payload = json.load(sys.stdin)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    markdown = "\n".join(
        [
            "# Offline report data",
            "",
            f"- Source root: `{payload['source_root']}`",
            f"- Records: `{summary['record_count']}`",
            f"- Table 1 eligible: `{summary['table_1_eligible_count']}`",
            f"- Table 2 eligible: `{summary['table_2_eligible_count']}`",
            f"- Identity conflicts with diagnostics: `{summary['identity_conflict_count']}`",
            "",
            "Raw artifact paths and conflicting metadata are preserved in the JSON bundle.",
            "",
        ]
    )
    OUTPUT_MARKDOWN.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MARKDOWN),
                "summary": summary,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

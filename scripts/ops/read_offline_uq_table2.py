"""Read and render THESIS validation/test uncertainty for report table 2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_REPORT_PATH = Path(
    "outputs/reporting/offline_phase_tables/offline_report_data.json"
)
ENTITIES = ("machine_1_6", "machine_3_4", "machine_3_9")
VARIANTS = ("O0", "O1")
EXPECTED_SEEDS = {6, 8, 36}


def load_values(report_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for record in payload["records"]:
        identity = record["identity"]
        if identity["method_name"] != "THESIS" or not record["table_2_eligible"]:
            continue
        key = (identity["variant_name"], identity["entity_id"], identity["seed"])
        for split in ("clean_validation", "test"):
            value = record["uq_summary"]["splits"][split]["mean_of_variances"]
            if value is None:
                raise ValueError(f"Missing {split} uncertainty for {record['run_id']}")
            grouped[key + (split,)].append(float(value))

    values: dict[tuple[str, str], dict[str, float]] = {}
    for variant in VARIANTS:
        for entity in ENTITIES:
            row = {}
            for split in ("clean_validation", "test"):
                seed_values = [
                    grouped[(variant, entity, seed, split)][0]
                    for seed in EXPECTED_SEEDS
                ]
                row[split] = mean(seed_values)
            values[(variant, entity)] = row
    return values


def render_table(values: dict[tuple[str, str], dict[str, float]]) -> str:
    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2" class="blank-corner"></th>',
    ]
    lines.extend(f'      <th colspan="2">{entity}</th>' for entity in ENTITIES)
    lines.extend(
        [
            "    </tr>",
            "    <tr>",
        ]
    )
    lines.extend(
        (
            '      <th class="validation-header" style="background-color: #dbeafe;">Validation</th>\n'
            "      <th>Test</th>"
            if entity_index == 0
            else "      <th>Validation</th>\n      <th>Test</th>"
        )
        for entity_index, _ in enumerate(ENTITIES)
    )
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    for variant in VARIANTS:
        cells = []
        for entity in ENTITIES:
            row = values[(variant, entity)]
            cells.extend(
                (
                    f"      <td>{row['clean_validation']:.3f}</td>",
                    f"      <td>{row['test']:.3f}</td>",
                )
            )
        lines.append("    <tr>")
        lines.append(f"      <th>THESIS + {variant}</th>")
        lines.extend(cells)
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    print(render_table(load_values(args.report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read and render THESIS metrics and validation/test uncertainty for table 2."""

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
ENTITY_LABELS = {entity: entity.replace("_", "-") for entity in ENTITIES}
VARIANTS = ("O0", "O1")
EXPECTED_SEEDS = {6, 8, 36}
METRICS = ("vus_pr", "affiliation_f1", "clean_validation", "test")


def load_values(report_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for record in payload["records"]:
        identity = record["identity"]
        if identity["method_name"] != "THESIS" or not record["table_2_eligible"]:
            continue
        key = (identity["variant_name"], identity["entity_id"], identity["seed"])
        grouped[key + ("vus_pr",)].append(float(record["metrics"]["vus_pr"]))
        grouped[key + ("affiliation_f1",)].append(
            float(record["metrics"]["affiliation_f1"])
        )
        for split in ("clean_validation", "test"):
            value = record["uq_summary"]["splits"][split]["mean_of_variances"]
            if value is None:
                raise ValueError(f"Missing {split} uncertainty for {record['run_id']}")
            grouped[key + (split,)].append(float(value))

    values: dict[tuple[str, str], dict[str, float]] = {}
    for variant in VARIANTS:
        for entity in ENTITIES:
            row = {}
            for metric in METRICS:
                seed_values = [
                    grouped[(variant, entity, seed, metric)][0]
                    for seed in EXPECTED_SEEDS
                ]
                row[metric] = mean(seed_values)
            values[(variant, entity)] = row
    return values


def render_table(values: dict[tuple[str, str], dict[str, float]]) -> str:
    averages = {
        variant: {
            metric: mean(values[(variant, entity)][metric] for entity in ENTITIES)
            for metric in METRICS
        }
        for variant in VARIANTS
    }
    directions = {
        "vus_pr": "max",
        "affiliation_f1": "max",
        "clean_validation": "min",
        "test": "max",
    }
    ranks = {}
    for metric in METRICS:
        displayed = {
            variant: float(f"{averages[variant][metric]:.3f}") for variant in VARIANTS
        }
        ordered = sorted(set(displayed.values()), reverse=directions[metric] == "max")
        ranks[metric] = {
            variant: ordered.index(displayed[variant]) + 1 for variant in VARIANTS
        }

    def format_average(variant: str, metric: str) -> str:
        text = f"{averages[variant][metric]:.3f}"
        rank = ranks[metric][variant]
        if rank == 1:
            return f"<strong>{text}</strong>"
        return text

    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2" class="blank-corner"></th>',
    ]
    lines.extend(
        f'      <th colspan="4">{ENTITY_LABELS.get(entity, entity)}</th>'
        for entity in (*ENTITIES, "Average")
    )
    lines.extend(
        [
            "    </tr>",
            "    <tr>",
        ]
    )
    lines.extend(
        (
            '      <th class="vus-pr-header">VUS-PR</th>\n'
            '      <th class="aff-f1-header">Aff. F1</th>\n'
            '      <th class="validation-header" style="background-color: #dbeafe;">Validation</th>\n'
            '      <th class="test-header">Test</th>'
        )
        for _ in (*ENTITIES, "Average")
    )
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])

    for variant in VARIANTS:
        cells = []
        for entity in ENTITIES:
            row = values[(variant, entity)]
            cells.extend(
                (
                    f"      <td>{row['vus_pr']:.3f}</td>",
                    f"      <td>{row['affiliation_f1']:.3f}</td>",
                    f"      <td>{row['clean_validation']:.3f}</td>",
                    f"      <td>{row['test']:.3f}</td>",
                )
            )
        cells.extend(
            f"      <td>{format_average(variant, metric)}</td>" for metric in METRICS
        )
        lines.append("    <tr>")
        lines.append(f"      <th>THESIS {variant}</th>")
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

from __future__ import annotations

"""Render report table 1 from the compact offline report bundle."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPORT_PATH = Path("outputs/reporting/offline_phase_tables/offline_report_data.json")
OUTPUT_PATH = Path(
    "outputs/reporting/offline_phase_tables/offline_phase_metric_table_report1.md"
)
ENTITIES = ("machine_1_6", "machine_3_4", "machine_3_9")
ENTITY_LABELS = {entity: entity.replace("_", "-") for entity in ENTITIES}
METRICS = (
    ("vus_pr", "VUS-PR"),
    ("affiliation_f1", "affiliation F1"),
    ("vus_roc", "VUS-ROC"),
)
METHOD_ORDER = (
    ("THESIS", "O0", "Thesis main + O0"),
    ("THESIS", "O1", "Thesis main + O1"),
    ("redlamp_baseline", "baseline", "RedLamp + baseline"),
    ("iforest", "baseline", "iForest"),
    ("kmeans_ad", "baseline", "KMeans-AD"),
    ("stumpy_channel_ab", "baseline", "STUMPY + channel AB"),
)


def load_rows() -> dict[tuple[str, str, str], dict[str, float]]:
    payload: dict[str, Any] = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for record in payload["records"]:
        if not record["table_1_eligible"]:
            continue
        identity = record["identity"]
        variant = identity.get("variant_name") or "baseline"
        key = (identity["method_name"], variant, identity["entity_id"])
        grouped[key].append(record["metrics"])
    result = {}
    for key, values in grouped.items():
        if len(values) != 3:
            raise ValueError(f"Expected 3 seeds for {key}, found {len(values)}")
        result[key] = {
            metric: mean(row[metric] for row in values) for metric, _ in METRICS
        }
    return result


def format_value(value: float, rank: int) -> str:
    text = f"{value:.4f}"
    if rank == 1:
        return f"<strong>{text}</strong>"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def render() -> str:
    values = load_rows()
    ranks: dict[tuple[str, str, str], int] = {}
    for entity in ENTITIES:
        for metric, _ in METRICS:
            unique_values = sorted(
                {
                    values[(method, variant, entity)][metric]
                    for method, variant, _ in METHOD_ORDER
                },
                reverse=True,
            )
            value_ranks = {
                value: rank for rank, value in enumerate(unique_values, start=1)
            }
            for method, variant, _ in METHOD_ORDER:
                value = values[(method, variant, entity)][metric]
                ranks[(method, variant, entity, metric)] = value_ranks[value]

    lines = [
        "# Offline Phase — Table 1",
        "",
        "Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) "
        "cho cùng method, variant và entity.",
        "",
        "Giá trị cao nhất được in đậm; giá trị cao thứ hai được gạch chân. "
        "Các metric đều được hiểu là càng cao càng tốt.",
        "",
        "<style>",
        "  .report-shared { border-collapse: collapse; }",
        "  .report-shared th, .report-shared td { padding: 0.55rem 1.25rem; text-align: center; }",
        "  .report-shared .blank-corner { background: #fff; border: 0; }",
        "</style>",
        '<table class="report-shared">',
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2" class="blank-corner"></th>',
    ]
    lines.extend(
        f'      <th colspan="{len(METRICS)}">{ENTITY_LABELS[entity]}</th>'
        for entity in ENTITIES
    )
    lines.extend(
        [
            "    </tr>",
            "    <tr>",
        ]
    )
    lines.extend(
        f"      <th>{metric_label}</th>"
        for _ in ENTITIES
        for _, metric_label in METRICS
    )
    lines.extend(
        [
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
    )
    for method, variant, label in METHOD_ORDER:
        cells = []
        for entity in ENTITIES:
            for metric, _ in METRICS:
                value = values[(method, variant, entity)][metric]
                rank = ranks[(method, variant, entity, metric)]
                cells.append(f"      <td>{format_value(value, rank)}</td>")
        lines.append("    <tr>")
        lines.append(f"      <th>{label}</th>")
        lines.extend(cells)
        lines.append("    </tr>")
    lines.extend(
        [
            "  </tbody>",
            "</table>",
        ]
    )
    lines.extend(
        [
            "",
            "## Nguồn dữ liệu",
            "",
            f"- `{REPORT_PATH}`",
            "- Bảng có 6 hàng, 3 entity và 3 metric; tổng cộng 54 report record "
            "được gộp thành 18 combination theo entity.",
            "- Bảng này chỉ dùng `vus_pr`, `affiliation_f1` và `vus_roc`; UQ "
            "được trình bày ở bảng report riêng.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(), encoding="utf-8")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

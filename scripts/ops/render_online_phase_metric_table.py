from __future__ import annotations

"""Render the online benchmark comparison table from the Table 3 JSON payload."""

import json
from pathlib import Path
from statistics import mean
from typing import Any


REPORT_PATH = Path("reporting/online_phase_tables/online_table3_metrics.json")
OUTPUT_PATH = Path("reporting/online_phase_tables/online_phase_metric_table_report3.md")
ENTITIES = ("machine_1_6", "machine_3_4", "machine_3_9")
ENTITY_LABELS = {entity: entity.replace("_", "-") for entity in ENTITIES}
METRICS = (
    ("vus_pr", "VUS-PR"),
    ("affiliation_f1", "affiliation F1"),
    ("vus_roc", "VUS-ROC"),
)
METHOD_ORDER = (
    ("thesis", "O0", "A0", "THESIS O0 + A0"),
    ("thesis", "O0", "A1", "THESIS O0 + A1"),
    ("thesis", "O0", "A2", "THESIS O0 + A2"),
    ("thesis", "O1", "A0", "THESIS O1 + A0"),
    ("thesis", "O1", "A1", "THESIS O1 + A1"),
    ("thesis", "O1", "A2", "THESIS O1 + A2"),
    ("m2n2", "", "main", "M2N2"),
    ("candi", "", "main", "CANDI"),
    ("iforest", "", "main", "Isolation Forest"),
    ("kmeans_ad", "", "main", "KMeansAD"),
    ("stumpy", "", "main", "StumPy"),
)


def _load_values() -> dict[tuple[str, str, str, str], dict[str, float]]:
    payload: dict[str, Any] = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    values = {}
    for record in payload["summary_by_method_variant_entity"]:
        key = (
            record["method"],
            record["offline_variant"] or "",
            record["online_variant"],
            record["entity_id"],
        )
        values[key] = {
            metric: record["metrics"][metric]["mean"] for metric, _ in METRICS
        }
    return values


def _format_value(value: float, rank: int) -> str:
    text = f"{value:.4f}"
    if rank == 1:
        return f"<strong>{text}</strong>"
    if rank == 2:
        return f"<u>{text}</u>"
    return text


def _build_ranks(
    values: dict[tuple[str, str, str, str], dict[str, float]],
) -> dict[tuple[str, str, str, str, str], int]:
    ranks = {}
    for entity in ENTITIES:
        for metric, _ in METRICS:
            method_values = [
                values[(method, offline_variant, online_variant, entity)][metric]
                for method, offline_variant, online_variant, _ in METHOD_ORDER
            ]
            value_ranks = {
                value: rank
                for rank, value in enumerate(
                    sorted(set(method_values), reverse=True), 1
                )
            }
            for method, offline_variant, online_variant, _ in METHOD_ORDER:
                value = values[(method, offline_variant, online_variant, entity)][
                    metric
                ]
                ranks[(method, offline_variant, online_variant, entity, metric)] = (
                    value_ranks[value]
                )
    return ranks


def _average_across_entities(
    values: dict[tuple[str, str, str, str], dict[str, float]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    averages = {}
    for method, offline_variant, online_variant, _ in METHOD_ORDER:
        averages[(method, offline_variant, online_variant)] = {
            metric: mean(
                values[(method, offline_variant, online_variant, entity)][metric]
                for entity in ENTITIES
            )
            for metric, _ in METRICS
        }
    return averages


def _build_average_ranks(
    averages: dict[tuple[str, str, str], dict[str, float]],
) -> dict[tuple[str, str, str, str], int]:
    ranks = {}
    for metric, _ in METRICS:
        method_values = [
            averages[(method, offline_variant, online_variant)][metric]
            for method, offline_variant, online_variant, _ in METHOD_ORDER
        ]
        value_ranks = {
            value: rank
            for rank, value in enumerate(sorted(set(method_values), reverse=True), 1)
        }
        for method, offline_variant, online_variant, _ in METHOD_ORDER:
            value = averages[(method, offline_variant, online_variant)][metric]
            ranks[(method, offline_variant, online_variant, metric)] = value_ranks[
                value
            ]
    return ranks


def _render_table(
    values: dict[tuple[str, str, str, str], dict[str, float]],
    ranks: dict[tuple[str, str, str, str, str], int],
    averages: dict[tuple[str, str, str], dict[str, float]],
    average_ranks: dict[tuple[str, str, str, str], int],
) -> str:
    lines = [
        "# Online Phase — Table 3",
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
    lines.append(f'      <th colspan="{len(METRICS)}">Average</th>')
    lines.extend(["    </tr>", "    <tr>"])
    lines.extend(
        f"      <th>{metric_label}</th>"
        for _ in (*ENTITIES, "average")
        for _, metric_label in METRICS
    )
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for method, offline_variant, online_variant, label in METHOD_ORDER:
        lines.append("    <tr>")
        lines.append(f"      <th>{label}</th>")
        for entity in ENTITIES:
            for metric, _ in METRICS:
                value = values[(method, offline_variant, online_variant, entity)][
                    metric
                ]
                rank = ranks[(method, offline_variant, online_variant, entity, metric)]
                lines.append(f"      <td>{_format_value(value, rank)}</td>")
        for metric, _ in METRICS:
            value = averages[(method, offline_variant, online_variant)][metric]
            rank = average_ranks[(method, offline_variant, online_variant, metric)]
            lines.append(f"      <td>{_format_value(value, rank)}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def render() -> str:
    values = _load_values()
    ranks = _build_ranks(values)
    averages = _average_across_entities(values)
    average_ranks = _build_average_ranks(averages)
    table = _render_table(values, ranks, averages, average_ranks)
    return "\n".join(
        [
            table,
            "",
            "## Nguồn dữ liệu",
            "",
            f"- `{REPORT_PATH}`",
            "- Bảng có 11 hàng, 3 entity và 3 metric; tổng cộng 99 run được "
            "gộp thành 33 combination theo entity.",
            "- Ba cột cuối là trung bình số học của từng metric theo 3 entity.",
            "- Score dùng để tính metric là `online/ewma_point_score`; "
            "threshold và prediction giữ nguyên từ runtime online.",
            "- Protocol VUS dùng `vus_max_buffer_size = 20` và "
            "`vus_num_thresholds = 200`.",
            "",
        ]
    )


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(), encoding="utf-8")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

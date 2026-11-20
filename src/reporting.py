"""Select one result per seed and write experiment tables."""

from collections import defaultdict
from collections.abc import Sequence
import csv
from html import escape
from io import StringIO
from math import isfinite
from pathlib import Path
from statistics import fmean

import numpy as np

from .artifacts import _attempt_order, _component, _hash, _json_bytes, _read_json, _write_atomic, _write_json
from .artifacts import find_runs, result_directory
from .config import _profile_reference, load_yaml
from .experiments import _definition, _REPORTS


_SCIENTIFIC_KEYS = ("prepared_identity", "data", "method", "source_method", "synthetic", "encoder",
                    "evaluation", "device", "limits")
_IDENTITY = ("run_id", "phase", "dataset", "entity", "method_name", "variant", "seed")


_CONTEXT_METRICS = ("segment_mean_vus_pr", "segment_mean_affiliation_f1")
_DETECTION_METRICS = {
    "segment_mean_vus_pr": "VUS-PR",
    "segment_mean_affiliation_f1": "Aff. F1",
    "segment_mean_vus_roc": "VUS-ROC",
    "fpr": "FPR",
}
_REPORT_CSS = """
body { color: #142634; font: 1rem system-ui, sans-serif; margin: 1.5rem; }
h1, h2 { margin-bottom: 0.5rem; }
h2 { overflow-wrap: anywhere; }
section { margin-top: 1.5rem; }
p { color: #445665; margin: 0.5rem 0; }
.table-scroll { overflow-x: auto; }
table {
    border-collapse: collapse;
    table-layout: fixed;
    width: max(100%, calc(10rem + var(--columns) * 6.75ch));
    font-size: 0.875rem;
    font-variant-numeric: tabular-nums;
}
col.method { width: 10rem; }
th, td { padding: 0.4rem 0.25rem; text-align: center; overflow-wrap: anywhere; }
th[scope="row"] { text-align: left; }
th[scope="colgroup"] { background: #ccdff1; }
th[scope="col"] { background: #dcebf8; }
thead { border-bottom: 2px solid #445665; }
tbody tr:nth-child(even) { background: #f4f7fa; }
tbody:not(.average) { border-bottom: 2px solid #445665; }
.average tr:first-child > * { border-top: 0.75rem solid white; }
.average td.empty { background: white; }
"""


def _method_key(run):
    return run["method_name"], run.get("reporting_variant", run["variant"])


def _cell_value(records, metric, seeds):
    selected = {}
    for run in sorted(records, key=_attempt_order, reverse=True):
        value = run.get("metrics", {}).get(metric)
        if run["status"] == "successful" and run["seed"] in seeds and type(value) in (float, int) and isfinite(value):
            selected.setdefault(run["seed"], run)
    identities = {_json_bytes({key: run["resolved_config"].get(key) for key in _SCIENTIFIC_KEYS}
                              | {"execution_identity": run.get("execution_identity")})
                  for run in selected.values()}
    if len(identities) > 1:
        raise ValueError(f"incompatible configurations for metric {metric}")
    numbers = [run["metrics"][metric] for run in selected.values()]
    return {"mean": float(np.mean(numbers)) if numbers else None,
            "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
            "valid": len(numbers), "expected": len(seeds)}


def aggregate_table(runs, reporting: dict, *, expected_seeds) -> dict:
    """Aggregate one dataset and phase without filesystem writes."""
    expected = set(expected_seeds)
    if not expected or len(expected) != len(expected_seeds):
        raise ValueError("expected seeds must be a nonempty list of distinct values")
    relevant = [record for record in runs if record["phase"] == reporting["phase"]
                and record["dataset"] == reporting["dataset"] and record["split"] in reporting["splits"]]
    entities = sorted({run["entity"] for run in relevant})
    columns = [(split, entity, metric) for split in reporting["splits"]
               for entity in entities for metric in reporting["metrics"]]
    records = defaultdict(list)
    for run in relevant:
        records[_method_key(run)].append(run)
    rows = []
    for (method, variant), candidates in sorted(records.items()):
        groups = defaultdict(list)
        for run in candidates:
            groups[run["split"], run["entity"]].append(run)
        profiles = sorted({run["profile_paths"]["method"] for run in candidates})
        cells = [_cell_value(groups[split, entity], metric, expected) for split, entity, metric in columns]
        rows.append({"method": method, "variant": variant, "method_profiles": profiles, "cells": cells})
    return {"columns": columns, "rows": rows}


def _precision(profile, metric_defs):
    expected = {"phase", "splits", "metrics"}
    if not expected <= profile.keys() or profile.keys() - expected - {"precision"}:
        raise ValueError("reporting profile has unsupported fields")
    for key in ("splits", "metrics"):
        values = profile[key]
        if not isinstance(values, list) or not values or any(type(item) is not str for item in values):
            raise ValueError(f"reporting {key} must be a nonempty name list")
        if len(set(values)) != len(values):
            raise ValueError(f"reporting {key} must contain distinct names")
    if profile["phase"] not in {"offline", "online"} or set(profile["splits"]) - {"train", "validation", "test"}:
        raise ValueError("invalid reporting phase or split")
    overrides = profile.get("precision", {})
    if not isinstance(overrides, dict) or overrides.keys() - set(profile["metrics"]):
        raise ValueError("precision overrides must name selected metrics")
    result = {}
    for metric in profile["metrics"]:
        shared = metric_defs.get(metric)
        if not isinstance(shared, dict) or shared.keys() != {"precision"}:
            raise ValueError(f"missing or invalid shared metric definition: {metric}")
        digits = overrides.get(metric, shared["precision"])
        if any(type(value) is not int or not 0 <= value <= 15 for value in (digits, shared["precision"])):
            raise ValueError(f"invalid precision for {metric}")
        result[metric] = digits
    return result


def _split_record(run, split):
    config = dict(run["resolved_config"])
    path = result_directory(config, split) / "result.json"
    base = {key: run[key] for key in _IDENTITY} | {
        "split": split, "status": run["status"], "started_at": run["started_at"],
        "completed_at": run.get("completed_at", run["started_at"]), "resolved_config": config,
        "profile_paths": run["profile_paths"],
        "execution_identity": run.get("details", {}).get("execution_identity"),
    }
    if "reporting_variant" in run:
        base["reporting_variant"] = run["reporting_variant"]
    if run["status"] != "successful" or not path.is_file():
        return base | {"metrics": {}}
    result = _read_json(path)
    if result.get("schema_version") != 1 or any(result.get(key) != base[key] for key in (*_IDENTITY, "split")):
        raise ValueError(f"result identity mismatch: {path}")
    if "point_scores_sha256" in result and _hash(path.parent / "point_scores.parquet") != result["point_scores_sha256"]:
        raise ValueError(f"score result hash mismatch: {path}")
    return base | {"metrics": result["metrics"]}


def _format(value, precision):
    return "N/A" if value is None else f"{value:.{precision}f}"


def _write_table(path, table, precision, *, detailed):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["method", "variant", "method_profile", *("/".join(column) for column in table["columns"])])
    for row in table["rows"]:
        values = []
        for (_, _, metric), cell in zip(table["columns"], row["cells"], strict=True):
            digits = precision[metric]
            mean = _format(cell["mean"], digits)
            if detailed:
                mean = f"{mean} ± {_format(cell['std'], digits)} ({cell['valid']}/{cell['expected']})"
            values.append(mean)
        writer.writerow([row["method"], row["variant"], " | ".join(row["method_profiles"]), *values])
    return _write_atomic(path, lambda file: file.write(stream.getvalue().encode("utf-8")))


def _entity_average(cells: Sequence[dict]) -> float | None:
    if not cells or any(_full_mean(cell) is None for cell in cells):
        return None
    return fmean(cell["mean"] for cell in cells)


def _metric_ranks(values: Sequence[float | None], *, higher_is_better: bool) -> list[int | None]:
    ordered = sorted({value for value in values if value is not None}, reverse=higher_is_better)
    ranks = {value: rank for rank, value in enumerate(ordered[:2], start=1)}
    return [ranks.get(value) for value in values]


def _detection_context(records, dataset, shared, seeds):
    settings = {"phase": "offline", "splits": ["test"], "metrics": list(_CONTEXT_METRICS)}
    context = aggregate_table(records, settings | {"dataset": dataset}, expected_seeds=seeds)
    return context | {"precision": _precision(settings, shared)}


def _full_mean(cell):
    return cell["mean"] if cell["valid"] == cell["expected"] else None


def _row_values(table):
    return {(row["method"], row["variant"]): dict(zip(table["columns"], row["cells"], strict=True))
            for row in table["rows"]}


def _render_report_html(table: dict, profile: dict, precision: dict[str, int], *,
                        experiment_id: str, dataset: str, smoke: bool,
                        detection_context: dict | None = None) -> str:
    entities = list(dict.fromkeys(entity for _, entity, _ in table["columns"]))
    values = _row_values(table)
    if detection_context is None:
        tables = _detection_tables(profile, entities, values, precision)
    else:
        for key, context in _row_values(detection_context).items():
            values[key].update(context)
        tables = _uncertainty_tables(profile, entities, values, precision, detection_context["precision"])
    title = "Smoke run report" if smoke else "Experiment report"
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_REPORT_CSS}</style></head><body><main>"
        f"<h1>{title}</h1><p>{escape(experiment_id)} · {escape(dataset)}</p>"
        f'{"".join(tables)}'
        "<p>N/A means the value is unavailable. The count shows valid seeds out of expected seeds.</p>"
        "</main></body></html>\n"
    )


def _detection_tables(profile, entities, values, precision):
    preferred = [metric for metric in _DETECTION_METRICS if metric in profile["metrics"]]
    metrics = preferred + [metric for metric in profile["metrics"] if metric not in preferred]
    labels = [_DETECTION_METRICS.get(metric, metric) for metric in metrics]
    for split in profile["splits"]:
        columns = [(split, metric, precision[metric]) for metric in metrics]
        rows, average = _table_values(values, entities, columns, rank=True)
        title = f"{profile['phase'].capitalize()} detection · {split.capitalize()} split"
        note = "Bold marks the best value. Underline marks the second distinct value. Lower FPR is better."
        groups = entities
        if average:
            groups = [*entities, "Average"]
            rows = [row + avg[1:] for row, avg in zip(rows, average, strict=True)]
            note += " Average gives each entity the same weight."
        yield _html_table(title, groups, labels, rows, note=note)


def _uncertainty_tables(profile, entities, values, precision, context_precision):
    context = [("test", name, context_precision[name]) for name in _CONTEXT_METRICS]
    for metric in profile["metrics"]:
        columns = [*context, ("validation", metric, precision[metric]), ("test", metric, precision[metric])]
        rows, average = _table_values(values, entities, columns)
        title = f"Offline uncertainty · {metric}"
        yield _html_table(title, entities, ["VUS-PR", "Aff. F1", "Validation", "Test"], rows,
                          average=average, note="VUS-PR and Aff. F1 use the test split.")


def _table_values(values, entities, columns, *, rank=False):
    cell_columns, averages = [], []
    missing = {"mean": None, "valid": 0, "expected": 1}
    for split, metric, digits in columns:
        by_entity = [[row.get((split, entity, metric), missing) for row in values.values()]
                     for entity in entities]
        cell_columns.append([_display_column(cells, metric, digits, rank) for cells in by_entity])
        if len(entities) > 1:
            means = [{"mean": _entity_average(cells), "valid": 1, "expected": 1}
                     for cells in zip(*by_entity, strict=True)]
            averages.append(_display_column(means, metric, digits, rank))
    ordered = [column for group in zip(*cell_columns, strict=True) for column in group]
    rows = [list(row) for row in zip(*ordered, strict=True)]
    labels = [_method_label(method, variant) for method, variant in values]
    rows = [[label, *row] for label, row in zip(labels, rows, strict=True)]
    average_rows = [[label, *row] for label, row in zip(labels, zip(*averages, strict=True), strict=True)] if averages else []
    return rows, average_rows


def _method_label(method, variant):
    name = {"mustard": "MUSTARD", "redlamp": "RedLamp", "candi": "CANDI", "m2n2": "M2N2"}.get(method, method)
    return escape(f"{name} {variant}")


def _display_column(cells, metric, digits, rank):
    ranks = (_metric_ranks([_full_mean(cell) for cell in cells], higher_is_better=metric != "fpr")
             if rank and metric in _DETECTION_METRICS else [None] * len(cells))
    output = []
    for cell, place in zip(cells, ranks, strict=True):
        value = _format(cell["mean"], digits)
        if cell["mean"] is not None and cell["valid"] < cell["expected"]:
            value += f' ({cell["valid"]}/{cell["expected"]})'
        if place == 1:
            value = f"<strong>{value}</strong>"
        elif place == 2:
            value = f"<u>{value}</u>"
        output.append(value)
    return output


def _html_table(title, groups, labels, rows, *, note, average=None):
    width = len(labels)
    count = width * len(groups)
    cols = '<col class="method">' + "<col>" * count
    heads = "".join(f'<th scope="colgroup" colspan="{width}">{escape(group)}</th>' for group in groups)
    metrics = "".join(f'<th scope="col">{escape(label)}</th>' for label in labels)
    html = (
        f'<section><h2>{escape(title)}</h2><div class="table-scroll" tabindex="0">'
        f'<table style="--columns:{count}">'
        f"<colgroup>{cols}</colgroup><thead>"
        f'<tr><th rowspan="2" scope="col">Method</th>{heads}</tr>'
        f"<tr>{metrics * len(groups)}</tr></thead><tbody>{_html_rows(rows)}</tbody>"
    )
    if average:
        remaining = count - width
        html += (
            '<tbody class="average">'
            f'<tr><td></td><th scope="colgroup" colspan="{width}">Average</th>{_empty_columns(remaining)}</tr>'
            f'<tr><td></td>{metrics}{_empty_columns(remaining)}</tr>'
            f"{_html_rows(average, remaining)}</tbody>"
        )
        note += " Average gives each entity the same weight."
    return f"{html}</table></div><p>{escape(note)}</p></section>"


def _html_rows(rows, remaining=0):
    return "".join(
        f'<tr><th scope="row">{label}</th>'
        + "".join(f"<td>{value}</td>" for value in values)
        + _empty_columns(remaining) + "</tr>"
        for label, *values in rows
    )


def _empty_columns(count):
    return f'<td class="empty" colspan="{count}"></td>' if count else ""


def report_experiment(path: Path, *, project_root: Path, smoke: bool = False) -> list[Path]:
    definition = _definition(path)
    experiment = definition["experiment_id"]
    mode, category = ("smoke", "smoke") if smoke else ("normal", "experiments")
    root = project_root / "outputs" / category / experiment
    config_root = (project_root / "configs").resolve()
    shared = load_yaml(config_root / "metrics.yaml")
    runs = [run for run in find_runs(root, filters={
        "experiment_id": experiment, "execution_mode": mode, "status": "successful",
    }, include_results=False) if "resolved_config" in run]
    output = []
    seeds = definition.get("random_seeds", [6])
    if smoke:
        seeds = seeds[:1]
    for reference in definition.get("reporting_profiles", _REPORTS):
        profile_path = _profile_reference(config_root, "reporting_profiles", reference)
        profile = load_yaml(profile_path)
        precision = _precision(profile, shared)
        selected = [run for run in runs if run["phase"] == profile["phase"]]
        for dataset in sorted({run["dataset"] for run in selected}):
            dataset = _component(dataset)
            dataset_runs = [run for run in selected if run["dataset"] == dataset]
            base = root / "reports" / dataset / _component(profile_path.stem)
            output.extend(_report_dataset(base, dataset_runs, reference,
                                          profile, precision, shared, seeds, experiment, smoke))
    return output


def _report_dataset(base, runs, reference, profile, precision, shared, seeds, experiment, smoke):
    dataset = runs[0]["dataset"]
    settings = profile | {"dataset": dataset}
    records = [_split_record(run, split) for run in runs for split in profile["splits"]]
    table = aggregate_table(records, settings, expected_seeds=seeds)
    context = None
    if base.name == "uncertainty" and profile["phase"] == "offline":
        test_records = records if "test" in profile["splits"] else [_split_record(run, "test") for run in runs]
        context = _detection_context(test_records, dataset, shared, seeds)
    html = _render_report_html(table, profile, precision, experiment_id=experiment,
                               dataset=dataset, smoke=smoke, detection_context=context)
    paths = [_write_table(base.parent / f"{base.name}_{suffix}.csv", table, precision, detailed=detailed)
             for suffix, detailed in (("detail", True), ("mean", False))]
    _write_json(base.parent / f"{base.name}.json", {"schema_version": 1, "profile": reference,
                "dataset": dataset, "phase": profile["phase"], "splits": profile["splits"],
                "expected_seeds": seeds, "precision": precision})
    paths.append(_write_atomic(base.parent / f"{base.name}.html", lambda file: file.write(html.encode("utf-8"))))
    return paths

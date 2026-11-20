"""Check seed selection, table coverage, and CSV formatting."""

import csv
import importlib
import json
from html.parser import HTMLParser

import pytest
import yaml

from src.artifacts import allocate_run, write_result, write_run_status


@pytest.fixture
def reporting():
    return importlib.import_module("src.reporting")


def _row(seed, value, *, day=1, split="test", width=20, variant="default", status="successful"):
    return {"dataset": "smd", "phase": "offline", "split": split, "entity": "machine",
            "method_name": "kmeans_ad", "variant": variant, "seed": seed, "status": status,
            "started_at": f"2026-11-{day:02d}T10:00:00Z", "run_id": f"run-{seed}-{day}",
            "metrics": {"fpr": value}, "resolved_config": {"data": {"window_size": width}},
            "profile_paths": {"method": "method_profiles/baselines/kmeans_ad.yaml"}}


def test_report_seed_selection_and_missing_values(reporting):
    rows = [_row(1, 0.1), _row(1, 0.3, day=2), _row(1, None, day=3),
            _row(2, 0.5), _row(3, 0.8, status="failed"), _row(1, None, split="validation")]
    settings = {"dataset": "smd", "phase": "offline", "splits": ["validation", "test"],
                "metrics": ["fpr"]}
    table = reporting.aggregate_table(rows, settings, expected_seeds=[1, 2, 3])
    validation, test = table["rows"][0]["cells"]
    assert validation == {"mean": None, "std": None, "valid": 0, "expected": 3}
    assert test["mean"] == pytest.approx(0.4)
    assert test["std"] == pytest.approx(0.141421356)
    assert test["valid"] == 2 and test["expected"] == 3
    single = reporting.aggregate_table(rows, settings, expected_seeds=[1])
    assert single["rows"][0]["cells"][1] == {"mean": 0.3, "std": None, "valid": 1, "expected": 1}


def test_report_rejects_incompatible_seeds(reporting):
    rows = [_row(1, 0.2), _row(2, 0.3, width=30)]
    with pytest.raises(ValueError, match="incompatible"):
        reporting.aggregate_table(rows, {"dataset": "smd", "phase": "offline", "splits": ["test"],
                                         "metrics": ["fpr"]}, expected_seeds=[1, 2])


def _report_project(tmp_path, profiles=None, *, precision=None):
    configs = tmp_path / "configs"
    config = configs / "experiments/try.yaml"
    config.parent.mkdir(parents=True)
    profiles = profiles or {"test": {"phase": "offline", "splits": ["test"], "metrics": ["fpr"]}}
    config.write_text(yaml.safe_dump({
        "experiment_id": "try", "random_seeds": [1, 2],
        "reporting_profiles": [f"reporting_profiles/{name}.yaml" for name in profiles],
    }))
    folder = configs / "reporting_profiles"
    folder.mkdir()
    for name, profile in profiles.items():
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(profile))
    metrics = {metric: {"precision": 4} for profile in profiles.values() for metric in profile["metrics"]}
    metrics.update({metric: {"precision": digits} for metric, digits in (precision or {}).items()})
    (configs / "metrics.yaml").write_text(yaml.safe_dump(metrics))
    return config


def _saved_run(root, results, *, mode="normal", status="successful", **settings):
    run = allocate_run(_row(1, None) | dict(
        experiment_id="try", execution_mode=mode, run_id=None,
    ) | settings, project_root=root)
    write_run_status(run, "running")
    paths = {split: write_result(run, split, {"metrics": metrics}) for split, metrics in results.items()}
    if status != "running":
        write_run_status(run, status)
    return paths


def test_report_formats_shared_precision_and_excludes_smoke(reporting, tmp_path):
    config = _report_project(tmp_path)
    profile = tmp_path / "configs/reporting_profiles/test.yaml"
    definition = yaml.safe_load(profile.read_text())
    for mode, value in (("normal", 0.123456), ("smoke", 0.987654)):
        _saved_run(tmp_path, {"test": {"fpr": value}}, mode=mode, variant="<demo>")
    result = reporting.report_experiment(config, project_root=tmp_path)
    details = next(path for path in result if path.stem.endswith("_detail"))
    means = next(path for path in result if path.stem.endswith("_mean"))
    with details.open() as file:
        row = list(csv.reader(file))[1]
    assert row == ["kmeans_ad", "<demo>", "method_profiles/baselines/kmeans_ad.yaml", "0.1235 ± N/A (1/2)"]
    with means.open() as file:
        assert list(csv.reader(file))[1][-1] == "0.1235"
    profile.write_text(yaml.safe_dump(definition | {"precision": {"fpr": 2}}))
    reporting.report_experiment(config, project_root=tmp_path)
    with means.open() as file:
        assert list(csv.reader(file))[1][-1] == "0.12"

    smoke_paths = reporting.report_experiment(config, project_root=tmp_path, smoke=True)
    smoke_dir = tmp_path / "outputs/smoke/try/reports/smd"
    normal_dir = tmp_path / "outputs/experiments/try/reports/smd"
    assert all(path.parent == smoke_dir for path in smoke_paths)
    assert all(path.parent == normal_dir for path in result)
    profile.write_text(yaml.safe_dump(definition))
    reporting.report_experiment(config, project_root=tmp_path, smoke=True)
    with (smoke_dir / "test_detail.csv").open() as file:
        assert list(csv.reader(file))[1][-1] == "0.9877 ± N/A (1/1)"
    for folder, paths in ((normal_dir, result), (smoke_dir, smoke_paths)):
        assert {path.name for path in paths} == {"test.html", "test_detail.csv", "test_mean.csv"}
        assert {path.name for path in folder.iterdir()} == {
            "test.html", "test_detail.csv", "test_mean.csv", "test.json",
        }
        html = (folder / "test.html").read_text()
        assert "<demo>" not in html and "&lt;demo&gt;" in html
        assert ("Smoke run report" in html) == (folder == smoke_dir)


def test_report_requires_successful_runs(reporting, tmp_path):
    config = _report_project(tmp_path)
    _saved_run(tmp_path, {"test": {"fpr": 0.1}}, status="failed")
    _saved_run(tmp_path, {"test": {"fpr": 0.2}}, status="running")
    _saved_run(tmp_path, {"test": {"fpr": 0.3}}, phase="online")
    assert reporting.report_experiment(config, project_root=tmp_path) == []
    assert not (tmp_path / "outputs/experiments/try/reports").exists()

    _saved_run(tmp_path, {})
    report = reporting.report_experiment(config, project_root=tmp_path)
    with next(path for path in report if path.stem.endswith("_mean")).open() as file:
        assert list(csv.reader(file))[1][-1] == "N/A"


@pytest.mark.parametrize("invalid, error", [
    ({"split": "validation"}, "result identity mismatch"),
    ({"point_scores_sha256": "invalid"}, "score result hash mismatch"),
])
def test_report_rejects_invalid_saved_result(reporting, tmp_path, invalid, error):
    config = _report_project(tmp_path)
    result = _saved_run(tmp_path, {"test": {"fpr": 0.1}})["test"]
    result.write_text(json.dumps(json.loads(result.read_text()) | invalid))
    (result.parent / "point_scores.parquet").write_bytes(b"scores")
    with pytest.raises(ValueError, match=error):
        reporting.report_experiment(config, project_root=tmp_path)


class _Tables(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tables = []
        self.cell = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table":
            self.tables.append({"rows": [], "cols": []})
        elif tag == "col":
            self.tables[-1]["cols"].append(attributes)
        elif tag == "tr":
            self.tables[-1]["rows"].append([])
        elif tag in {"td", "th"}:
            self.cell = [int(attributes.get("colspan", "1")), "", None]
            self.tables[-1]["rows"][-1].append(self.cell)
        elif tag in {"strong", "u"} and self.cell is not None:
            self.cell[2] = tag

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[1] += data


_DETECTION = ["segment_mean_vus_pr", "segment_mean_vus_roc", "segment_mean_affiliation_f1", "fpr"]


def _render_detection(reporting, rows, profile, seeds):
    table = reporting.aggregate_table(rows, profile | {"dataset": "smd"}, expected_seeds=seeds)
    return reporting._render_report_html(
        table, profile, dict.fromkeys(profile["metrics"], 4),
        experiment_id="try", dataset="smd", smoke=False,
    )


@pytest.mark.parametrize("phase, entities, splits, metrics, labels", [
    ("offline", ["a", "b"], ["test"], _DETECTION, ["VUS-PR", "Aff. F1", "VUS-ROC", "FPR"]),
    ("online", ["a", "b"], ["test"], _DETECTION, ["VUS-PR", "Aff. F1", "VUS-ROC", "FPR"]),
    ("offline", ["a"], ["validation", "test"], ["pr_auc", "fpr"], ["FPR", "pr_auc"]),
])
def test_detection_html(reporting, phase, entities, splits, metrics, labels):
    profile = {"phase": phase, "splits": splits, "metrics": metrics}
    rows = [_row(1, 0.2, split=split, variant=variant) | {"phase": phase, "entity": entity}
            for split in splits for variant in ["O0", "O1"] for entity in entities]
    html = _render_detection(reporting, rows, profile, [1])
    groups = entities + (["Average"] if len(entities) > 1 else [])
    tables = _Tables(html).tables
    assert len(tables) == len(splits)
    for table in tables:
        group_head, metric_head, first, second = table["rows"]
        assert [(span, label) for span, label, _ in group_head[1:]] == [(len(labels), group) for group in groups]
        assert [label for _, label, _ in metric_head] == labels * len(groups)
        assert len(table["cols"]) == 1 + len(labels) * len(groups)
        assert all(column == table["cols"][1] for column in table["cols"][1:])
        assert first[0][1].endswith("O0") and second[0][1].endswith("O1")


def test_html_coverage_and_ranks(reporting):
    rows = []
    for variant, values in {
        "A": (0.2, 0.4), "B": (0.2, 0.4), "C": (0.20001, 0.40001),
        "D": (0.20002, 0.40002), "E": (0.1, None),
    }.items():
        for entity, value in zip(("a", "b"), values, strict=True):
            for seed in ([1] if variant == "E" else [1, 2]):
                rows.append(_row(seed, value, variant=variant) | {"entity": entity})
    profile = {"phase": "offline", "splits": ["test"], "metrics": ["fpr"]}
    content = _Tables(_render_detection(reporting, rows, profile, [1, 2])).tables[0]["rows"][2:]
    assert all([label for _, label, _ in row[1:]] == ["0.2000", "0.4000", "0.3000"] for row in content[:4])
    assert [[style for _, _, style in row[1:]] for row in content] == [
        ["strong"] * 3, ["strong"] * 3, ["u"] * 3, [None] * 3, [None] * 3,
    ]
    assert [label for _, label, _ in content[4][1:]] == ["0.1000 (1/2)", "N/A", "N/A"]


def test_uncertainty_html(reporting, tmp_path):
    vus_pr, affiliation = "segment_mean_vus_pr", "segment_mean_affiliation_f1"
    metrics = ["point_anomaly_score_variance_std", "point_anomaly_score_variance_mean"]
    profile = {"phase": "offline", "splits": ["validation", "test"], "metrics": metrics}
    config = _report_project(tmp_path, {"uncertainty": profile}, precision={vus_pr: 2, affiliation: 3})
    for entity, offset in [("a", 0.1), ("b", 0.2)]:
        for variant, shift in [("O0", 0.0), ("O1", 0.1)]:
            for seed in [1, 2]:
                _saved_run(tmp_path, {
                    "validation": {metrics[0]: offset + shift, metrics[1]: 0.01},
                    "test": {vus_pr: 0.5 + offset + shift, affiliation: 0.7 if entity == "a" else None,
                             metrics[0]: offset + shift + 0.01, metrics[1]: 0.02},
                }, entity=entity, reporting_variant=variant, seed=seed)
    paths = reporting.report_experiment(config, project_root=tmp_path)
    html = next(path for path in paths if path.suffix == ".html").read_text()
    assert html.index(metrics[0]) < html.index(metrics[1])
    first, second = _Tables(html).tables
    main = first["rows"]
    assert [(span, label) for span, label, _ in main[0][1:]] == [(4, "a"), (4, "b")]
    assert [label for _, label, _ in main[1]] == ["VUS-PR", "Aff. F1", "Validation", "Test"] * 2
    assert [label for _, label, _ in main[2][1:]] == [
        "0.60", "0.700", "0.1000", "0.1100", "0.70", "N/A", "0.2000", "0.2100",
    ]
    assert [label for _, label, _ in main[3][1:5]] == ["0.70", "0.700", "0.2000", "0.2100"]
    assert main[4][1][:2] == [4, "Average"]
    assert main[5][1:5] == main[1][:4]
    assert [label for _, label, _ in main[6][1:5]] == ["0.65", "N/A", "0.1500", "0.1600"]
    assert [label for _, label, _ in second["rows"][2][3:5]] == ["0.0100", "0.0200"]
    assert all(style is None for table in (first, second) for row in table["rows"] for _, _, style in row)
    with next(path for path in paths if path.stem.endswith("_mean")).open() as file:
        assert {column.rsplit("/", 1)[-1] for column in next(csv.reader(file))[3:]} == set(metrics)


def test_ignores_corrupt_results_outside_report_selection(reporting, tmp_path):
    config = _report_project(tmp_path)
    unselected = [
        (_saved_run(tmp_path, {"test": {}}, status="failed"), "test"),
        (_saved_run(tmp_path, {"test": {}}, phase="online"), "test"),
        (_saved_run(tmp_path, {"validation": {}, "test": {"fpr": 0.1}}), "validation"),
    ]
    for results, split in unselected:
        results[split].write_text("{")
    paths = reporting.report_experiment(config, project_root=tmp_path)
    with next(path for path in paths if path.stem.endswith("_mean")).open() as file:
        assert list(csv.reader(file))[1][-1] == "0.1000"


def test_uncertainty_context_and_profile_use_separate_precision(reporting, tmp_path):
    vus_pr, affiliation = "segment_mean_vus_pr", "segment_mean_affiliation_f1"
    profile = {"phase": "offline", "splits": ["validation", "test"], "metrics": [vus_pr],
               "precision": {vus_pr: 6}}
    config = _report_project(tmp_path, {"uncertainty": profile}, precision={vus_pr: 2, affiliation: 4})
    _saved_run(tmp_path, {"validation": {vus_pr: 0.345678}, "test": {vus_pr: 0.456789}})
    paths = reporting.report_experiment(config, project_root=tmp_path)
    html = next(path for path in paths if path.suffix == ".html").read_text()
    row = _Tables(html).tables[0]["rows"][2]
    assert [value for _, value, _ in row[1:]] == [
        "0.46 (1/2)", "N/A", "0.345678 (1/2)", "0.456789 (1/2)",
    ]

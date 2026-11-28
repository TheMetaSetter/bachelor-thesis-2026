"""Check definition edits, read-only planning, and experiment execution."""

import importlib
import json
from pathlib import Path
from shutil import copytree

import pytest
import yaml

from src.config import load_yaml


@pytest.fixture
def experiments(tmp_path):
    module = importlib.import_module("src.experiments")
    checkout = Path(__file__).resolve().parents[1]
    copytree(checkout / "configs", tmp_path / "configs")
    # Planning fixtures use the existing checked prepared files without copying datasets.
    for file in (tmp_path / "configs/data_profiles").glob("*.yaml"):
        profile = load_yaml(file)
        profile["prepared_root"] = str(checkout / profile["prepared_root"])
        file.write_text(yaml.safe_dump(profile))
    return module, tmp_path


def test_definition_editing_and_defaults(experiments):
    module, project = experiments
    path = module.create_experiment("example", project_root=project)
    assert "example" in module.list_experiments(project_root=project)
    with pytest.raises(FileExistsError):
        module.create_experiment("example", project_root=project)
    for field, values in (("datasets", ["smd"]), ("methods", ["mustard", "kmeans_ad"])):
        module.edit_experiment(path, "add", field, values, project_root=project)
    module.edit_experiment(
        path, "set", "random_seeds", ["7", "2", "9"], project_root=project
    )
    with pytest.raises(ValueError):
        module.edit_experiment(
            path, "set", "output_dir", ["extra"], project_root=project
        )
    definition = load_yaml(path)
    assert definition["random_seeds"] == [7, 2, 9]
    assert definition["methods"] == ["mustard", "kmeans_ad"]
    plan = module.plan_experiment(path, project_root=project, smoke=True)
    assert [(run["method_name"], run["phase"], run["seed"]) for run in plan] == [
        ("mustard", "offline", 7),
        ("kmeans_ad", "offline", 7),
        ("mustard", "online", 7),
        ("kmeans_ad", "online", 7),
    ]
    assert all(run["profile_paths"]["data"] == "data_profiles/smd.yaml" for run in plan)
    assert all(run["entity"] == "machine-1-1" for run in plan)


def test_experiment_rejects_two_profiles_for_one_dataset(experiments):
    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    definition = load_yaml(path)
    profile = project / "configs/data_profiles/smd.yaml"
    profile.with_stem("alternate_smd").write_text(profile.read_text())
    definition["cells"][1]["data_profile"] = "data_profiles/alternate_smd.yaml"
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="one data profile"):
        module.plan_experiment(path, project_root=project)


def test_planning_has_no_filesystem_side_effects(experiments):
    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    original = path.read_bytes()
    plan = module.plan_experiment(path, project_root=project)
    assert len(plan) == 2
    assert all("run_id" not in cell for cell in plan)
    assert not (project / "outputs").exists()
    assert path.read_bytes() == original


def test_shared_profile_updates_two_new_plans(experiments):
    from src.dataset_selection import add_datasets

    module, project = experiments
    first = project / "configs/experiments/smd_comparison.yaml"
    second = first.with_stem("second")
    second.write_text(
        first.read_text().replace(
            "experiment_id: smd_comparison", "experiment_id: second"
        )
    )
    profile = project / "configs/data_profiles/smd.yaml"
    before = Path(load_yaml(profile)["prepared_root"])
    version = project / "data/preprocessed/smd/selected"
    entity = "machine-1-1"
    copytree(before / entity, version / entity)
    manifest = json.loads((before / "manifest.json").read_text())
    manifest["entities"] = {entity: manifest["entities"][entity]}
    manifest["version_id"] = version.name
    (version / "manifest.json").write_text(json.dumps(manifest))

    add_datasets(first, ["smd"], project, lambda _: {"smd": version})
    for path in (first, second):
        plan = module.plan_experiment(path, project_root=project, smoke=True)
        assert {run["data"]["prepared_root"] for run in plan} == {version}


def test_smoke_adds_pinned_offline_dependency_for_selected_variant(experiments):
    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    definition = load_yaml(path)
    baseline = project / "configs/method_profiles/mustard_a0_online.yaml"
    profile = load_yaml(baseline)
    profile["required_offline_variant"] = "O1"
    baseline.with_stem("mustard_a0_from_o1").write_text(yaml.safe_dump(profile))
    definition["cells"][1]["method_profile"] = "method_profiles/mustard_a0_from_o1.yaml"
    definition["random_seeds"] = [2, 4]
    path.write_text(yaml.safe_dump(definition))
    plan = module.plan_experiment(path, project_root=project, smoke=True)
    assert [(cell["variant"], cell["seed"]) for cell in plan] == [
        ("O0", 2),
        ("O1", 2),
        ("A0", 2),
    ]
    assert [cell["execution_mode"] for cell in plan] == ["smoke"] * 3
    assert all(cell["limits"]["max_validation_windows"] == 64 for cell in plan)


def test_unsupported_combination_stops_planning(experiments):
    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    definition = load_yaml(path)
    definition["cells"][0]["method_profile"] = "method_profiles/mustard_a0_online.yaml"
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match="does not support offline"):
        module.plan_experiment(path, project_root=project, smoke=True)


def test_experiment_keeps_failures_and_runs_independent_cells(experiments, monkeypatch):
    from src.artifacts import write_run_status
    from src.online import OfflineDependencyUnavailable

    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    plans = module.plan_experiment(path, project_root=project, smoke=True)
    offline, online = plans
    schedule = [offline, offline | {"seed": 9}, online, online | {"seed": 9}]
    events = []

    def run(config, *, resume=None):
        phase, seed = config["phase"], config["seed"]
        events.append((phase, seed))
        write_run_status(config, "running")
        if seed == 6:
            status = "failed" if phase == "offline" else "blocked"
            write_run_status(config, status, details={"reason": "fixture failure"})
            if phase == "offline":
                raise RuntimeError("fixture failure")
            raise OfflineDependencyUnavailable("fixture failure")
        write_run_status(config, "successful")
        return {"status": "successful", "run_id": config["run_id"]}

    monkeypatch.setattr(module, "plan_experiment", lambda *_, **__: schedule)
    monkeypatch.setattr(module, "run_offline", run)
    monkeypatch.setattr(module, "run_online", run)
    results = module.run_experiment(path, project_root=project, smoke=True)
    assert events == [("offline", 6), ("offline", 9), ("online", 6), ("online", 9)]
    assert [result["status"] for result in results] == [
        "failed",
        "successful",
        "blocked",
        "successful",
    ]


def test_resume_uses_saved_configs_and_skips_successful_runs(experiments, monkeypatch):
    from src.artifacts import save_resume, write_run_status

    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    offline, online = module.plan_experiment(path, project_root=project, smoke=True)
    schedule = [offline | {"seed": 5}, offline, online]
    events = []

    def run(config, *, resume=None):
        assert config["data"]["prepared_root"] == offline["data"]["prepared_root"]
        events.append(
            (config["phase"], config["seed"], config["run_id"], resume is not None)
        )
        write_run_status(config, "running")
        if config["seed"] == 6 and config["phase"] == "offline" and resume is None:
            save_resume(
                config,
                {
                    "config": config,
                    "cursor": 1,
                    "method_state": {},
                    "scoring_state": {},
                    "rng_state": {},
                },
            )
            write_run_status(config, "interrupted")
            raise KeyboardInterrupt
        write_run_status(config, "successful")
        return {"status": "successful", "run_id": config["run_id"]}

    monkeypatch.setattr(module, "plan_experiment", lambda *_, **__: schedule)
    monkeypatch.setattr(module, "run_offline", run)
    monkeypatch.setattr(module, "run_online", run)
    with pytest.raises(KeyboardInterrupt):
        module.run_experiment(path, project_root=project, smoke=True)
    path.write_text("experiment_id: smd_comparison\nrandom_seeds: [999]\n")
    profile_path = project / "configs/data_profiles/smd.yaml"
    profile = load_yaml(profile_path)
    profile_path.write_text(yaml.safe_dump(profile | {"prepared_root": "changed"}))
    rows = module.resume_experiment(path, project_root=project)
    assert len(rows) == 3 and all(row["status"] == "successful" for row in rows)
    assert events[1][:3] == events[2][:3]
    assert events[2][3] is True
    assert events[3][0:2] == ("online", 6)
    assert sum(event[1] == 5 for event in events) == 1


@pytest.mark.parametrize(
    "smoke, enabled", [(False, False), (True, True), (False, True)]
)
def test_remote_flags_on_run_and_resume(experiments, monkeypatch, smoke, enabled):
    from src.artifacts import save_resume, write_run_status

    module, project = experiments
    path = project / "configs/experiments/smd_comparison.yaml"
    flags = {"wandb_logging": enabled, "kaggle_backup": enabled}
    definition = load_yaml(path) | flags
    definition.pop("datasets", None)
    definition["cells"] = definition["cells"][:1]
    path.write_text(yaml.safe_dump(definition))
    run_flags, provider_calls = [], []

    def run(config, *, resume=None):
        run_flags.append({name: config[name] for name in flags})
        write_run_status(config, "running")
        if resume is None:
            save_resume(
                config,
                {
                    "config": config,
                    "cursor": 1,
                    "method_state": {},
                    "scoring_state": {},
                    "rng_state": {},
                },
            )
            write_run_status(config, "interrupted")
            raise KeyboardInterrupt
        write_run_status(config, "successful")
        return {"status": "successful", "run_id": config["run_id"]}

    def log(event, *, session=None):
        provider_calls.append(event["kind"])
        return object() if event["kind"] == "start" else None

    def backup(*_):
        provider_calls.append("backup")
        return {"status": "successful"}

    monkeypatch.setattr(module, "run_offline", run)
    monkeypatch.setattr(module, "log_wandb", log)
    monkeypatch.setattr(module, "backup_kaggle", backup)
    with pytest.raises(KeyboardInterrupt):
        module.run_experiment(path, project_root=project, smoke=smoke)
    path.write_text(yaml.safe_dump(definition | dict.fromkeys(flags, not enabled)))
    results = module.resume_experiment(path, project_root=project)
    assert [result["status"] for result in results] == ["successful"]
    active = enabled and not smoke
    assert run_flags == [dict.fromkeys(flags, active)] * 2
    assert provider_calls == (
        ["start", "finish", "backup", "start", "metrics", "finish", "backup"]
        if active
        else []
    )

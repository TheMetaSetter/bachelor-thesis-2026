"""Edit experiment definitions and plan runs from prepared data."""

from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import re

import yaml

from .artifacts import (
    _read_json,
    _write_atomic,
    _write_json,
    allocate_run,
    execution_identity,
    load_resume,
    result_directory,
)
from .config import _CNN_METHODS, _profile_reference, list_names, load_yaml, resolve_run
from .integrations import backup_kaggle, log_wandb
from .offline import run_offline
from .online import run_online
from .prepared_data import iter_batches, load_entity


_ENCODER = {
    "hidden_channels": 64,
    "output_dim": 64,
    "kernel_size": 3,
    "num_layers": 3,
    "dropout": 0.1,
}
_REPORTS = [
    f"reporting_profiles/{name}.yaml"
    for name in ("offline_detection", "online_detection", "uncertainty")
]
_FIELDS = set(
    "experiment_id datasets methods cells random_seeds encoder device reporting_profiles "
    "wandb_logging kaggle_backup".split()
)
_FLAGS = {"wandb_logging", "kaggle_backup"}
_DIMENSIONS = ("dataset", "entity", "method_name", "variant", "seed")


def list_experiments(*, project_root: Path) -> list[str]:
    return list_names("experiments", project_root=project_root)


def _experiment_name(name):
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name
    ):
        raise ValueError("experiment name must be a safe file name")
    return name


def create_experiment(name: str, *, project_root: Path) -> Path:
    name = _experiment_name(name)
    folder = project_root / "configs/experiments"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    definition = {
        "experiment_id": name,
        "datasets": [],
        "methods": [],
        "random_seeds": [6],
        "encoder": _ENCODER,
        "wandb_logging": False,
        "kaggle_backup": False,
        "reporting_profiles": _REPORTS,
    }
    with path.open("x", encoding="utf-8") as output:
        yaml.safe_dump(definition, output, sort_keys=False)
    return path


def _definition(path):
    definition = load_yaml(path)
    _experiment_name(definition.get("experiment_id"))
    if definition["experiment_id"] != Path(path).stem:
        raise ValueError("experiment ID must match its file name")
    if unknown := (definition.keys() - _FIELDS):
        raise ValueError(f"unsupported experiment fields: {sorted(unknown)}")
    for field in _FLAGS:
        if field in definition and type(definition[field]) is not bool:
            raise ValueError(f"{field}: expected true or false")
    seeds = definition.get("random_seeds", [6])
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(item) is not int or item < 0 for item in seeds)
    ):
        raise ValueError("random_seeds: expected nonnegative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("random_seeds: duplicate seed")
    return definition


def _profile_key(project, cell, field):
    folder, key = (
        ("data_profiles", "dataset")
        if field == "datasets"
        else ("method_profiles", "method")
    )
    root = (project / "configs").resolve()
    return load_yaml(_profile_reference(root, folder, cell[f"{folder[:-1]}"]))[key]


def _edit_selection(definition, operation, field, values, project):
    category = "datasets" if field == "datasets" else "methods"
    unknown = set(values) - set(list_names(category, project_root=project))
    if unknown:
        raise ValueError(f"unknown {category}: {sorted(unknown)}")
    cells = definition.get("cells", [])
    current = list(definition.get(field, ()))
    if operation == "add":
        current = list(dict.fromkeys([*current, *values]))
    else:
        current = [value for value in current if value not in values]
        if cells:
            definition["cells"] = [
                cell
                for cell in cells
                if _profile_key(project, cell, field) not in values
            ]
    definition[field] = current


def edit_experiment(
    path: Path, operation: str, field: str, values, *, project_root: Path
) -> dict:
    definition = _definition(path)
    if not values:
        raise ValueError(f"{field}: supply at least one value")
    if operation in {"add", "remove"} and field in {"datasets", "methods"}:
        _edit_selection(definition, operation, field, values, project_root)
    elif operation == "set" and field == "random_seeds":
        if any(not re.fullmatch(r"[0-9]+", value) for value in values):
            raise ValueError("random_seeds: expected nonnegative integers")
        parsed = [int(value) for value in values]
        if len(parsed) != len(set(parsed)):
            raise ValueError("random_seeds: duplicate seed")
        definition[field] = parsed
    elif operation == "set" and field in _FLAGS:
        if len(values) != 1 or values[0] not in {"true", "false"}:
            raise ValueError(f"{field}: use true or false")
        definition[field] = values[0] == "true"
    else:
        raise ValueError(f"unsupported experiment edit: {operation} {field}")
    contents = yaml.safe_dump(definition, sort_keys=False).encode("utf-8")
    _write_atomic(path, lambda file: file.write(contents))
    return definition


def _method_profiles(method):
    if method == "mustard":
        return [
            (f"method_profiles/mustard_{variant}_{phase}.yaml", phase)
            for phase, variants in (
                ("offline", ("o0", "o1", "o2")),
                ("online", ("a0", "a1", "a2")),
            )
            for variant in variants
        ]
    names = ("moment", "moment_linear_probing") if method == "moment" else (method,)
    return [
        (f"method_profiles/baselines/{name}.yaml", phase)
        for phase in ("offline", "online")
        for name in names
    ]


def _selected_cells(definition, project):
    cells = list(definition.get("cells", ()))
    if any(not isinstance(cell, dict) for cell in cells):
        raise ValueError("experiment cells must be mappings")
    datasets = definition.get("datasets", [])
    methods = definition.get("methods", [])
    if any(
        not isinstance(items, list) or len(set(items)) != len(items)
        for items in (datasets, methods)
    ):
        raise ValueError("datasets and methods must be distinct lists")
    explicit = {
        (
            _profile_key(project, cell, "datasets"),
            _profile_key(project, cell, "methods"),
        )
        for cell in cells
    }
    inferred_datasets = list(
        dict.fromkeys(_profile_key(project, cell, "datasets") for cell in cells)
    )
    inferred_methods = list(
        dict.fromkeys(_profile_key(project, cell, "methods") for cell in cells)
    )
    for dataset in dict.fromkeys([*datasets, *inferred_datasets]):
        for method in dict.fromkeys([*methods, *inferred_methods]):
            if (dataset, method) not in explicit:
                yield from _default_cells(project, dataset, method, cells)
    yield from cells


def _dataset_profile(project, dataset, explicit):
    root = (project / "configs").resolve()
    paths = {
        _profile_reference(root, "data_profiles", cell["data_profile"])
        for cell in explicit
        if _profile_key(project, cell, "datasets") == dataset
    }
    if len(paths) > 1:
        raise ValueError(f"{dataset}: select one data profile per experiment")
    return next(
        iter(paths),
        _profile_reference(root, "data_profiles", f"data_profiles/{dataset}.yaml"),
    )


def _default_cells(project, dataset, method, explicit):
    path = _dataset_profile(project, dataset, explicit)
    reference = path.relative_to((project / "configs").resolve()).as_posix()
    data = load_yaml(path)
    manifest = json.loads(
        (project / data["prepared_root"] / "manifest.json").read_text()
    )
    entities = list(manifest["entities"])
    if not entities:
        raise ValueError(f"{dataset}: no prepared entities")
    evaluation = "raw_input" if method == "mustard" else "native"
    for entity in entities:
        for method_profile, phase in _method_profiles(method):
            yield {
                "data_profile": reference,
                "method_profile": method_profile,
                "phase": phase,
                "entity": entity,
                "evaluation_profile": f"evaluation_profiles/{evaluation}_q99_ewma075.yaml",
            }


def _cell_config(definition, cell, project, seed, smoke):
    for key in (
        "seed",
        "experiment_id",
        "run_id",
        "output_dir",
        "offline_parent_dir",
        "execution_mode",
        *_FLAGS,
    ):
        if key in cell:
            raise ValueError(f"experiment cells do not define {key}")
    if cell.get("evaluation_profile") is None:
        raise ValueError("an experiment run requires an evaluation profile")
    method = _profile_key(project, cell, "methods")
    fields = dict(cell)
    fields.update(
        experiment_id=definition["experiment_id"],
        seed=seed,
        device=cell.get("device", definition.get("device", "cpu")),
        execution_mode="smoke" if smoke else "normal",
        **{key: definition.get(key, False) and not smoke for key in _FLAGS},
    )
    if method in _CNN_METHODS:
        fields["encoder"] = definition.get("encoder", _ENCODER)
    if smoke:
        count = 4 if method == "moment" else 64
        caps = {
            "max_epochs_per_stage": 1,
            "max_train_windows": count,
            "max_validation_windows": count,
            "max_test_windows": count,
            "max_online_steps": 4 if method == "moment" else 32,
        }
        fields["limits"] = fields.get("limits", {}) | {
            key: min(value, fields.get("limits", {}).get(key, value))
            for key, value in caps.items()
        }
    resolved = resolve_run(fields, project_root=project)
    if resolved["phase"] == "online" and method == "mustard":
        resolved["reporting_variant"] = (
            f"{resolved['required_offline_variant']}_{resolved['variant']}"
        )
    return resolved


def _dependency_key(run):
    return tuple(run[key] for key in ("dataset", "entity", "method_name", "seed")) + (
        run.get("required_offline_variant")
        if run["phase"] == "online"
        else run["variant"],
        run["profile_paths"]["evaluation"],
    )


def _run_key(run):
    return (
        run["phase"],
        *(_dimension(run, name) for name in _DIMENSIONS),
        run["profile_paths"]["evaluation"],
    )


def _dimension(run, name):
    return run.get("reporting_variant", run[name]) if name == "variant" else run[name]


def _add_dependencies(plans, definition, project, smoke):
    keys = {_dependency_key(run) for run in plans if run["phase"] == "offline"}
    dependencies = []
    for run in plans:
        if run["phase"] != "online" or run.get("required_offline_variant") is None:
            continue
        key = _dependency_key(run)
        if key in keys:
            continue
        path = run["profile_paths"]["method"]
        if run["method_name"] == "mustard":
            path = f"method_profiles/mustard_{run['required_offline_variant'].lower()}_offline.yaml"
        cell = {
            "data_profile": run["profile_paths"]["data"],
            "method_profile": path,
            "evaluation_profile": run["profile_paths"]["evaluation"],
            "phase": "offline",
            "entity": run["entity"],
            "device": run["device"],
        }
        parent = _cell_config(definition, cell, project, run["seed"], smoke)
        keys.add(key)
        dependencies.append(parent)
    return sorted([*plans, *dependencies], key=lambda run: run["phase"] == "online")


def _group_runs(plans):
    counts = defaultdict(set)
    for run in plans:
        prefix = (run["phase"],)
        for dimension in _DIMENSIONS:
            value = _dimension(run, dimension)
            counts[prefix].add(value)
            prefix += (value,)
    for run in plans:
        prefix = (run["phase"],)
        grouping = {}
        for dimension in _DIMENSIONS:
            grouping[dimension] = len(counts[prefix]) > 1
            prefix += (_dimension(run, dimension),)
        run["grouping"] = grouping


def _check_readiness(plans):
    checked = {}
    stages = set()
    for run in plans:
        key = (run["data"]["prepared_root"], run["entity"])
        if key not in checked:
            checked[key] = load_entity(*key)
        entity = checked[key]
        model_stages = (run["method"].get("training") or {}).get("stages")
        if model_stages and run["phase"] == "offline":
            stages.add(sum(stage["epochs"] for stage in model_stages))
        settings = run["data"] | {"limits": run["limits"]}
        required = (
            ("train", "validation", "test")
            if run["phase"] == "offline"
            else ("validation", "test")
        )
        for split in required:
            next(
                iter_batches(
                    entity["splits"][split],
                    settings,
                    phase=run["phase"],
                    seed=run["seed"],
                )
            )
    if len(stages) > 1:
        raise ValueError("fitted neural methods must have equal total epoch counts")


def plan_experiment(
    path: Path, *, project_root: Path, smoke: bool = False
) -> list[dict]:
    definition = _definition(path)
    plans, selected, profiles, keys, failures = [], set(), {}, set(), []
    for cell in _selected_cells(definition, project_root):
        for seed in definition.get("random_seeds", [6]):
            try:
                run = _cell_config(definition, cell, project_root, seed, smoke)
                path = run["profile_paths"]["data"]
                if profiles.setdefault(run["dataset"], path) != path:
                    raise ValueError(
                        f"{run['dataset']}: select one data profile per experiment"
                    )
                key = _run_key(run)
                if key in keys:
                    raise ValueError(f"duplicate planned run: {key}")
                keys.add(key)
                choice = (run["method_name"], run["phase"])
                if not smoke or choice not in selected:
                    plans.append(run)
                    selected.add(choice)
            except (ValueError, OSError, KeyError) as error:
                failures.append(str(error))
    if failures:
        raise ValueError(
            "unsupported experiment combinations:\n"
            + "\n".join(dict.fromkeys(failures))
        )
    if not plans:
        raise ValueError("add at least one dataset and method before planning")
    plans = _add_dependencies(plans, definition, project_root, smoke)
    _group_runs(plans)
    _check_readiness(plans)
    return plans


def _status(config):
    path = result_directory(config, "state") / "run.json"
    return _read_json(path) if path.exists() else None


def _committed_metrics(result):
    return {
        f"{split}/{key}": value
        for split, path in result.get("splits", {}).items()
        for key, value in _read_json(Path(path))["metrics"].items()
    }


def _call_runner(config, *, resume=None, progress_factory=None):
    run = run_offline if config["phase"] == "offline" else run_online
    enabled = config.get("wandb_logging", False)
    event = {
        "enabled": enabled,
        "run_id": config["run_id"],
        "experiment_id": config.get("experiment_id"),
    }
    session, provider = None, None
    if enabled:
        try:
            session = log_wandb(event | {"kind": "start"})
        except RuntimeError:
            provider = {"status": "failed", "reason": "W&B start failed"}
    try:
        display = progress_factory(config) if progress_factory else nullcontext()
        with display as progress:
            options = {"progress": progress} if progress is not None else {}
            result = run(config, resume=resume, **options)
        if session is not None:
            try:
                metrics = _committed_metrics(result)
                log_wandb(
                    event | {"kind": "metrics", "metrics": metrics}, session=session
                )
                provider = {"status": "successful"}
            except (OSError, RuntimeError, KeyError, ValueError):
                provider = {"status": "failed", "reason": "W&B metrics failed"}
    except Exception as error:
        record = _status(config)
        status = record["status"] if record else "failed"
        result = {"run_id": config["run_id"], "status": status, "reason": str(error)}
    finally:
        if session is not None:
            try:
                log_wandb(event | {"kind": "finish"}, session=session)
            except RuntimeError:
                provider = {"status": "failed", "reason": "W&B finish failed"}
        if provider is not None:
            _write_json(result_directory(config, "state") / "wandb.json", provider)
    if provider is not None:
        result["wandb"] = provider
    return result


def _with_backup(runs, execute):
    first = runs[0]
    if not first.get("kaggle_backup"):
        return execute()
    root = Path(first["output_root"])
    try:
        results = execute()
    finally:
        settings = {"kaggle_backup": True, "experiment_id": first["experiment_id"]}
        record = backup_kaggle(root, settings)
        _write_json(root / "backup.json", record)
    if results:
        results[-1]["kaggle"] = record
    return results


def run_experiment(
    path: Path, *, project_root: Path, smoke: bool = False, progress_factory=None
) -> list[dict]:
    plan = plan_experiment(path, project_root=project_root, smoke=smoke)
    runs = [allocate_run(config, project_root=project_root) for config in plan]
    attempt = Path(runs[0]["output_root"]) / "attempts" / f"{runs[0]['run_id']}.json"
    _write_json(
        attempt, {"schema_version": 1, "identity": execution_identity(), "runs": runs}
    )
    return _with_backup(
        runs,
        lambda: [
            _call_runner(config, progress_factory=progress_factory) for config in runs
        ],
    )


def _config_paths(config):
    """Restore path values from the experiment schedule's JSON form."""
    for field in ("project_root", "output_root"):
        config[field] = Path(config[field])
    config["data"]["prepared_root"] = Path(config["data"]["prepared_root"])
    for field in ("manifest_path", "scaler_path"):
        config["prepared_identity"][field] = Path(config["prepared_identity"][field])
    return config


def _resume_attempt(runs, progress_factory=None):
    results = []
    for config in runs:
        record = _status(config)
        if record is None:
            results.append(_call_runner(config, progress_factory=progress_factory))
            continue
        status = record["status"]
        if status == "successful":
            results.append({"run_id": config["run_id"], "status": status})
            continue
        try:
            snapshot = load_resume(config)
        except FileNotFoundError:
            if status != "blocked" or config["phase"] != "online":
                results.append(
                    {
                        "run_id": config["run_id"],
                        "status": status,
                        "reason": "no committed continuation",
                    }
                )
                continue
            snapshot = {"unstarted": True}
        except ValueError as error:
            results.append(
                {"run_id": config["run_id"], "status": "failed", "reason": str(error)}
            )
            continue
        results.append(
            _call_runner(config, resume=snapshot, progress_factory=progress_factory)
        )
    return results


def resume_experiment(
    path: Path, *, project_root: Path, progress_factory=None
) -> list[dict]:
    """Continue all unfinished recorded attempts with their original settings."""
    name = _experiment_name(path.stem)
    attempts = sorted(
        file
        for mode in ("experiments", "smoke")
        for file in (project_root / "outputs" / mode / name / "attempts").glob("*.json")
    )
    if not attempts:
        raise ValueError(f"{name}: no recorded execution to resume")
    identity = execution_identity()
    results = []
    for attempt in attempts:
        saved = _read_json(attempt)
        if saved["schema_version"] != 1:
            raise ValueError(f"{name}: invalid attempt schema")
        runs = [_config_paths(config) for config in saved["runs"]]
        statuses = [(_status(config) or {}).get("status") for config in runs]
        if all(status == "successful" for status in statuses):
            continue
        if saved["identity"] != identity:
            results.extend(
                {
                    "run_id": config["run_id"],
                    "status": "failed",
                    "reason": "different code or dependencies",
                }
                for config, status in zip(runs, statuses, strict=True)
                if status != "successful"
            )
            continue
        results.extend(
            _with_backup(runs, lambda: _resume_attempt(runs, progress_factory))
        )
    return results

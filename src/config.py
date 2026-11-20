"""Read and resolve the thesis runtime configuration."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any

import yaml

from . import protocols


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys, including keys in nested mappings."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f"duplicate YAML key {key!r}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping. Include its path in format errors."""
    path = Path(path)
    try:
        result = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(result, dict):
            raise ValueError("expected a mapping at the root")
    except (yaml.YAMLError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    return result


def list_names(kind: str, *, project_root: Path) -> list[str]:
    """List names that the command line can discover."""
    directories = {
        "datasets": "data_profiles",
        "methods": "method_profiles",
        "experiments": "experiments",
        "standalones": "standalones",
    }
    if kind not in directories:
        raise ValueError(f"unknown discovery category: {kind}")
    folder = project_root / "configs" / directories[kind]
    if kind == "methods":
        return sorted({load_yaml(path)["method"] for path in folder.rglob("*.yaml")})
    return sorted(path.stem for path in folder.glob("*.yaml") if path.is_file())


_PHASES = {"offline", "online"}
_SPLITS = {"train", "validation", "test", "online"}
_CELL_REQUIRED = set("data_profile method_profile evaluation_profile entity phase seed device".split())
_CELL_OPTIONAL = set(
    "encoder experiment_id run_id output_dir offline_parent_dir required_offline_variant limits "
    "wandb_logging kaggle_backup execution_mode".split()
)
_DATA_REQUIRED = set(
    "dataset prepared_root window_size train_stride validation_stride test_stride online_stride "
    "tail_policy gap_policy expected_interval_seconds batch_size shuffle_train".split()
)
_DATA_OPTIONAL = {"method_window_sizes", "method_strides", "synthetic_profile"}
_EVALUATION_REQUIRED = {
    "score_space", "point_score_transform", "metrics", "test_label_usage",
    "point_adjustment", "calibration_label_policy", "online_reporting_score",
}
_EVALUATION_OPTIONAL = set(
    "offline_threshold_split online_threshold_split offline_threshold_quantile online_threshold_quantile "
    "offline_point_threshold_source offline_window_threshold_source offline_overlap_reduction "
    "online_point_threshold_source adaptation_threshold_split triage vus_max_buffer_size "
    "vus_num_thresholds fpr_budgets uncertainty_fields uncertainty_statistics "
    "ewma_current_weight ewma_previous_weight".split()
)
_CNN_METHODS = {"mustard", "redlamp", "candi", "m2n2"}


def _fields(value, required, optional, source):
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: expected a mapping")
    extra = value.keys() - required - optional
    if extra:
        raise ValueError(f"{source}: unknown fields {sorted(extra, key=str)}")
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{source}: missing fields {sorted(missing)}")


def _integer(value, minimum, field):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field}: expected an integer >= {minimum}")


def _number(value, lower, upper, field, *, closed=False):
    valid = type(value) in (int, float) and isfinite(value)
    if not valid or not (lower <= value <= upper if closed else lower < value < upper):
        raise ValueError(f"{field}: invalid number {value!r}")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonempty string")
    return value


def _profile_reference(config_root, category, reference):
    relative = Path(_text(reference, f"{category} reference"))
    path = (config_root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or path.suffix != ".yaml":
        raise ValueError(f"invalid {category} reference: {reference}")
    if not path.is_relative_to(config_root / category):
        raise ValueError(f"{category} reference escapes its folder: {reference}")
    return path


def _validate_cell(cell):
    _fields(cell, _CELL_REQUIRED, _CELL_OPTIONAL, "run")
    for name in ("entity", "phase", "device"):
        _text(cell[name], name)
    if cell["phase"] not in _PHASES:
        raise ValueError(f"phase: unknown value {cell['phase']!r}")
    _integer(cell["seed"], 0, "seed")
    if "run_id" in cell:
        run_id = _text(cell["run_id"], "run_id")
        if "experiment_id" in cell or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
            raise ValueError("run_id: expected a safe standalone ID")
    if "experiment_id" in cell and ("output_dir" in cell or "offline_parent_dir" in cell):
        raise ValueError("experiment runs derive their output and parent paths")
    for field in ("wandb_logging", "kaggle_backup"):
        if field in cell and type(cell[field]) is not bool:
            raise ValueError(f"{field}: expected a boolean")
    limits = cell.get("limits", {})
    counts = set("max_epochs_per_stage max_train_windows max_validation_windows max_test_windows max_online_steps".split())
    range_fields = {"stream_start", "stream_end"}
    _fields(limits, set(), counts | range_fields, "limits")
    for key in limits.keys() & counts:
        _integer(limits[key], 1, f"limits.{key}")
    if limits.keys() & range_fields:
        if not range_fields <= limits.keys() or cell["phase"] != "online":
            raise ValueError("limits: use both stream bounds in an online run")
        _integer(limits["stream_start"], 0, "limits.stream_start")
        _integer(limits["stream_end"], limits["stream_start"] + 1, "limits.stream_end")


def _validate_data(data, path):
    _fields(data, _DATA_REQUIRED, _DATA_OPTIONAL, path)
    for key in {"window_size", "batch_size"} | {f"{split}_stride" for split in _SPLITS}:
        _integer(data[key], 1, f"{path}: {key}")
    _fields(data["tail_policy"], _SPLITS, set(), f"{path}: tail_policy")
    if any(value not in {"drop", "end_align"} for value in data["tail_policy"].values()):
        raise ValueError(f"{path}: invalid tail policy")
    if data["tail_policy"]["online"] != "drop":
        raise ValueError(f"{path}: online tail policy must be drop")
    interval = data["expected_interval_seconds"]
    if interval is not None:
        _number(interval, 0, float("inf"), f"{path}: expected_interval_seconds")
    expected_gap = "preserve_source_rows" if interval is None else "break_at_time_gaps"
    if data["gap_policy"] != expected_gap or type(data["shuffle_train"]) is not bool:
        raise ValueError(f"{path}: invalid gap policy or training shuffle")
    data.setdefault("method_window_sizes", {"moment": 512})
    data.setdefault("method_strides", {})
    for method, size in data["method_window_sizes"].items():
        _integer(size, 1, f"{path}: method_window_sizes.{method}")
    for method, strides in data["method_strides"].items():
        _fields(strides, set(), _SPLITS, f"{path}: method_strides.{method}")
        for split, stride in strides.items():
            _integer(stride, 1, f"{path}: method_strides.{method}.{split}")


def _validate_encoder(cell, method):
    if method not in _CNN_METHODS:
        if cell.get("encoder") is not None:
            raise ValueError(f"encoder: {method} does not use the shared CNN")
        return
    encoder = cell.get("encoder")
    counts = {"hidden_channels", "output_dim", "kernel_size", "num_layers"}
    _fields(encoder, counts | {"dropout"}, set(), "encoder")
    for key in counts:
        _integer(encoder[key], 2 if key == "num_layers" else 1, f"encoder.{key}")
    _number(encoder["dropout"], 0, 1, "encoder.dropout", closed=True)
    if encoder["dropout"] == 1:
        raise ValueError("encoder.dropout: must be less than one")


def _validate_evaluation(evaluation, method, phase, path):
    _fields(evaluation, _EVALUATION_REQUIRED, _EVALUATION_OPTIONAL, path)
    score_space = evaluation["score_space"]
    name = method["method"]
    if score_space not in {"raw_input", "normalized_input", "method_native", "latent"}:
        raise ValueError(f"{path}: unknown score_space")
    if (name == "mustard") == (score_space == "method_native"):
        raise ValueError(f"{path}: {name} cannot use {score_space}")
    transform = evaluation["point_score_transform"]
    if transform not in {"identity", "validation_median_iqr", "validation_mad_logistic"}:
        raise ValueError(f"{path}: unsupported point_score_transform")
    if score_space == "latent" and (
        transform != "identity"
        or phase == "offline" and not method["parameters"]["continuous"]["enabled"]
    ):
        raise ValueError(f"{path}: latent MSE requires active continuous retrieval and identity")
    weights = ("ewma_previous_weight", "ewma_current_weight")
    if sum(key in evaluation for key in weights) == 1:
        raise ValueError(f"{path}: specify both EWMA weights")
    for key, default in zip(weights, (0.25, 0.75), strict=True):
        _number(evaluation.setdefault(key, default), 0, 1, f"{path}: {key}", closed=True)
    if abs(sum(evaluation[key] for key in weights) - 1) > 1e-12:
        raise ValueError(f"{path}: EWMA weights must sum to one")
    reduction = evaluation.setdefault("offline_overlap_reduction", "mean")
    source = evaluation.setdefault("online_point_threshold_source", "phase_calibration")
    if reduction not in {"mean", "last_ewma_estimate"} or source not in {"phase_calibration", "offline_parent"}:
        raise ValueError(f"{path}: invalid threshold source or overlap reduction")
    if source == "offline_parent" and (name != "mustard" or reduction != "last_ewma_estimate"
                                      or "online_threshold_quantile" in evaluation):
        raise ValueError(f"{path}: invalid configuration for an offline-parent point threshold")
    if phase != "online" or source != "offline_parent":
        key = f"{phase}_threshold_quantile"
        _number(evaluation.get(key), 0, 1, f"{path}: {key}")
    if evaluation.get(f"{phase}_threshold_split") not in (
        {"train", "validation", "synthetic_validation"} if phase == "offline" else {"train", "validation"}
    ):
        raise ValueError(f"{path}: invalid calibration split")
    if (evaluation["test_label_usage"] != "metrics_only" or evaluation["point_adjustment"] is not False
            or evaluation["online_reporting_score"] != "last_ewma_estimate"
            or evaluation["calibration_label_policy"] != "exclude_known_anomalies"):
        raise ValueError(f"{path}: invalid scoring policy")
    if name != "m2n2":
        evaluation["adaptation_threshold_split"] = None
    elif evaluation.get("adaptation_threshold_split") != "train":
        raise ValueError(f"{path}: M2N2 adaptation calibration requires train")


def _prepared_identity(root, data, entity):
    prepared_root = (root / _text(data["prepared_root"], "prepared_root")).resolve()
    manifest_path = prepared_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete" or entity not in manifest.get("entities", {}):
        raise ValueError(f"{manifest_path}: no complete prepared data for {entity!r}")
    features = manifest["entities"][entity]["features"]
    scaler_path = f"{entity}/scaler.json"
    if scaler_path not in manifest["files"]:
        raise ValueError(f"{manifest_path}: missing scaler reference for {entity!r}")
    data["prepared_root"] = prepared_root
    return {
        "dataset": data["dataset"], "entity": entity, "version_id": manifest["version_id"],
        "manifest_path": manifest_path, "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "scaler_path": prepared_root / scaler_path, "scaler_sha256": manifest["files"][scaler_path],
        "features": list(features),
        "window_size": data["window_size"],
    }


def resolve_run(cell: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    """Resolve one cell without writing files, allocating IDs, or selecting a parent."""
    _validate_cell(cell)
    root = project_root.resolve()
    resolved = deepcopy(dict(cell))
    config_root = (root / "configs").resolve()
    paths = {}
    for key in ("data", "method", "evaluation"):
        field = f"{key}_profile"
        reference = resolved.pop(field)
        if key == "evaluation" and reference is None:
            resolved[key] = None
            paths[key] = None
            continue
        path = _profile_reference(config_root, field + "s", reference)
        paths[key] = path.relative_to(config_root).as_posix()
        resolved[key] = load_yaml(path)
    data, method, evaluation = (resolved[key] for key in ("data", "method", "evaluation"))
    _validate_data(data, paths["data"])
    protocols.validate_method_profile(method)
    phase = cell["phase"]
    name = method["method"]
    if phase not in method["phases"]:
        raise ValueError(f"{paths['method']}: {name} does not support {phase}")
    _validate_encoder(cell, name)
    if evaluation is not None:
        _validate_evaluation(evaluation, method, phase, paths["evaluation"])
    _resolve_window_settings(data, name)
    dependency = method.get("required_offline_variant")
    if "required_offline_variant" in cell:
        selected = cell["required_offline_variant"]
        if phase != "online" or (selected != dependency and (name != "mustard" or selected not in ("O0", "O1", "O2"))):
            raise ValueError("required_offline_variant: choose a compatible offline variant")
        dependency = selected
    if dependency is not None:
        resolved["required_offline_variant"] = dependency
    resolved.update(
        project_root=root, dataset=data["dataset"], method_name=name, variant=method["variant"],
        profile_paths=paths, prepared_identity=_prepared_identity(root, data, cell["entity"]),
    )
    resolved.setdefault("limits", {})
    _bind_method_inputs(resolved, config_root)
    protocols.validate_run_input(resolved)
    return resolved


def _bind_method_inputs(config, config_root):
    name = config["method_name"]
    if config["phase"] == "offline" and name in {"mustard", "redlamp"}:
        path = _profile_reference(config_root, "data_profiles/synthetic_profiles", config["data"]["synthetic_profile"])
        config["synthetic"] = load_yaml(path)
    if name == "mustard" and config["phase"] == "online":
        variant = config["required_offline_variant"].lower()
        path = config_root / "method_profiles" / f"mustard_{variant}_offline.yaml"
        source = load_yaml(path)
        protocols.validate_method_profile(source)
        if source["method"] != "mustard" or source["variant"].lower() != variant:
            raise ValueError("MUSTARD offline source profile does not match the requested variant")
        verification = config["method"]["parameters"]["verification"]
        continuous = source["parameters"]["continuous"]
        if verification and verification["signature_top_k"] > continuous["num_prototypes"]:
            raise ValueError("MUSTARD signature_top_k exceeds the prototype count")
        config["source_method"] = source


def _resolve_window_settings(data, name):
    data["window_size"] = data["method_window_sizes"].get(name, data["window_size"])
    for split, value in data["method_strides"].get(name, {}).items():
        data[f"{split}_stride"] = value


def create_method(config: dict, *, encoder=None):
    """Load only the chosen constructor and its training settings."""
    from importlib import import_module

    constructors = {
        "mustard": (".methods.mustard.model", "Mustard"),
        "redlamp": (".methods.redlamp", "RedLamp"),
        "stumpy_channel_ab": (".methods.stumpy_channel_ab", "StumpyChannelAB"),
        "kmeans_ad": (".methods.kmeans_ad", "KMeansAD"),
        "iforest": (".methods.iforest", "IsolationForestAD"),
        "candi": (".methods.candi", "CANDI"),
        "m2n2": (".methods.m2n2", "M2N2"),
        "moment": (".methods.moment", "Moment"),
    }
    name = config["method_name"]
    if name not in constructors:
        raise ValueError(f"unsupported method: {name}")
    settings = config
    if name in {"mustard", "redlamp"} and config["phase"] == "offline" and "synthetic" not in config:
        root = (config["project_root"] / "configs").resolve()
        reference = config["data"].get("synthetic_profile")
        path = _profile_reference(root, "data_profiles/synthetic_profiles", reference)
        settings = config | {"synthetic": load_yaml(path)}
    module, class_name = constructors[name]
    return getattr(import_module(module, __package__), class_name)(settings, encoder=encoder)

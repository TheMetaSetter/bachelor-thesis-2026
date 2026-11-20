"""Fit, calibrate, and evaluate one offline run on prepared data."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    _hash, _json_bytes, _write_json, capture_rng, close_resume, execution_identity,
    load_checkpoint, load_resume, restore_rng, result_directory, run_lock,
    save_resume, seed_execution, write_result, write_run_status,
)
from .config import _profile_reference, create_method, load_yaml
from .evaluation_metrics import compute_metrics
from .methods.augmentation import augment
from .prepared_data import _segments, iter_batches, load_entity
from . import protocols
from .scoring import _eligible_record, aggregate_points, calibrate, score_batch


def _checked_checkpoint(config, directory, path, role):
    path = Path(path).resolve()
    if path.parent != directory.resolve():
        raise ValueError("checkpoint is outside the run state directory")
    descriptor = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    identity = descriptor["identity"]
    expected = {
        "method": config["method_name"], "variant": config["variant"],
        "prepared_identity": config["prepared_identity"],
    }
    for name in expected.keys() & identity.keys():
        if _json_bytes(expected[name]) != _json_bytes(identity[name]):
            raise ValueError(f"checkpoint {name} does not match the run")
    state = load_checkpoint(path, expected_identity=identity | {"role": role})
    return state, {"path": path, "role": role, "sha256": descriptor["sha256"]}


def _restore_best(method, config, directory, fit):
    if config["method_name"] == "moment" and config["variant"] == "zero_shot":
        identity = method.pretrained_identity
        return {"role": "pretrained", "sha256": identity["weights_sha256"], "identity": identity}
    definitions = (config["method"].get("training") or {}).get("stages", [{"name": "fit"}])
    stages = fit["stages"]
    if [item["name"] for item in stages] != [item["name"] for item in definitions]:
        raise ValueError("fit did not produce all required stage checkpoints")
    if fit["selected_checkpoint"] != stages[-1]["best_checkpoint"]:
        raise ValueError("the selected checkpoint is not the final stage best")
    for stage in stages:
        name = stage["name"]
        _checked_checkpoint(config, directory, stage["initial_checkpoint"], f"{name}.initial")
        state, checkpoint = _checked_checkpoint(config, directory, stage["best_checkpoint"], f"{name}.best")
    method.load_state_dict(state)
    return checkpoint


def _scored_batches(method, split, config, *, synthetic=None):
    settings = config["data"] | {"limits": config.get("limits", {}), "shuffle_train": False}
    for batch in iter_batches(split, settings, phase="offline", seed=config["seed"]):
        method_batch = {key: value for key, value in batch.items() if key != "labels"}
        mask = None
        if synthetic is not None:
            recipe, generator = synthetic
            values, labels = augment(torch.as_tensor(batch["x"], dtype=torch.float32), recipe, generator=generator)
            recipe["class_rotation_offset"] = labels["class_rotation_offset"]
            method_batch["x"] = values
            mask = np.asarray(labels["synthetic_anomaly_mask"])
        raw = method.score(method_batch)
        protocols.validate_calibration_input(raw, {})
        indices = np.asarray(raw["indices"])
        schedule = batch["indices"] if raw["score_kind"] == "window" else batch["indices"][:, -1]
        if not np.array_equal(indices, schedule):
            raise ValueError("method scores do not match the prepared window schedule")
        labels = np.asarray(split["labels"])[indices]
        yield raw | {"labels": labels, "source_window_labels": split["labels"][batch["indices"]]}, mask


def _calibration_input(method, split, config, synthetic):
    records, masks = zip(*_scored_batches(method, split, config, synthetic=synthetic), strict=True)
    values = {name: np.concatenate([record[name] for record in records])
              for name in ("raw_scores", "indices", "labels", "source_window_labels")}
    if synthetic is not None:
        values["generated_mask"] = np.concatenate(masks)
    return values | {"length": len(split["x"]), "score_kind": records[0]["score_kind"]}


def _synthetic_thresholds(raw, settings, identity):
    selected, counts = _eligible_record(raw)
    source_ok = ~(raw["labels"] == 1).any(axis=1)
    mask = raw["generated_mask"][source_ok]
    if settings.get("point_score_transform", "identity") != "identity":
        raise ValueError("synthetic SMD calibration requires the identity score transform")
    if selected["score_kind"] != "window" or mask.shape != selected["raw_scores"].shape:
        raise ValueError("synthetic calibration requires aligned point scores and masks")
    # The reference reduces overlapping scores first and marks an index if any window masked it.
    points = aggregate_points(iter([selected]), length=raw["length"], settings=settings)
    affected = np.zeros(raw["length"], dtype=bool)
    np.logical_or.at(affected, selected["indices"].ravel(), mask.ravel())
    point_scores = points["scores"][points["coverage"] & ~affected]
    window_scores = selected["raw_scores"].mean(axis=1)[~mask.any(axis=1)]
    if not len(point_scores) or not len(window_scores):
        raise ValueError("synthetic calibration has no eligible point or window scores")
    quantile = settings["offline_threshold_quantile"]
    return {
        **identity, **counts, "kind": "calibration", "source_split": "synthetic_validation_normal",
        "score_space": settings["score_space"], "transform": {"name": "identity"},
        "point_threshold": float(np.quantile(point_scores, quantile)),
        "window_threshold": float(np.quantile(window_scores, settings["triage"]["window_quantile"])),
        "normal_data_assumption": counts["unknown_label_windows"] > 0,
        "quantile": quantile, "overlap_reduction": settings.get("offline_overlap_reduction", "mean"),
        "ewma_previous_weight": settings.get("ewma_previous_weight", 0.25),
        "ewma_current_weight": settings.get("ewma_current_weight", 0.75),
    }


def _fit_calibration(method, entity, config, checkpoint):
    settings = config["evaluation"]
    source = settings["offline_threshold_split"]
    identity = {"schema_version": 1, "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_role": checkpoint["role"], "method_name": config["method_name"],
                "variant": config["variant"], "prepared_identity": config["prepared_identity"],
                "data": config["data"], "execution_mode": config.get("execution_mode", "normal"),
                "limits": config.get("limits", {})}
    if source == "synthetic_validation":
        recipe = config.get("synthetic")
        if recipe is None:
            root = Path(config["project_root"]) / "configs"
            reference = _profile_reference(root, "data_profiles/synthetic_profiles", config["data"]["synthetic_profile"])
            recipe = load_yaml(reference)
        identity["generation"] = {"source_split": "validation", "seed": config["seed"], "settings": recipe}
        generator = torch.Generator().manual_seed(config["seed"])
        raw = _calibration_input(method, entity["splits"]["validation"], config, (dict(recipe), generator))
        result = _synthetic_thresholds(raw, settings, identity)
    elif source in {"train", "validation"}:
        raw = _calibration_input(method, entity["splits"][source], config, None)
        result = calibrate(raw | {"source_split": source}, settings, identity=identity)
    else:
        raise ValueError(f"unsupported offline calibration source: {source}")
    protocols.validate_calibration_output(result, config)
    return result


def _evaluate(method, split, config, calibration):
    settings = config["evaluation"]
    uncertainty = defaultdict(list)

    def score_records():
        state = None
        for record, _ in _scored_batches(method, split, config):
            scored, state = score_batch(record, calibration, settings, state=state)
            for key, value in scored.get("uncertainty", {}).items():
                uncertainty[key].append(np.asarray(value).ravel())
            yield scored

    points = aggregate_points(score_records(), length=len(split["x"]), settings=settings)
    points["segments"] = _segments(split, config["data"])
    points["uncertainty"] = {key: np.concatenate(parts) for key, parts in uncertainty.items()}
    labels = split["labels"]
    protocols.validate_run_input(config | {"kind": "evaluation", "scores": points["scores"], "labels": labels})
    result = compute_metrics(points, labels, settings | {"threshold": calibration["point_threshold"]})
    result.update({key: config[key] for key in protocols.IDENTITY_KEYS})
    result.update(status="successful", point_scores=points["scores"], calibration_source=calibration["source_split"])
    protocols.validate_run_output(result, config)
    return result


def _start_offline(config, resume, identity):
    directory = result_directory(config, "state")
    metadata = directory / "run.json"
    if resume is not None:
        committed = load_resume(config)
        if committed["cursor"] != resume["cursor"] or _json_bytes(committed["config"]) != _json_bytes(config):
            raise ValueError("offline resume does not match the committed run")
        if committed["execution_identity"] != identity:
            raise ValueError("offline resume has different code or dependency versions")
        if json.loads(metadata.read_text())["status"] == "running":
            write_run_status(config, "interrupted", details={"reason": "previous writer stopped"})
        resume = committed
    elif metadata.exists():
        raise ValueError("existing offline run requires explicit resume")
    write_run_status(config, "running")
    return resume


def _check_cached_results(paths):
    for path in paths.values():
        record = json.loads(Path(path).read_text())
        file = Path(path).parent / record["point_scores_path"]
        if _hash(file) != record["point_scores_sha256"]:
            raise ValueError(f"split result hash mismatch: {file}")


def _fit_references(fit_state):
    paths = {}
    for stage in fit_state.get("stages", ()):
        for role in ("initial", "best"):
            paths[f"{stage['name']}.{role}"] = stage[f"{role}_checkpoint"]
    name = fit_state.get("stage", "fit")
    for key, role in (("initial", "initial"), ("best", "best"), ("selected_checkpoint", "best")):
        value = fit_state.get(key)
        if isinstance(value, (str, Path)):
            paths[f"{name}.{role}"] = value
    return paths


def _prune_candidates(directory, active_paths):
    active = {Path(path).resolve() for path in active_paths}
    for candidate in directory.glob("*.best.*.pt"):
        if candidate.resolve() not in active:
            candidate.with_suffix(".pt.json").unlink(missing_ok=True)
            candidate.unlink()


def _execute_offline(config, snapshot, identity):
    directory = result_directory(config, "state")
    entity = load_entity(config["data"]["prepared_root"], config["entity"])
    expected = dict(config["prepared_identity"])
    actual = entity["identity"] | {"features": entity["features"]}
    if any(expected.get(key) != value for key, value in actual.items()):
        raise ValueError("prepared data does not match the resolved identity")
    seed_execution(config["seed"], config["device"])
    method = create_method(config | {"scaler": entity["scaler"]})
    state = dict(snapshot or {"config": config, "cursor": 0, "method_state": None, "scoring_state": {},
                              "execution_identity": identity, "results": {}})
    if snapshot:
        restore_rng(snapshot["rng_state"], config["device"])

    def commit(**updates):
        state.update(updates, cursor=state["cursor"] + 1, rng_state=capture_rng(config["device"]))
        save_resume(config, state)

    def commit_fit(fit_state):
        references = _fit_references(fit_state)
        commit(method_state=fit_state, role_references=references)
        _prune_candidates(directory, references.values())

    if not snapshot:
        commit()
    splits = entity["splits"]
    for name, split in splits.items():
        splits[name] = split | {"segments": _segments(split, config["data"])}
    if "fit" not in state:
        fit = None if config["method_name"] == "moment" and config["variant"] == "zero_shot" else method.fit(
            splits["train"], splits["validation"], checkpoint_dir=directory, resume=state["method_state"],
            on_checkpoint=commit_fit,
        )
        checkpoint = _restore_best(method, config, directory, fit)
        references = _fit_references(fit or {})
        commit(fit=fit, role_references=references | {"selected": checkpoint}, method_state=None)
        _prune_candidates(directory, references.values())
    else:
        checkpoint = _restore_best(method, config, directory, state["fit"])
    if config["evaluation"] is not None:
        _run_offline_splits(method, entity, config, state, checkpoint, commit)
    _check_cached_results(state["results"])
    return {
        "status": "successful", "run_id": config["run_id"], "splits": state["results"],
        "selected_checkpoint": checkpoint.get("path"), "checkpoint": checkpoint,
        "stages": (state["fit"] or {}).get("stages", []), "calibration": state.get("calibration_path"),
        "calibration_sha256": state.get("calibration_sha256"), "execution_identity": identity,
    }


def _run_offline_splits(method, entity, config, state, checkpoint, commit):
    directory = result_directory(config, "state")
    if not state["scoring_state"]:
        initial_state = method.state_dict()
        rng_state = capture_rng(config["device"])
        calibration = _fit_calibration(method, entity, config, checkpoint)
        method.load_state_dict(initial_state)
        restore_rng(rng_state, config["device"])
        path = _write_json(directory / "calibration.json", calibration)
        commit(scoring_state=calibration, calibration_path=path, calibration_sha256=_hash(path),
               evaluation_state=method.state_dict())
    else:
        path = Path(state["calibration_path"])
        if _hash(path) != state["calibration_sha256"]:
            raise ValueError("offline calibration hash mismatch")
        method.load_state_dict(state["evaluation_state"])
        restore_rng(state["rng_state"], config["device"])
    for name in ("train", "validation", "test"):
        if name not in state["results"]:
            result = _evaluate(method, entity["splits"][name], config, state["scoring_state"])
            path = write_result(config, name, result)
            commit(results=state["results"] | {name: path}, evaluation_state=method.state_dict())


def run_offline(config: dict, *, resume: dict | None = None) -> dict:
    """Complete one allocated offline run, or continue its committed state."""
    protocols.validate_run_input(config)
    if config["phase"] != "offline":
        raise ValueError("offline execution requires an offline run")
    identity = execution_identity()
    with run_lock(config):
        snapshot = _start_offline(config, resume, identity)
        try:
            result = _execute_offline(config, snapshot, identity)
        except KeyboardInterrupt:
            write_run_status(config, "interrupted")
            raise
        except Exception as error:
            write_run_status(config, "failed", details={"reason": str(error)})
            raise
        write_run_status(config, "successful", details=result)
        close_resume(config)
        return result

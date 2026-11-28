"""Score and update one online run in arrival order."""

import signal
from collections import defaultdict
from contextlib import contextmanager
from itertools import islice
import json
from pathlib import Path
from threading import current_thread, main_thread
from uuid import uuid4

import numpy as np

from .artifacts import (
    _hash,
    _json_bytes,
    _write_json,
    capture_rng,
    close_resume,
    execution_identity,
    load_resume,
    restore_rng,
    result_directory,
    run_lock,
    save_checkpoint,
    save_resume,
    seed_execution,
    select_offline_parent,
    write_result,
    write_run_status,
)
from .config import create_method
from .evaluation_metrics import compute_metrics
from .offline import _check_cached_results, _checked_checkpoint
from .prepared_data import _segments, _window_schedule, iter_batches, load_entity
from .progress import evaluation_stats
from . import protocols
from .scoring import calibrate, score_batch


class OfflineDependencyUnavailable(ValueError):
    """No eligible completed offline run is available."""


def classify_online_window(
    input_window_score: float,
    latent_window_score: float,
    thresholds: dict,
) -> str:
    required = (
        "input_window_threshold",
        "latent_window_low_threshold",
        "latent_window_high_threshold",
    )
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise KeyError(f"full-spec triage thresholds missing keys: {missing}")
    input_threshold = float(thresholds[required[0]])
    latent_low = float(thresholds[required[1]])
    latent_high = float(thresholds[required[2]])
    if latent_low > latent_high:
        raise ValueError("latent low threshold must not exceed high threshold")
    if input_window_score <= input_threshold:
        return "normal"
    if latent_window_score <= latent_low:
        return "hard_old_normality"
    if latent_window_score <= latent_high:
        return "gray_zone"
    return "strong_anomaly"


def _windows(split, config):
    settings = config["data"] | {"limits": config.get("limits", {})}
    return iter_batches(split, settings, phase="online", seed=config["seed"])


def _raw_score(method, batch):
    raw = method.score(batch)
    indices = (
        batch["indices"] if raw["score_kind"] == "window" else batch["indices"][:, -1]
    )
    if not np.array_equal(raw["indices"], indices):
        raise ValueError("method scores do not match the prepared online window")
    return raw


def _load_parent(method, config, identity):
    try:
        parent = select_offline_parent(
            config | {"execution_identity": identity},
            project_root=Path(config["project_root"]),
        )
    except ValueError as error:
        if str(error) != "no compatible successful offline parent":
            raise
        raise OfflineDependencyUnavailable(str(error)) from error
    details = parent["details"]
    if details.get("execution_identity") != identity:
        raise ValueError("offline parent has different code or dependency versions")
    fit_role = "stage_b.best" if config["method_name"] == "mustard" else "fit.best"
    source_config = config | {"variant": config["required_offline_variant"]}
    method_state, checkpoint = _checked_checkpoint(
        source_config,
        Path(parent["state_dir"]),
        details["selected_checkpoint"],
        fit_role,
    )
    if _json_bytes(checkpoint) != _json_bytes(details["checkpoint"]):
        raise ValueError("offline parent checkpoint reference has changed")
    path = Path(details["calibration"])
    checksum = _hash(path)
    calibration = json.loads(path.read_text())
    if (
        checksum != details["calibration_sha256"]
        or calibration.get("checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise ValueError("offline parent calibration does not match its checkpoint")
    method.load_state_dict(method_state)
    reference = {
        "run_id": parent["run_id"],
        "checkpoint": checkpoint,
        "calibration": path,
        "calibration_sha256": checksum,
    }
    return calibration, reference


def _replay_calibration(
    method, split, config, settings, checkpoint, parent_calibration
):
    method_state, rng_state = method.state_dict(), capture_rng(config["device"])
    items, inputs, latents = [], [], []
    excluded, unknown = 0, 0
    for batch in _windows(split, config):
        labels = np.asarray(split["labels"])[batch["indices"]]
        batch.pop("labels", None)
        if np.any(labels == 1):
            excluded += 1
            continue
        unknown += int(np.any(np.isnan(labels)))
        raw = _raw_score(method, batch)
        indices = np.asarray(raw["indices"])
        items.append(raw | {"labels": split["labels"][indices]})
        if "input_window_score" in raw.get("auxiliary", {}):
            inputs.extend(raw["auxiliary"]["input_window_score"])
            latents.extend(raw["auxiliary"]["latent_window_score"])
    method.load_state_dict(method_state)
    restore_rng(rng_state, config["device"])
    if not items:
        raise ValueError("online calibration has no eligible windows")
    if parent_calibration is None:
        values = {
            name: np.concatenate([item[name] for item in items])
            for name in ("indices", "raw_scores", "labels")
        }
        values.update(
            length=len(split["x"]),
            score_kind=items[0]["score_kind"],
            source_split=split["identity"]["split"],
        )
        parent_calibration = calibrate(
            values,
            settings,
            identity={
                "schema_version": 1,
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_role": checkpoint["role"],
                "prepared_identity": config["prepared_identity"],
                "data": config["data"],
                "limits": config.get("limits", {}),
            },
        )
        parent_calibration.update(
            excluded_windows=excluded,
            unknown_label_windows=unknown,
            normal_data_assumption=unknown > 0,
        )
    if (
        settings.get("triage")
        and config["method_name"] == "mustard"
        and config["variant"] != "A0"
    ):
        triage = settings["triage"]
        parent_calibration.update(
            input_window_threshold=float(
                np.quantile(inputs, triage["window_quantile"])
            ),
            latent_window_low_threshold=float(
                np.quantile(latents, triage["latent_low_quantile"])
            ),
            latent_window_high_threshold=float(
                np.quantile(latents, triage["latent_high_quantile"])
            ),
            triage_source_split=split["identity"]["split"],
            triage_eligibility={
                "selected_windows": len(items),
                "excluded_windows": excluded,
                "unknown_label_windows": unknown,
            },
        )
    return parent_calibration


def _initialize_online(method, entity, config, identity):
    settings = config["evaluation"] | {"phase": "online"}
    reuse = settings.get("online_point_threshold_source") == "offline_parent"
    calibration = None
    if config.get("required_offline_variant") is not None:
        calibration, parent = _load_parent(method, config, identity)
        checkpoint = parent["checkpoint"]
    else:
        parent = None
        pretrained = method.pretrained_identity
        checkpoint = {
            "role": "pretrained",
            "sha256": pretrained["weights_sha256"],
            "identity": pretrained,
        }
    if reuse:
        if (
            calibration is None
            or calibration.get("score_space") != settings["score_space"]
        ):
            raise OfflineDependencyUnavailable(
                "online point calibration requires a matching offline threshold"
            )
        weights = ("ewma_previous_weight", "ewma_current_weight")
        if calibration["overlap_reduction"] != "last_ewma_estimate" or any(
            calibration[key] != settings[key] for key in weights
        ):
            raise ValueError(
                "offline parent threshold uses a different score reduction"
            )
    if not reuse or (config["method_name"] == "mustard" and config["variant"] != "A0"):
        source = settings["online_threshold_split"]
        if source not in {"train", "validation"}:
            raise ValueError("online calibration requires a pre-test split")
        calibration = _replay_calibration(
            method,
            entity["splits"][source],
            config,
            settings,
            checkpoint,
            calibration if reuse else None,
        )
    return calibration, parent


def _record_window(scored, state):
    state["scores"][scored["indices"].ravel()] = scored["scores"].ravel()


def _update_context(config, batch, raw, scored, calibration, state):
    indices = batch["indices"]
    context = {
        "calibration": calibration,
        "recorded_score": scored["scores"],
        "recorded_scores": raw["raw_scores"],
        "arrived_index": int(indices[0, -1]),
        "allow_update": bool(config["method"]["update"]) and config["variant"] != "A0",
    }
    if not context["allow_update"]:
        return context
    auxiliary = raw.get("auxiliary", {})
    if config["method_name"] == "mustard":
        decision = classify_online_window(
            float(auxiliary["input_window_score"][0]),
            float(auxiliary["latent_window_score"][0]),
            calibration,
        )
        hard_old = decision == "hard_old_normality" and config["variant"] == "A2"
        context.update(
            decision=decision,
            allow_update=decision == "gray_zone"
            or (hard_old and indices[0, 0] >= state["hard_old_end"]),
        )
    elif config["method_name"] == "candi":
        context.update(
            recorded_score=float(auxiliary["window_score"][0]),
            latent=auxiliary["latent"][0],
        )
    return context


def _named_parameters(method, name):
    if name == "candi":
        return {
            f"{group}.{key}": value
            for group in ("backbone", "adapters")
            for key, value in getattr(method, group).named_parameters()
        }
    return dict(method.network.named_parameters())


def _update_window(method, config, batch, context, state):
    protocols.validate_update_input(method, batch, context)
    if not context["allow_update"]:
        return
    name = config["method_name"]
    parameters = _named_parameters(method, name)
    prefixes = {"mustard": "projector.", "candi": "adapters."}
    allowed = [
        key
        for key in parameters
        if name == "m2n2" or key.startswith(prefixes.get(name, "\0"))
    ]
    before = {key: (value, value._version) for key, value in parameters.items()}
    result = method.update(batch, context)
    after = _named_parameters(method, name)
    changed = {
        key
        for key, (value, version) in before.items()
        if key not in after or after[key] is not value or after[key]._version != version
    }
    changed.update(after.keys() - before.keys())
    protocols.validate_update_output(
        result | {"changed_parameters": changed},
        context | {"allowed_parameters": allowed},
    )
    if result["did_update"] and context.get("decision") == "hard_old_normality":
        state["hard_old_end"] = context["arrived_index"] + 1


@contextmanager
def _defer_interrupt():
    requested = False

    def request_stop(_signal, _frame):
        nonlocal requested
        requested = True

    if current_thread() is not main_thread():
        yield lambda: requested
        return
    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        yield lambda: requested
    finally:
        signal.signal(signal.SIGINT, previous)


def _consume(method, split, config, calibration, state, commit, should_stop, publish):
    settings = config["evaluation"] | {"phase": "online"}
    scoring = {}
    state["uncertainty"] = defaultdict(list, state["uncertainty"])
    batches = iter(_windows(split, config))
    for batch in islice(batches, state["cursor"]):
        pass
    if state["cursor"]:
        positions = batch["indices"][0]
        values = state["pending"]
        scoring = (
            {
                "last_start": int(positions[0]) if values else int(positions[-1]),
                "pending": dict(zip(positions.tolist(), values)),
            }
            if len(positions)
            else {}
        )
    for batch in batches:
        batch.pop("labels", None)
        raw = _raw_score(method, batch)
        scored, scoring = score_batch(raw, calibration, settings, state=scoring)
        _record_window(scored, state)
        context = _update_context(config, batch, raw, scored, calibration, state)
        _update_window(method, config, batch, context, state)
        for key, value in raw.get("uncertainty", {}).items():
            state["uncertainty"][key].append(np.asarray(value).ravel())
        state["cursor"] += 1
        state["pending"] = list(scoring["pending"].values())
        if state["cursor"] % 8 == 0 or should_stop():
            commit()
        if publish is not None:
            publish(state["cursor"])
        if should_stop():
            raise KeyboardInterrupt
    commit()


def _end_reason(config, split, count):
    settings = config["data"] | {"limits": dict(config.get("limits", {}))}
    maximum = settings["limits"].pop("max_online_steps", None)
    if maximum != count:
        return "stream_end"
    windows = _window_schedule(split, settings, "online", config["seed"])
    return (
        "work_limit"
        if next(islice(windows, count, None), None) is not None
        else "stream_end"
    )


def _online_result(config, split, calibration, parent, state):
    settings = config["evaluation"] | {"threshold": calibration["point_threshold"]}
    limits = config.get("limits", {})
    start, stop = (
        limits.get("stream_start", 0),
        limits.get("stream_end", len(state["scores"])),
    )
    scores = state["scores"][start:stop]
    labels = split["labels"][start:stop]
    segments = [
        (max(start, left) - start, min(stop, right) - start)
        for left, right in _segments(split, config["data"])
        if left < stop and right > start
    ]
    record = {
        "scores": scores,
        "coverage": np.isfinite(scores),
        "segments": segments,
        "uncertainty": {
            key: np.concatenate(values) for key, values in state["uncertainty"].items()
        },
    }
    protocols.validate_run_input(
        config | {"kind": "evaluation", "scores": scores, "labels": labels}
    )
    result = compute_metrics(record, labels, settings)
    for segment in result["segments"]:
        segment["bounds"] = [position + start for position in segment["bounds"]]
    result.update({key: config[key] for key in protocols.IDENTITY_KEYS})
    result.update(
        status="successful",
        point_scores=scores,
        stream_range=[start, stop],
        calibration_source=calibration["source_split"],
        offline_parent=parent,
        processed_windows=state["cursor"],
        end_reason=_end_reason(config, split, state["cursor"]),
    )
    protocols.validate_run_output(result, config)
    return write_result(config, "test", result)


def _start_online(config, resume, identity):
    directory = result_directory(config, "state")
    metadata = directory / "run.json"
    if resume is None:
        if metadata.exists():
            raise ValueError("existing online run requires explicit resume")
    elif resume.get("unstarted"):
        existing = json.loads(metadata.read_text())
        if existing["status"] != "blocked" or _json_bytes(
            existing["resolved_config"]
        ) != _json_bytes(config):
            raise ValueError(
                "only a matching unstarted blocked run can retry dependency lookup"
            )
        resume = None
    else:
        committed = load_resume(config)
        if committed["cursor"] != resume["cursor"] or _json_bytes(
            committed["config"]
        ) != _json_bytes(config):
            raise ValueError("online resume does not match the committed run")
        if committed["execution_identity"] != identity:
            raise ValueError("online resume has different code or dependency versions")
        if json.loads(metadata.read_text())["status"] == "running":
            write_run_status(
                config, "interrupted", details={"reason": "previous writer stopped"}
            )
        resume = committed
    write_run_status(config, "running")
    return resume


def _stream_progress(progress, config, split):
    if progress is None:
        return None
    settings = config["data"] | {"limits": config.get("limits", {})}
    total = sum(1 for _ in _window_schedule(split, settings, "online", config["seed"]))

    def publish(completed, stats=None):
        progress(
            stage="test", unit="window", completed=completed, total=total, stats=stats
        )

    return publish


def _execute_online(config, snapshot, identity, progress):
    directory = result_directory(config, "state")
    entity = load_entity(config["data"]["prepared_root"], config["entity"])
    actual = entity["identity"] | {"features": entity["features"]}
    if any(
        config["prepared_identity"].get(key) != value for key, value in actual.items()
    ):
        raise ValueError("prepared data does not match the resolved identity")
    seed_execution(config["seed"], config["device"])
    method = create_method(config | {"scaler": entity["scaler"]})
    split = entity["splits"]["test"]
    path = directory / "calibration.json"
    if snapshot:
        method.load_state_dict(snapshot["method_state"])
        restore_rng(snapshot["rng_state"], config["device"])
        state = snapshot["scoring_state"]
        calibration, parent = snapshot["calibration"], snapshot["offline_parent"]
        if parent and (
            _hash(Path(parent["checkpoint"]["path"])) != parent["checkpoint"]["sha256"]
            or _hash(Path(parent["calibration"])) != parent["calibration_sha256"]
        ):
            raise ValueError("recorded offline parent artifacts have changed")
        if _hash(path) != snapshot["calibration_sha256"]:
            raise ValueError("online calibration hash mismatch")
    else:
        calibration, parent = _initialize_online(method, entity, config, identity)
        state = {
            "cursor": 0,
            "scores": np.full(len(split["x"]), np.nan),
            "pending": [],
            "hard_old_end": 0,
            "uncertainty": {},
        }
        _write_json(path, calibration)
    protocols.validate_calibration_output(calibration | {"kind": "calibration"}, config)
    checksum = _hash(path)

    def commit():
        save_resume(
            config,
            {
                "config": config,
                "cursor": state["cursor"],
                "method_state": method.state_dict(),
                "rng_state": capture_rng(config["device"]),
                "scoring_state": state,
                "execution_identity": identity,
                "calibration": calibration,
                "calibration_sha256": checksum,
                "offline_parent": parent,
                "role_references": {"offline_parent": parent},
            },
        )

    if not snapshot:
        commit()
    publish = _stream_progress(progress, config, split)
    if publish is not None:
        publish(state["cursor"])
    with _defer_interrupt() as should_stop:
        _consume(
            method, split, config, calibration, state, commit, should_stop, publish
        )
    result = _online_result(config, split, calibration, parent, state)
    _check_cached_results({"test": result})
    if publish is not None:
        publish(state["cursor"], evaluation_stats(json.loads(result.read_text())))
    checkpoint = save_checkpoint(
        directory / f"online.final.{uuid4().hex}.pt",
        method.state_dict(),
        identity=config["prepared_identity"],
        role="online.final",
    )
    return {
        "status": "successful",
        "run_id": config["run_id"],
        "splits": {"test": result},
        "offline_parent": parent,
        "calibration": path,
        "calibration_sha256": checksum,
        "final_checkpoint": checkpoint,
        "execution_identity": identity,
    }


def run_online(config: dict, *, resume: dict | None = None, progress=None) -> dict:
    """Score an allocated stream and continue only from its committed state."""
    protocols.validate_run_input(config)
    if config["phase"] != "online":
        raise ValueError("online execution requires an online run")
    identity = execution_identity()
    with run_lock(config):
        snapshot = _start_online(config, resume, identity)
        try:
            result = _execute_online(config, snapshot, identity, progress)
        except OfflineDependencyUnavailable as error:
            write_run_status(config, "blocked", details={"reason": str(error)})
            raise
        except KeyboardInterrupt:
            write_run_status(config, "interrupted")
            raise
        except Exception as error:
            write_run_status(config, "failed", details={"reason": str(error)})
            raise
        write_run_status(config, "successful", details=result)
        close_resume(config)
        return result

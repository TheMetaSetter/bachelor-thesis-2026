"""Shared library helpers for method training, checkpointing, and score records."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import load_checkpoint, save_checkpoint
from .config import load_yaml
from .prepared_data import iter_batches


def resolve_source_parameters(config: Mapping[str, Any], source_variant: str) -> dict:
    """Return offline source parameters for an online method from config or disk."""
    if config["phase"] != "online":
        return config["method"]["parameters"]
    source = config.get("source_method")
    if source is None:
        path = (
            Path(config["project_root"])
            / "configs/method_profiles"
            / f"{config['method']['method']}_{source_variant.lower()}_offline.yaml"
        )
        source = load_yaml(path)
    return source["parameters"]


def require_no_labels(batch: Mapping[str, Any], method_name: str) -> None:
    """Reject batches that pass ground-truth labels into scoring or updating."""
    if "labels" in batch or "point_labels" in batch:
        raise ValueError(f"{method_name} scoring does not accept labels")


def to_device_tensor(values: Any, device: torch.device) -> torch.Tensor:
    """Convert array-like inputs to float32 tensors on the target device."""
    return torch.as_tensor(values, device=device, dtype=torch.float32)


def build_score_record(
    batch: Mapping[str, Any], raw_scores: Any, *, score_kind: str
) -> dict[str, Any]:
    """Wrap raw model scores with the standard batch alignment metadata."""
    record: dict[str, Any] = {"score_kind": score_kind, "raw_scores": raw_scores}
    if score_kind == "point" and "indices" in batch:
        record["indices"] = np.asarray(batch["indices"])[:, -1]
    elif "indices" in batch:
        record["indices"] = batch["indices"]
    for key in ("segment_ids", "identity"):
        if key in batch:
            record[key] = batch[key]
    return record


def corrected_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, correction: Mapping[str, float]
) -> torch.Tensor:
    """Compute label-corrected cross-entropy with class-0 prior adjustment."""
    classes = logits.shape[-1]
    alpha, beta = correction["alpha"], correction["beta"]
    one_hot = F.one_hot(labels.long(), classes).to(logits.dtype)
    targets = one_hot * (1.0 - alpha - beta * classes) + beta
    targets[:, 0] += alpha
    return -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def window_matrix(x: Any, length: int, settings: Mapping[str, Any]) -> np.ndarray:
    """Flatten [B, L, C] windows to [B, L*C] with optional per-window normalization."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 3 or values.shape[1] != length or not len(values):
        raise ValueError("estimator requires nonempty windows of the configured length")
    if not np.isfinite(values).all():
        raise ValueError("estimator requires finite feature values")
    rows = values.reshape(len(values), -1)
    if not settings["normalize_windows"]:
        return rows
    spread = (
        rows.std(axis=1, keepdims=True, ddof=1)
        if rows.shape[1] > 1
        else np.zeros((len(rows), 1))
    )
    scale = np.where(spread >= settings["normalization_epsilon"], spread, 1)
    return (rows - rows.mean(axis=1, keepdims=True)) / scale


def build_adam(
    parameters: Any, optimizer_config: Mapping[str, Any], *, adamw: bool = False
) -> torch.optim.Optimizer:
    """Construct Adam or AdamW from a stage optimizer config."""
    cls = torch.optim.AdamW if adamw else torch.optim.Adam
    return cls(
        parameters,
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
    )


def _save_stage_checkpoint(
    method: Any, directory: Path, role: str, identity: dict[str, Any]
) -> Path:
    return save_checkpoint(
        directory / f"{role}.{uuid4().hex}.pt",
        method.state_dict(),
        identity=identity,
        role=role,
    )


def _select_epoch_checkpoint(
    method: Any,
    state: dict[str, Any],
    record: dict[str, Any],
    directory: Path,
    identity: dict[str, Any],
    monitor_epoch: Callable | None,
    save_best_per_epoch: bool,
) -> None:
    loss = record["validation"]["total_loss"]
    if monitor_epoch is None:
        value, best = loss, state["best_loss"]
    else:
        monitor = monitor_epoch(method, stage=state["stage"])
        record["monitor"] = monitor
        value = monitor["held_out_fpr"]
        best = state.get("best_held_out_fpr", float("inf"))
    if value is not None and np.isfinite(value) and value < best:
        state["best"] = (
            _save_stage_checkpoint(
                method, directory, f"{state['stage']}.best", identity
            )
            if save_best_per_epoch
            else method.state_dict()
        )
        state["best_loss"] = loss
        if monitor_epoch is not None:
            state["best_held_out_fpr"] = value


def _start_neural_stage(
    method: Any,
    stage_name: str,
    directory: Path,
    identity: dict[str, Any],
    stages: list[dict[str, Any]],
    resume: dict[str, Any] | None,
    on_stage_start: Callable[[str], None] | None,
) -> dict[str, Any]:
    if resume and resume.get("stage") == stage_name and "next_epoch" in resume:
        method.load_state_dict(resume["method_state"])
        return resume | {"history": list(resume["history"])}
    if on_stage_start is not None:
        on_stage_start(stage_name)
    return {
        "stage": stage_name,
        "next_epoch": 0,
        "initial": _save_stage_checkpoint(
            method, directory, f"{stage_name}.initial", identity
        ),
        "best": None,
        "best_loss": float("inf"),
        "stages": list(stages),
        "history": [],
    }


def _finalize_neural_stage(
    method: Any,
    state: dict[str, Any],
    directory: Path,
    identity: dict[str, Any],
    monitor_epoch: Callable | None,
    save_best_per_epoch: bool,
    post_fit: Callable[[], None] | None,
) -> dict[str, Any]:
    best = state["best"]
    stage_name = state["stage"]
    if best is None:
        detail = (
            "defined held-out FPR" if monitor_epoch else "finite best validation loss"
        )
        raise ValueError(f"{identity['method']} {stage_name}: no {detail}")
    if save_best_per_epoch:
        method.load_state_dict(
            load_checkpoint(
                Path(best), expected_identity=identity | {"role": f"{stage_name}.best"}
            )
        )
        best_path = best
    else:
        method.load_state_dict(best)
        if post_fit is not None:
            post_fit()
        best_path = _save_stage_checkpoint(
            method, directory, f"{stage_name}.best", identity
        )
    result = {
        "name": stage_name,
        "initial_checkpoint": state["initial"],
        "best_checkpoint": best_path,
        "best_validation_loss": state["best_loss"],
        "epochs": state["history"],
    }
    if monitor_epoch is not None:
        result["best_held_out_fpr"] = state["best_held_out_fpr"]
    return result


def fit_neural_stages(
    method: Any,
    definitions: list[dict[str, Any]],
    *,
    checkpoint_dir: Path,
    identity: dict[str, Any],
    run_epoch: Callable[[dict[str, Any], int], dict[str, Any]],
    resume: dict[str, Any] | None = None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
    on_stage_start: Callable[[str], None] | None = None,
    monitor_epoch: Callable | None = None,
    post_fit: Callable[[], None] | None = None,
    save_best_per_epoch: bool = False,
) -> dict[str, Any]:
    """Run multi-stage neural training with checkpointing and resume support."""
    stages = list(resume["stages"]) if resume and "stages" in resume else []
    max_epochs = method.config.get("limits", {}).get("max_epochs_per_stage")
    for definition in definitions[len(stages) :]:
        stage_name = definition["name"]
        state = _start_neural_stage(
            method, stage_name, checkpoint_dir, identity, stages, resume, on_stage_start
        )
        limit = (
            definition["epochs"]
            if max_epochs is None
            else min(definition["epochs"], max_epochs)
        )
        for epoch in range(state["next_epoch"], limit):
            record = run_epoch(definition, epoch)
            _select_epoch_checkpoint(
                method,
                state,
                record,
                checkpoint_dir,
                identity,
                monitor_epoch,
                save_best_per_epoch,
            )
            state["next_epoch"] = epoch + 1
            state["history"].append(record)
            if on_checkpoint is not None:
                on_checkpoint(
                    state
                    | {
                        "history": list(state["history"]),
                        "method_state": method.state_dict(),
                    }
                )
        stage_result = _finalize_neural_stage(
            method,
            state,
            checkpoint_dir,
            identity,
            monitor_epoch,
            save_best_per_epoch,
            post_fit,
        )
        stages.append(stage_result)
        if on_checkpoint is not None and len(definitions) > 1:
            on_checkpoint(
                {
                    "stage": stage_name,
                    "stages": list(stages),
                    "method_state": method.state_dict(),
                }
            )
    return {"selected_checkpoint": stages[-1]["best_checkpoint"], "stages": stages}


def fit_endpoint_model(
    method: Any,
    train: dict[str, Any],
    validation: dict[str, Any],
    *,
    method_name: str,
    checkpoint_dir: Path,
    resume: dict[str, Any] | None,
    on_checkpoint: Callable[[dict[str, Any]], None] | None,
    fit_fn: Callable[[dict[str, Any]], None],
    calibrate_fn: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Fit and calibrate an endpoint estimator with initial/best checkpointing."""
    config = method.config
    if config["phase"] != "offline":
        raise ValueError(f"{method_name} fitting requires an offline run")
    identity = {"method": method_name, "prepared_identity": config["prepared_identity"]}
    initial = (
        resume["initial"]
        if resume
        else _save_stage_checkpoint(method, checkpoint_dir, "fit.initial", identity)
    )
    if on_checkpoint is not None and not resume:
        on_checkpoint(
            {
                "initial": initial,
                "method_state": method.state_dict(),
                "fit_complete": False,
            }
        )
    if resume and resume["fit_complete"]:
        method.load_state_dict(resume["method_state"])
        selected = resume["selected_checkpoint"]
    else:
        fit_fn(train)
        calibrate_fn(validation)
        selected = _save_stage_checkpoint(method, checkpoint_dir, "fit.best", identity)
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "initial": initial,
                    "method_state": method.state_dict(),
                    "fit_complete": True,
                    "selected_checkpoint": selected,
                }
            )
    return {
        "selected_checkpoint": selected,
        "stages": [
            {"name": "fit", "initial_checkpoint": initial, "best_checkpoint": selected}
        ],
    }


def eligible_validation_batches(validation: dict[str, Any], config: Mapping[str, Any]):
    """Yield validation windows that contain no known anomaly label."""
    data = config["data"] | {"limits": config.get("limits", {})}
    for batch in iter_batches(validation, data, phase="offline", seed=config["seed"]):
        labels = np.asarray(
            batch.get("labels", np.full(batch["indices"].shape, np.nan))
        )
        eligible = ~(labels == 1).any(axis=1)
        if eligible.any():
            yield batch["x"][eligible]

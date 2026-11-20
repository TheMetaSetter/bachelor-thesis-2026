"""Capture and restore MUSTARD state at execution boundaries."""

from collections.abc import Mapping
from copy import deepcopy


def pack_state(method) -> dict:
    """Return a snapshot with independent tensor storage."""
    return deepcopy({
        "schema_version": 1, "variant": method.variant, "stage": method.stage,
        "source_variant": method.source_variant,
        "source_parameters": method.source_parameters,
        "network": method.network.state_dict(), "memory": method.memory,
        "rng": method.rng.get_state(), "class_rotation_offset": method.class_rotation_offset,
        "optimizer": None if method.optimizer is None else method.optimizer.state_dict(),
        "verification": None if method.verification_buffer is None else vars(method.verification_buffer),
        "shape": method.shape,
    })


def restore_state(method, state: Mapping) -> None:
    """Restore the same variant or initialize online from the required offline variant."""
    if state["schema_version"] != 1 or tuple(state["shape"]) != method.shape:
        raise ValueError("MUSTARD state shape or version mismatch")
    parent = method.config["phase"] == "online" and state["variant"] == method.source_variant
    if not parent and (state["variant"] != method.variant or state["source_variant"] != method.source_variant):
        raise ValueError("MUSTARD state variant mismatch")
    if parent and (state["stage"] != "stage_b" or not state["memory"]):
        raise ValueError("MUSTARD online requires a complete Stage B checkpoint")
    method.network.load_state_dict(state["network"], strict=not parent)
    method.memory = {key: value.to(method.device).clone() for key, value in state["memory"].items()}
    method.source_parameters = deepcopy(state["source_parameters"])
    if parent:
        return
    method._set_stage(state["stage"])
    method.rng.set_state(state["rng"].cpu())
    method.class_rotation_offset = state["class_rotation_offset"]
    if state["optimizer"] is not None:
        method.optimizer.load_state_dict(state["optimizer"])
    if method.verification_buffer is not None and state["verification"] is not None:
        saved = deepcopy(state["verification"])
        for entry in saved["_items"]:
            entry["x"] = entry["x"].to(method.device)
        vars(method.verification_buffer).update(saved)

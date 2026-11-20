"""Read prepared observations and yield windows in split order."""

from collections.abc import Iterator
from hashlib import file_digest, sha256
from itertools import islice
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from . import protocols


def _manifest_profile(manifest):
    identity = manifest["identity"]
    return identity.get("profile", identity.get("settings", {}))


def _candidate(path, dataset):
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
        entities = manifest["entities"]
        if (manifest["status"] == "complete" and manifest["version_id"] == path.name
                and _manifest_profile(manifest)["dataset"] == dataset
                and isinstance(entities, dict) and entities):
            return path, tuple(entities), manifest_path.stat().st_mtime_ns
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def discover_versions(project: Path, dataset: str, configured: Path | None) -> list[tuple[Path, tuple[str, ...]]]:
    """List valid manifest choices with the default first."""
    manifests = (project / "data/preprocessed" / dataset).glob("*/manifest.json")
    roots = {manifest.parent.resolve() for manifest in manifests}
    if configured is not None:
        configured = (project / configured).resolve()
        roots.add(configured)
    choices = [candidate for path in roots if (candidate := _candidate(path, dataset)) is not None]
    choices.sort(key=lambda choice: (choice[0] != configured, -choice[2], choice[0].name, choice[0]))
    return [(path, entities) for path, entities, _ in choices]


def _checked_path(root: Path, name: str, files: dict) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"{path}: prepared file escapes its version directory")
    with path.open("rb") as file:
        actual = file_digest(file, "sha256").hexdigest()
    if actual != files.get(name):
        raise ValueError(f"{path}: prepared file hash mismatch")
    return path


def _read_split(path, features, count, identity):
    table = pq.read_table(path)
    required = {"source_file", "source_row", "label", *features}
    if set(table.column_names) not in (required, required | {"timestamp"}):
        raise ValueError(f"{path}: invalid prepared columns")
    if count < 1 or table.num_rows != count:
        raise ValueError(f"{path}: prepared row count mismatch")
    x = np.column_stack([table[name].to_numpy() for name in features])
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: features must be finite numbers")
    labels = np.array(table["label"].to_pylist(), dtype=float)
    if not np.all(np.isnan(labels) | np.isin(labels, [0, 1])):
        raise ValueError(f"{path}: labels must be 0, 1, or unknown")
    source_row = table["source_row"].to_numpy()
    source_file = table["source_file"].to_numpy()
    if not np.issubdtype(source_row.dtype, np.integer) or np.any(source_row < 0):
        raise ValueError(f"{path}: invalid source rows")
    if not all(isinstance(value, str) and value for value in source_file):
        raise ValueError(f"{path}: invalid source files")
    split = dict(x=x, labels=labels, source_file=source_file, source_row=source_row,
                 segments=[(0, count)], identity=identity)
    if "timestamp" in table.column_names:
        timestamps = table["timestamp"].to_numpy().astype("datetime64[ns]")
        if np.any(np.isnat(timestamps)) or np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError(f"{path}: timestamps must be present and sorted")
        split["timestamps"] = timestamps
    return split


def load_entity(prepared_root: Path, entity: str) -> dict[str, Any]:
    """Verify and load one entity without reopening its original source files."""
    root = prepared_root.resolve()
    manifest_path = root / "manifest.json"
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    if manifest.get("status") != "complete" or manifest.get("version_id") != root.name:
        raise ValueError(f"{manifest_path}: incomplete or mismatched manifest")
    record = manifest.get("entities", {}).get(entity)
    if record is None:
        raise ValueError(f"{manifest_path}: unknown entity {entity!r}")
    features = record["features"]
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError(f"{manifest_path}: expected distinct feature names")
    files = manifest["files"]
    scaler_name = f"{entity}/scaler.json"
    scaler_path = _checked_path(root, scaler_name, files)
    scaler = json.loads(scaler_path.read_text())
    if scaler.get("features") != features:
        raise ValueError(f"{scaler_path}: scaler feature order mismatch")
    profile = _manifest_profile(manifest)
    identity = dict(dataset=profile.get("dataset"), entity=entity, version_id=root.name,
                    manifest_path=manifest_path, manifest_sha256=sha256(contents).hexdigest(),
                    scaler_path=scaler_path, scaler_sha256=files[scaler_name])
    splits = {}
    for name in ("train", "validation", "test"):
        path = _checked_path(root, f"{entity}/{name}.parquet", files)
        count = record["splits"][name]["rows"]
        splits[name] = _read_split(path, features, count, dict(identity, split=name))
    prepared = dict(splits=splits, features=list(features), scaler=scaler, identity=identity)
    protocols.validate_data_input(prepared, identity)
    return prepared


def _segments(split, config):
    count = len(split["x"])
    interval = config["expected_interval_seconds"]
    if "timestamps" not in split or config["gap_policy"] != "break_at_time_gaps":
        return [(0, count)]
    timestamps = split["timestamps"]
    jumps = np.diff(timestamps) / np.timedelta64(1, "s")
    boundaries = [0, *(np.flatnonzero(jumps > interval) + 1).tolist(), count]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _window_starts(segments, start, stop, width, stride, tail):
    for segment_id, (left, right) in enumerate(segments):
        left, right = max(start, left), min(stop, right)
        final = right - width
        if final < left:
            continue
        for offset in range(left, final + 1, stride):
            yield segment_id, offset
        if tail == "end_align" and (final - left) % stride:
            yield segment_id, final


def _window_schedule(split, config, phase, seed):
    name = split["identity"]["split"]
    kind = "online" if phase == "online" else name
    limits = config.get("limits", {})
    start = limits.get("stream_start", 0) if phase == "online" and name == "test" else 0
    stop = limits.get("stream_end", len(split["x"])) if phase == "online" and name == "test" else len(split["x"])
    if not 0 <= start < stop <= len(split["x"]):
        raise ValueError(f"{name}: invalid stream range")
    width = config["window_size"]
    iterator = _window_starts(
        _segments(split, config), start, stop, width, config[f"{kind}_stride"],
        config["tail_policy"][kind],
    )
    if kind == "train" and config["shuffle_train"]:
        starts = list(iterator)
        rng = np.random.default_rng(seed)
        iterator = (starts[index] for index in rng.permutation(len(starts)))
    key = "max_online_steps" if phase == "online" and name == "test" else f"max_{name}_windows"
    return islice(iterator, limits.get(key))


def iter_batches(split: dict, data_config: dict, *, phase: str, seed: int) -> Iterator[dict]:
    """Yield one batch at a time and keep original split positions."""
    if phase not in {"offline", "online"}:
        raise ValueError(f"unknown phase: {phase}")
    starts = _window_schedule(split, data_config, phase, seed)
    size = 1 if phase == "online" else data_config["batch_size"]
    positions = np.arange(data_config["window_size"])
    entries = list(islice(starts, size))
    if not entries:
        raise ValueError(f"{split['identity']['split']}: no usable windows")
    while entries:
        segment_ids, offsets = np.asarray(entries).T
        indices = offsets[:, None] + positions
        batch = dict(x=split["x"][indices], indices=indices, segment_ids=segment_ids,
                     identity=split["identity"])
        for field in ("labels", "timestamps"):
            if field in split:
                batch[field] = split[field][indices]
        protocols.validate_data_output(batch, {**split["identity"], "data": data_config, "phase": phase})
        yield batch
        entries = list(islice(starts, size))

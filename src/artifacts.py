"""Store run identities, compact results, and checked state."""

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import file_digest
import json
from pathlib import Path
import os
import re
import secrets
from tempfile import NamedTemporaryFile
from time import time_ns
from typing import Any
from uuid import UUID


_DIMENSIONS = ("dataset", "entity", "method_name", "variant", "seed")
_STATUSES = {
    None: {"running", "blocked"},
    "running": {"successful", "failed", "interrupted", "blocked"},
    "blocked": {"running"},
    "failed": {"running"},
    "interrupted": {"running"},
    "successful": set(),
}


def _timestamp():
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _uuid7():
    timestamp = time_ns() // 1_000_000
    high, low = secrets.randbits(12), secrets.randbits(62)
    return str(UUID(int=timestamp << 80 | 7 << 76 | high << 64 | 2 << 62 | low))


def _component(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError(f"invalid path component {value!r}")
    return value


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot store {type(value).__name__} in run metadata")


def _json_bytes(value):
    return (
        json.dumps(
            value, default=_json_default, allow_nan=False, indent=2, sort_keys=True
        ).encode()
        + b"\n"
    )


def _write_json(path, value):
    return _write_atomic(path, lambda file: file.write(_json_bytes(value)))


def _read_json(path):
    return json.loads(path.read_bytes())


def _hash(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _output_root(config, project, run_id):
    if "experiment_id" in config:
        mode = config.get("execution_mode", "normal")
        folder = {"normal": "experiments", "smoke": "smoke"}[mode]
        return project / "outputs" / folder / _component(config["experiment_id"])
    if config.get("output_dir"):
        return (project / Path(config["output_dir"])).resolve()
    return project / "outputs/standalones" / run_id


def allocate_run(config: dict, *, project_root: Path) -> dict:
    """Reserve a unique run ID before creating run state."""
    project = project_root.resolve()
    requested = config.get("run_id")
    if requested is not None and (
        "experiment_id" in config
        or not isinstance(requested, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", requested)
    ):
        raise ValueError("run_id must be a safe standalone ID")
    if requested is not None and find_runs(
        project / "outputs", filters={"run_id": requested}
    ):
        raise FileExistsError(f"run_id already used: {requested}")
    registry = project / "outputs/run_ids"
    registry.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run_id = requested if requested is not None else _uuid7()
        root = _output_root(config, project, run_id)
        run = deepcopy(config)
        run.update(run_id=run_id, started_at=_timestamp(), output_root=root)
        state_path = result_directory(run, "state") / "run.json"
        reservation = registry / f"{run_id}.json"
        try:
            with reservation.open("xb") as file:
                file.write(_json_bytes({"run_id": run_id, "run_path": state_path}))
                file.flush()
                os.fsync(file.fileno())
        except FileExistsError:
            if requested is not None:
                raise FileExistsError(f"run_id already used: {run_id}") from None
            continue
        _sync_directory(registry)
        if "experiment_id" not in run:
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise FileExistsError(
                    f"standalone output directory is already in use: {root}"
                )
            root.mkdir(parents=True, exist_ok=True)
            state_path.parent.mkdir(exist_ok=False)
        return run
    raise FileExistsError("could not reserve a unique run_id")


def result_directory(config: dict, split: str) -> Path:
    """Return a split or state directory from the recorded grouping."""
    if split not in {"train", "validation", "test", "state"}:
        raise ValueError(f"invalid result split: {split}")
    root = Path(config["output_root"])
    if "experiment_id" not in config:
        return root / split
    group = config.get("grouping", {})
    dimensions = [
        _component(
            config.get("reporting_variant", config[name])
            if name == "variant"
            else config[name]
        )
        for name in _DIMENSIONS
        if group.get(name, False)
    ]
    return root.joinpath(
        _component(config["phase"]), *dimensions, split, _component(config["run_id"])
    )


def _run_metadata(config):
    keys = (
        *_DIMENSIONS,
        "phase",
        "experiment_id",
        "run_id",
        "started_at",
        "profile_paths",
        "prepared_identity",
        "execution_mode",
        "grouping",
        "offline_parent",
        "reporting_variant",
    )
    return {key: config[key] for key in keys if key in config}


def write_run_status(config: dict, status: str, *, details: dict | None = None) -> Path:
    """Write a status transition atomically."""
    path = result_directory(config, "state") / "run.json"
    previous = _read_json(path) if path.exists() else {}
    prior_status = previous.get("status")
    if status not in _STATUSES.get(prior_status, ()):
        raise ValueError(f"invalid run transition: {prior_status} to {status}")
    metadata = _run_metadata(config)
    at = _timestamp()
    record = dict(previous, **metadata)
    record.update(
        schema_version=1, status=status, state_dir=path.parent, resolved_config=config
    )
    record["history"] = [*previous.get("history", ()), {"status": status, "at": at}]
    record["resume_count"] = previous.get("resume_count", 0) + int(
        status == "running" and bool(previous)
    )
    if status == "running":
        record.pop("completed_at", None)
    else:
        record["completed_at"] = at
    if details:
        record["details"] = details
    return _write_json(path, record)


def write_result(config: dict, split: str, result: dict) -> Path:
    """Write compact split results separately from whole-run status."""
    if split == "state":
        raise ValueError("state is not a result split")
    path = result_directory(config, split)
    record = dict(result, **_run_metadata(config), split=split, schema_version=1)
    scores = record.pop("point_scores", None)
    if scores is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {"score": pa.array(scores, type=pa.float64(), from_pandas=True)}
        )
        score_path = _write_atomic(
            path / "point_scores.parquet", lambda file: pq.write_table(table, file)
        )
        record["point_scores_path"] = score_path.name
        record["point_scores_sha256"] = _hash(score_path)
    return _write_json(path / "result.json", record)


def _attempt_order(record):
    def time(key):
        value = record.get(key)
        return (
            datetime.fromisoformat(value) if value else datetime.min.replace(tzinfo=UTC)
        )

    return time("started_at"), time("completed_at"), record.get("run_id", "")


def find_runs(
    root: Path, *, filters: Mapping[str, Any], include_results: bool = True
) -> list[dict]:
    """Find run records and, if requested, split results by exact metadata."""
    records = []
    filenames = ("run.json", "result.json") if include_results else ("run.json",)
    for filename in filenames:
        for path in sorted(root.rglob(filename)):
            row = _read_json(path)
            if row.get("schema_version") == 1 and all(
                row.get(key) == value for key, value in filters.items()
            ):
                records.append(dict(row, record_path=path))
    return sorted(records, key=_attempt_order, reverse=True)


def _compatible_parent(parent, config):
    if parent.get("status") != "successful":
        return False
    if (
        "execution_identity" in config
        and parent.get("details", {}).get("execution_identity")
        != config["execution_identity"]
    ):
        return False
    actual = parent.get("resolved_config", {})
    for key in ("prepared_identity", "encoder", "scientific_identity"):
        if key in config and _json_bytes(config[key]) != _json_bytes(actual.get(key)):
            return False
    if "data" in config and _json_bytes(config["data"]) != _json_bytes(
        actual.get("data")
    ):
        return False
    expected_method = config.get("source_method", config.get("method"))
    if expected_method is not None and _json_bytes(expected_method) != _json_bytes(
        actual.get("method")
    ):
        return False
    parent_evaluation = actual.get("evaluation") or {}
    evaluation = config.get("evaluation") or {}
    keys = (
        "score_space",
        "point_score_transform",
        "offline_overlap_reduction",
        "ewma_current_weight",
        "ewma_previous_weight",
    )
    if evaluation.get("online_point_threshold_source") == "offline_parent":
        keys += ("offline_threshold_quantile", "offline_threshold_split")
    return all(evaluation.get(key) == parent_evaluation.get(key) for key in keys)


def select_offline_parent(config: dict, *, project_root: Path) -> dict:
    """Select a compatible successful parent before checking its files."""
    filters = {key: config[key] for key in ("dataset", "entity", "seed", "method_name")}
    filters.update(variant=config["required_offline_variant"], phase="offline")
    if "experiment_id" in config:
        root = _output_root(config, project_root.resolve(), "")
        filters["experiment_id"] = config["experiment_id"]
        candidates = find_runs(root, filters=filters)
    else:
        root = (project_root / Path(config["offline_parent_dir"])).resolve()
        candidates = find_runs(root, filters=filters)
    selected = next(
        (
            item
            for item in candidates
            if "state_dir" in item and _compatible_parent(item, config)
        ),
        None,
    )
    if selected is None:
        raise ValueError("no compatible successful offline parent")
    details = selected.get("details", {})
    for name in ("selected_checkpoint", "calibration"):
        reference = details.get(name)
        if reference is None or not Path(reference).is_file():
            raise FileNotFoundError(
                f"offline parent {selected['run_id']}: missing {reference or name}"
            )
    return selected


_STATE_TAG = "__thesis_state_type__"
_RESUME_FIELDS = {"config", "cursor", "method_state", "rng_state", "scoring_state"}


def _encode_state(value):
    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields:
            raise ValueError("unsupported array dtype in checkpoint")
        bits = torch.from_numpy(np.frombuffer(value.tobytes(), dtype=np.uint8).copy())
        return {
            _STATE_TAG: "ndarray",
            "bytes": bits,
            "dtype": value.dtype.str,
            "shape": value.shape,
        }
    if isinstance(value, np.generic):
        return _encode_state(value.item())
    if isinstance(value, Path):
        return {_STATE_TAG: "path", "value": str(value)}
    if isinstance(value, Mapping):
        if _STATE_TAG in value:
            raise ValueError("reserved checkpoint key")
        return {key: _encode_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_encode_state(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float, torch.Tensor)):
        return value
    raise ValueError(f"unsupported checkpoint state type: {type(value).__name__}")


def _decode_state(value):
    import numpy as np

    if isinstance(value, dict):
        kind = value.get(_STATE_TAG)
        if kind == "ndarray":
            return (
                np.frombuffer(value["bytes"].numpy().tobytes(), dtype=value["dtype"])
                .copy()
                .reshape(value["shape"])
            )
        if kind == "path":
            return Path(value["value"])
        return {key: _decode_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_decode_state(item) for item in value)
    return value


def _write_payload(path, state):
    import torch

    body = _encode_state(state)
    return _write_atomic(path, lambda file: torch.save(body, file))


def _load_payload(path, expected_hash):
    import torch

    with path.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != expected_hash:
            raise ValueError(f"{path}: payload hash mismatch")
        file.seek(0)
        payload = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: checkpoint must contain a state mapping")
    return _decode_state(payload)


def _descriptor_path(path):
    return path.with_name(path.name + ".json")


def save_checkpoint(
    path: Path, state: Mapping[str, Any], *, identity: dict, role: str
) -> Path:
    """Write one immutable checkpoint and publish its descriptor last."""
    if path.exists() or _descriptor_path(path).exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    _write_payload(path, state)
    _write_json(
        _descriptor_path(path),
        {
            "schema_version": 1,
            "codec": "torch_weights_only",
            "sha256": _hash(path),
            "role": role,
            "identity": identity,
        },
    )
    return path


def _check_descriptor(descriptor, path):
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("codec") != "torch_weights_only"
    ):
        raise ValueError(f"{path}: unsupported checkpoint descriptor")
    if not re.fullmatch(r"[a-f0-9]{64}", str(descriptor.get("sha256", ""))):
        raise ValueError(f"{path}: missing or invalid payload hash")


def load_checkpoint(path: Path, *, expected_identity: dict) -> dict:
    """Verify the descriptor and hash before reading checkpoint state."""
    descriptor = _read_json(_descriptor_path(path))
    _check_descriptor(descriptor, path)
    expected = dict(expected_identity)
    if "role" in expected and descriptor["role"] != expected.pop("role"):
        raise ValueError(f"{path}: checkpoint role mismatch")
    if _json_bytes(descriptor["identity"]) != _json_bytes(expected):
        raise ValueError(f"{path}: checkpoint identity mismatch")
    return _load_payload(path, descriptor["sha256"])


def _validate_resume(state, config):
    missing = _RESUME_FIELDS - state.keys()
    if missing:
        raise ValueError(f"resume: missing fields {sorted(missing)}")
    if type(state["cursor"]) is not int or state["cursor"] < 0:
        raise ValueError("resume: cursor must be a nonnegative integer")
    if state["config"].get("run_id") != config["run_id"]:
        raise ValueError("resume: run ID mismatch")


def save_resume(config: dict, state: dict) -> Path:
    """Update the inactive payload before committing its pointer."""
    _validate_resume(state, config)
    directory = result_directory(config, "state")
    pointer = directory / "resume.json"
    old = _read_json(pointer) if pointer.is_file() else None
    if old is not None:
        _check_descriptor(old, pointer)
        if old.get("slot") not in (0, 1) or old.get("run_id") != config["run_id"]:
            raise ValueError(f"{pointer}: invalid committed resume identity")
    slot = 1 - old["slot"] if old else 0
    generation = old["generation"] + 1 if old else 1
    payload = directory / f"resume-{slot}.pt"
    _write_payload(payload, state)
    return _write_json(
        pointer,
        {
            "schema_version": 1,
            "codec": "torch_weights_only",
            "sha256": _hash(payload),
            "run_id": config["run_id"],
            "slot": slot,
            "generation": generation,
            "cursor": state["cursor"],
            "role_references": state.get("role_references", {}),
            "writer_offsets": state.get("writer_offsets", {}),
        },
    )


def load_resume(run: dict) -> dict:
    """Read only the generation named by the committed pointer."""
    directory = (
        Path(run["state_dir"]) if "state_dir" in run else result_directory(run, "state")
    )
    pointer = directory / "resume.json"
    descriptor = _read_json(pointer)
    _check_descriptor(descriptor, pointer)
    if (
        descriptor.get("slot") not in (0, 1)
        or descriptor.get("run_id") != run["run_id"]
    ):
        raise ValueError(f"{pointer}: invalid committed resume identity")
    payload = directory / f"resume-{descriptor['slot']}.pt"
    state = _load_payload(payload, descriptor["sha256"])
    _validate_resume(state, run)
    if state["cursor"] != descriptor["cursor"]:
        raise ValueError(f"{pointer}: cursor mismatch")
    return state


def execution_identity() -> dict:
    """Identify the source and installed dependencies used by a run."""
    from hashlib import sha256
    from importlib.metadata import PackageNotFoundError, version

    root = Path(__file__).resolve().parent
    files = sorted(
        (str(path.relative_to(root)), _hash(path)) for path in root.rglob("*.py")
    )
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "torch",
        "pyarrow",
        "PyYAML",
        "scipy",
        "scikit-learn",
        "skops",
        "stumpy",
        "momentfm",
        "transformers",
        "vus",
        "affiliation",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "code_sha256": sha256(_json_bytes(files)).hexdigest(),
        "dependencies": versions,
    }


def capture_rng(device) -> dict:
    import random

    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    device = torch.device(device)
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device) -> None:
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device"])


def seed_execution(seed, device) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.device(device).type == "mps":
        torch.mps.manual_seed(seed)


@contextmanager
def run_lock(config):
    """Reject another active writer for this allocated run."""
    import fcntl

    directory = result_directory(config, "state")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"run already has an active writer: {config['run_id']}"
            ) from None
        yield


def close_resume(config):
    """Remove the working state after successful publication."""
    directory = result_directory(config, "state")
    (directory / "resume.json").unlink(missing_ok=True)
    for slot in (0, 1):
        (directory / f"resume-{slot}.pt").unlink(missing_ok=True)
    _sync_directory(directory)

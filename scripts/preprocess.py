#!/usr/bin/env python3
"""Prepare five local datasets with one shared flow.

Requires Python 3.10+, NumPy, pandas, PyArrow, and PyYAML.
Run: python scripts/preprocess.py --dataset smd --entities machine-1-1
Optional: --profile settings.yaml --input-root PATH --output-root PATH

YAML overrides named profile settings; --show-profile prints them without reading data.
Relative input overrides use the current directory. Built-in input paths
and the default output root use the directory above this script.

Flow: read -> split -> check missing values -> scale -> Parquet splits and scaler.json.
Complete versions are checked before reuse; damaged versions stop without overwrite.
Save source file names and row positions. Unknown labels remain null.
Validation takes the last floor(validation_ratio * n) original training rows.
New splits use floor(train_ratio * n) and floor(test_start_ratio * n);
training ends before its first labeled anomaly. Empty or overlapping splits stop.
SWaT uses trimmed early normal training and later mixed-label testing; validate its three blocks.
ICCAD is one Console entity with 5XX counts. Missing counts mean no recorded
events by profile assumption, not proof of complete collection. Its five-minute
intervals overlap inclusive UTC annotations. NASA endpoints are also inclusive.
SMD training labels remain unknown. NASA P-2 and T-10 are excluded.
IOPS columns are value and binary label. Publisher provenance still needs review.
Fit scaling on final training only: linear quartiles for robust scaling,
population standard deviation for standard scaling, and divisor 1 for zero spread.
Do not create windows, fill timestamps, interpolate, or use future data for filling.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "data/preprocessed"
DATASETS = ("iccad", "smd", "iops", "swat", "nasa")
EXCLUDED = ("MV101", "AIT201", "MV201", "P201", "P202", "P204", "MV303")
META = ("source_file", "source_row", "label", "timestamp")


@dataclass
class EntityData:
    """One entity, with source splits and a fixed feature order."""

    entity: str
    frames: dict[str, pd.DataFrame]
    features: list[str]


def default_profile(dataset):
    sources = {
        "iccad": {"data": "pivoted_data_all.parquet", "labels": "anomaly_windows.csv"},
        "smd": {
            "train": "train/*.txt",
            "test": "test/*.txt",
            "labels": "test_label/*.txt",
        },
        "iops": {"train": "*.train.out", "test": "*.test.out"},
        "swat": {"train": "normal.csv", "test": "attack.csv"},
        "nasa": {
            "train": "train/*.npy",
            "test": "test/*.npy",
            "labels": "labeled_anomalies.csv",
        },
    }
    folders = {
        "iccad": "ibm-cloud-console-anomaly-dataset-iccad",
        "smd": "ServerMachineDataset",
        "iops": "IOPS",
        "swat": "SWaT",
        "nasa": "NASA",
    }
    return {
        "dataset": dataset,
        "input_root": str(ROOT / "data" / folders[dataset]),
        "sources": sources[dataset],
        "entities": None,
        "sort_timestamps": True,
        "duplicates": "keep",
        "without_timestamps": "source_order",
        "validation_ratio": 0.2,
        "train_ratio": 0.6,
        "test_start_ratio": 0.8,
        "train_end": None,
        "test_start": None,
        "scaler": "robust",
        "missing_values": "zero_counts" if dataset == "iccad" else "error",
        "features": "5xx_counts" if dataset == "iccad" else "all",
        "exclude_features": list(EXCLUDED) if dataset == "swat" else [],
        "labels": {
            "iccad": "interval_overlap",
            "smd": "test_binary",
            "iops": "second_column",
            "swat": "normal_attack",
            "nasa": "inclusive_intervals",
        }[dataset],
    }


def load_profile(dataset, profile_path=None):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    profile = default_profile(dataset)
    if profile_path is not None:
        with Path(profile_path).open() as stream:
            overrides = yaml.safe_load(stream)
        if not isinstance(overrides, dict) or set(overrides) - set(profile):
            raise ValueError("Profile must be a mapping with supported fields only")
        profile.update(overrides)
    fixed = (
        "dataset",
        "duplicates",
        "without_timestamps",
        "features",
        "exclude_features",
        "missing_values",
        "labels",
    )
    if any(profile[key] != default_profile(dataset)[key] for key in fixed):
        raise ValueError("Profile changes an unapproved dataset rule")
    if not isinstance(profile["sources"], dict) or set(profile["sources"]) != set(
        default_profile(dataset)["sources"]
    ):
        raise ValueError("Profile sources must keep the dataset's source roles")
    if any(not isinstance(v, str) or not v for v in profile["sources"].values()):
        raise ValueError("Source patterns must be nonempty strings")
    if type(profile["sort_timestamps"]) is not bool or profile["scaler"] not in (
        "robust",
        "standard",
    ):
        raise ValueError("Invalid timestamp sorting or scaler setting")
    for key in ("validation_ratio", "train_ratio", "test_start_ratio"):
        value = profile[key]
        if type(value) not in (int, float) or not 0 < value < 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if profile["train_ratio"] >= profile["test_start_ratio"]:
        raise ValueError("Training must end before testing starts")
    for key in ("train_end", "test_start"):
        if profile[key] is not None and (
            type(profile[key]) is not int or profile[key] < 1
        ):
            raise ValueError(f"{key} must be a positive row boundary or null")
    if (profile["train_end"] is None) != (profile["test_start"] is None):
        raise ValueError("Supply both explicit row boundaries or neither")
    if profile["entities"] is not None and (
        not isinstance(profile["entities"], list) or not profile["entities"]
    ):
        raise ValueError("entities must be a nonempty list or null")
    if not isinstance(profile["input_root"], str) or not profile["input_root"]:
        raise ValueError("input_root must be a nonempty path string")
    return profile


def source_groups(dataset, input_root, profile, entities=None):
    """Resolve source files before hashing or reading their values."""
    root = Path(input_root).resolve()
    singleton = dataset in ("iccad", "swat")
    indexed, common = {}, {}
    for role, pattern in profile["sources"].items():
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("Source patterns must stay inside input_root")
        paths = sorted(root.glob(pattern))
        if not paths or any(
            not p.is_file() or not p.resolve().is_relative_to(root) for p in paths
        ):
            raise ValueError(f"Missing or invalid source for {role}: {pattern}")
        if singleton or (dataset == "nasa" and role == "labels"):
            if len(paths) != 1:
                raise ValueError(f"Expected one source for {role}")
            common[role] = paths[0]
        else:
            names = [
                p.name.rsplit(".", 2)[0] if dataset == "iops" else p.stem for p in paths
            ]
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate entities in {role}")
            indexed[role] = dict(zip(names, paths))
    if dataset == "nasa":
        # P-2 has no verified recording match; T-10 has no annotation row.
        for files in indexed.values():
            for excluded in ("P-2", "T-10"):
                files.pop(excluded, None)
    available = (
        {("console" if dataset == "iccad" else "swat")}
        if singleton
        else set(indexed["train"])
    )
    selected = entities if entities is not None else profile["entities"]
    if (
        dataset == "nasa"
        and selected is not None
        and any(e in selected for e in ("P-2", "T-10"))
    ):
        raise ValueError(
            "NASA P-2 and T-10 are excluded: source labels cannot be verified"
        )
    selected = sorted(available if selected is None else selected)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one distinct entity")
    groups = {}
    for entity in selected:
        if (
            not isinstance(entity, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", entity)
            or entity not in available
        ):
            raise ValueError(f"Unknown or unsafe entity: {entity}")
        if any(entity not in files for files in indexed.values()):
            raise ValueError(f"Source roles do not match for entity: {entity}")
        groups[entity] = {
            **common,
            **{role: files[entity] for role, files in indexed.items()},
        }
    if (
        entities is None
        and profile["entities"] is None
        and any(set(v) != available for v in indexed.values())
    ):
        raise ValueError("Source entity sets do not match")
    training_paths = {v["train"].resolve() for v in groups.values() if "train" in v}
    testing_paths = {v["test"].resolve() for v in groups.values() if "test" in v}
    if training_paths & testing_paths:
        raise ValueError("Train and test source files overlap")
    return groups


def make_frame(values, path, root, features, labels=None, timestamps=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] != len(features):
        raise ValueError(f"Invalid feature shape: {path.name}")
    if not features or len(set(features)) != len(features) or set(features) & set(META):
        raise ValueError(f"Invalid feature names: {path.name}")
    frame = pd.DataFrame(values, columns=features)
    frame["source_file"] = path.relative_to(root).as_posix()
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(frame) or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Invalid or missing binary labels: {path.name}")
    frame["label"] = pd.array(
        [pd.NA] * len(frame) if labels is None else labels, dtype="Int8"
    )
    if timestamps is not None:
        times = pd.to_datetime(timestamps, errors="raise")
        if len(times) != len(frame) or pd.isna(times).any():
            raise ValueError(f"Invalid timestamps: {path.name}")
        frame["timestamp"] = times.array
    return frame


def array_entity(dataset, entity, paths, root):
    frames, features = {}, None
    for split in ("train", "test"):
        path = paths[split]
        values = (
            np.load(path, allow_pickle=False)
            if dataset == "nasa"
            else np.loadtxt(path, delimiter=",", ndmin=2)
        )
        labels = None
        if dataset == "iops":
            if values.shape[1] != 2:
                raise ValueError(f"IOPS requires value and label columns: {path.name}")
            values, labels = values[:, :1], values[:, 1]
        if values.ndim != 2:
            raise ValueError(f"Expected a matrix: {path.name}")
        names = [f"feature_{i}" for i in range(values.shape[1])]
        if features is not None and names != features:
            raise ValueError(f"Feature mismatch: {entity}")
        features = names
        if split == "test" and dataset == "smd":
            labels = np.loadtxt(paths["labels"], delimiter=",", ndmin=1)
        if split == "test" and dataset == "nasa":
            labels = nasa_labels(paths["labels"], entity, len(values))
        frames[split] = make_frame(values, path, root, features, labels)
    return EntityData(entity, frames, features)


def nasa_labels(path, entity, length):
    annotations = pd.read_csv(path)
    rows = annotations.loc[annotations["chan_id"] == entity]
    if len(rows) != 1 or int(rows.iloc[0]["num_values"]) != length:
        raise ValueError(f"NASA annotation mismatch: {entity}")
    labels = np.zeros(length, dtype=np.int8)
    intervals = ast.literal_eval(rows.iloc[0]["anomaly_sequences"])
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"Invalid NASA interval: {entity}")
        start, end = interval
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end < length
        ):
            raise ValueError(f"NASA interval out of bounds: {entity}")
        labels[start : end + 1] = 1
    return labels


def iccad_entity(entity, paths, root):
    names = pq.ParquetFile(paths["data"]).schema_arrow.names
    # Match the status code, endpoint, and statistic at the end of a name.
    features = [
        name for name in names if re.search(r"_5[0-9]{2}_endpoint[^_]+_count$", name)
    ]
    if not features or "interval_start" not in names:
        raise ValueError("ICCAD requires interval_start and 5XX count columns")
    raw = pd.read_parquet(paths["data"], columns=["interval_start", *features])
    times = pd.to_datetime(raw["interval_start"], unit="s", utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("ICCAD has missing timestamps")
    labels = np.zeros(len(raw), dtype=np.int8)
    for row in pd.read_csv(paths["labels"]).itertuples(index=False):
        start, end = (
            pd.to_datetime(row.anomaly_start, utc=True),
            pd.to_datetime(row.anomaly_end, utc=True),
        )
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("Invalid ICCAD anomaly interval")
        overlap = (times <= end) & (times + pd.Timedelta(seconds=300) > start)
        labels[overlap.to_numpy()] = 1
    frame = make_frame(
        raw[features].to_numpy(), paths["data"], root, features, labels, times
    )
    return EntityData(entity, {"sequence": frame}, features)


def swat_entity(entity, paths, root):
    frames, features = {}, None
    for split in ("train", "test"):
        raw = pd.read_csv(paths[split], low_memory=False)
        raw.columns = raw.columns.str.strip()
        if raw.columns.duplicated().any() or not set(EXCLUDED).issubset(raw.columns):
            raise ValueError("SWaT has duplicate columns or missing excluded columns")
        features_now = [
            c for c in raw if c not in (*EXCLUDED, "Timestamp", "Normal/Attack")
        ]
        if len(features_now) != 44 or (
            features is not None and features_now != features
        ):
            raise ValueError(
                "SWaT requires the same 44 retained features in each split"
            )
        features = features_now
        labels = (
            raw["Normal/Attack"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"normal": 0, "attack": 1})
        )
        times = pd.to_datetime(
            raw["Timestamp"].astype(str).str.strip(), format="%d/%m/%Y %I:%M:%S %p"
        )
        frames[split] = make_frame(
            raw[features].apply(pd.to_numeric).to_numpy(),
            paths[split],
            root,
            features,
            labels,
            times,
        )
    normal, attack = frames["train"], frames["test"]
    cuts = np.flatnonzero(normal.timestamp.diff().dt.total_seconds().to_numpy() < 0)
    if len(cuts) != 2 or not normal.label.eq(0).all() or not attack.label.eq(1).all():
        raise ValueError(
            "SWaT source blocks or source labels differ from the approved layout"
        )
    full, trimmed = normal.iloc[cuts[0] : cuts[1]], normal.iloc[cuts[1] :]
    offset = len(full) - len(trimmed)
    columns = ["timestamp", *features]
    if offset <= 0 or not full.iloc[offset:][columns].reset_index(drop=True).equals(
        trimmed[columns].reset_index(drop=True)
    ):
        raise ValueError(
            "SWaT source blocks do not contain matching full and trimmed normal versions"
        )
    if trimmed.timestamp.iloc[0] - full.timestamp.iloc[0] != pd.Timedelta(minutes=30):
        raise ValueError("SWaT trimmed normal version must omit the first 30 minutes")
    # Keep original file names and row positions when rebuilding the test source.
    frames = {
        "train": trimmed.copy(),
        "test": pd.concat([normal.iloc[: cuts[0]], attack], ignore_index=True),
    }
    return EntityData(entity, frames, features)


def read_entities(dataset, input_root, profile, entities=None):
    """Yield one source entity without applying shared preprocessing rules."""
    root = Path(input_root).resolve()
    for entity, paths in source_groups(dataset, root, profile, entities).items():
        try:
            if dataset == "iccad":
                yield iccad_entity(entity, paths, root)
            elif dataset == "swat":
                yield swat_entity(entity, paths, root)
            else:
                yield array_entity(dataset, entity, paths, root)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{dataset}/{entity}: {error}") from error


def prepare_splits(record, profile):
    ordered = {}
    for name, frame in record.frames.items():
        if "timestamp" in frame:
            if profile["sort_timestamps"]:
                frame = frame.sort_values("timestamp", kind="stable")
            elif not frame["timestamp"].is_monotonic_increasing:
                raise ValueError(f"{record.entity}/{name}: timestamps go backward")
        ordered[name] = frame.reset_index(drop=True)
    if "sequence" in ordered:
        frame = ordered["sequence"]
        train_end = profile["train_end"] or int(len(frame) * profile["train_ratio"])
        test_start = profile["test_start"] or int(
            len(frame) * profile["test_start_ratio"]
        )
        if not 0 < train_end < test_start < len(frame):
            raise ValueError(f"{record.entity}: invalid configured split boundaries")
        if frame["label"].isna().any():
            raise ValueError(f"{record.entity}: new splits require known labels")
        anomalies = np.flatnonzero(frame["label"].to_numpy() == 1)
        if len(anomalies):
            train_end = min(train_end, int(anomalies[0]))
        if not 0 < train_end < test_start < len(frame):
            raise ValueError(
                f"{record.entity}: split boundaries leave an empty or invalid split"
            )
        return {
            "train": frame.iloc[:train_end].copy(),
            "validation": frame.iloc[train_end:test_start].copy(),
            "test": frame.iloc[test_start:].copy(),
        }
    if profile["train_end"] is not None:
        raise ValueError("Explicit row boundaries apply only to a new sequence")
    training, testing = ordered["train"], ordered["test"]
    if (
        "timestamp" in training
        and training["timestamp"].iloc[-1] >= testing["timestamp"].iloc[0]
    ):
        raise ValueError(
            f"{record.entity}: source timeline overlaps; supply a valid original split"
        )
    count = int(len(training) * profile["validation_ratio"])
    if not 0 < count < len(training) or testing.empty:
        raise ValueError(f"{record.entity}: split boundaries leave an empty split")
    return {
        "train": training.iloc[:-count].copy(),
        "validation": training.iloc[-count:].copy(),
        "test": testing.copy(),
    }


def scale_splits(record, splits, profile):
    missing_counts = {}
    for name, frame in splits.items():
        values = frame[record.features].to_numpy(dtype=np.float64)
        missing_counts[name] = dict(
            zip(record.features, np.isnan(values).sum(axis=0).tolist())
        )
        if profile["missing_values"] == "zero_counts":
            values = np.where(np.isnan(values), 0.0, values)
        bad = ~np.isfinite(values)
        if bad.any():
            columns = np.asarray(record.features)[bad.any(axis=0)].tolist()
            raise ValueError(
                f"{record.entity}/{name}: missing or infinite features: {columns}"
            )
        frame[record.features] = values
    training = splits["train"][record.features].to_numpy()
    if profile["scaler"] == "robust":
        lower, center, upper = np.quantile(
            training, [0.25, 0.5, 0.75], axis=0, method="linear"
        )
        divisor = upper - lower
    else:
        center, divisor = training.mean(axis=0), training.std(axis=0, ddof=0)
    zero = divisor == 0
    divisor = np.where(zero, 1.0, divisor)
    if not np.isfinite(center).all() or not np.isfinite(divisor).all():
        raise ValueError(f"{record.entity}: scaler parameters are not finite")
    for name, frame in splits.items():
        frame[record.features] = (frame[record.features].to_numpy() - center) / divisor
        if not np.isfinite(frame[record.features].to_numpy()).all():
            raise ValueError(f"{record.entity}/{name}: scaled values are not finite")
    return {
        "type": profile["scaler"],
        "features": record.features,
        "missing_counts_before_fill": missing_counts,
        "center": center.tolist(),
        "divisor": divisor.tolist(),
        "zero_divisor_features": np.asarray(record.features)[zero].tolist(),
        "fit_rows": len(training),
        "quantile_method": "linear",
        "standard_ddof": 0,
    }


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def check_existing(directory, identity, entities):
    """Reuse only a complete version whose recorded outputs still match."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        expected = {
            f"{entity}/{name}"
            for entity in entities
            for name in (
                "train.parquet",
                "validation.parquet",
                "test.parquet",
                "scaler.json",
            )
        }
        if manifest["identity"] != identity or manifest["status"] != "complete":
            raise ValueError("Manifest identity or completion status differs")
        if manifest["version_id"] != directory.name:
            raise ValueError("Manifest version ID differs from its directory")
        if set(manifest["files"]) != expected or set(manifest["entities"]) != set(
            entities
        ):
            raise ValueError("Manifest does not list all required outputs")
        for relative, digest in manifest["files"].items():
            path = directory / relative
            if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
                raise ValueError(f"Output is missing or damaged: {relative}")
            if path.suffix == ".parquet":
                entity, split = path.parent.name, path.stem
                rows = pq.read_metadata(path).num_rows
                if (
                    rows < 1
                    or rows != manifest["entities"][entity]["splits"][split]["rows"]
                ):
                    raise ValueError(f"Output row count differs: {relative}")
            else:
                json.loads(path.read_text())
        return directory
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Incomplete or damaged version: {directory}. No files were overwritten. {error}"
        ) from error


def save_entity(directory, record, splits, scaler, original):
    folder = directory / record.entity
    folder.mkdir()
    summary = {
        "features": record.features,
        "original_source_splits": original,
        "splits": {},
    }
    for name, frame in splits.items():
        frame.to_parquet(folder / f"{name}.parquet", index=False, engine="pyarrow")
        timestamps = frame.get("timestamp")
        summary["splits"][name] = {
            "rows": len(frame),
            "labeled_anomalies": int(frame["label"].eq(1).sum()),
            "unknown_labels": int(frame["label"].isna().sum()),
            "first_source_row": int(frame["source_row"].iloc[0]),
            "last_source_row": int(frame["source_row"].iloc[-1]),
            "start_time": None if timestamps is None else str(timestamps.iloc[0]),
            "end_time": None if timestamps is None else str(timestamps.iloc[-1]),
        }
    summary["time_order"] = "source_row_order; elapsed time unverified"
    if "timestamp" in splits["train"]:
        timezone = splits["train"]["timestamp"].dt.tz
        summary["time_order"] = (
            "UTC" if timezone is not None else "source wall clock; timezone unspecified"
        )
    write_json(folder / "scaler.json", scaler)
    return summary


def preprocess(
    dataset, profile_path=None, input_root=None, output_root=None, entities=None
):
    """Prepare raw entities and return their checked version directory."""
    profile = load_profile(dataset, profile_path)
    root = Path(input_root or profile["input_root"]).resolve()
    profile["input_root"] = str(root)
    groups = source_groups(dataset, root, profile, entities)
    profile["entities"] = list(groups)
    sources = sorted({path for group in groups.values() for path in group.values()})
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in sources}
    identity = {
        "profile": profile,
        "sources": hashes,
        "code_sha256": file_hash(__file__),
    }
    version = hashlib.sha256(json_bytes(identity)).hexdigest()
    directory = (
        Path(output_root or ROOT / DEFAULT_OUTPUT_ROOT).resolve() / dataset / version
    )
    if directory.exists():
        return check_existing(directory, identity, groups)
    directory.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for record in read_entities(dataset, root, profile, list(groups)):
        print(f"Preparing {dataset}/{record.entity}", flush=True)
        original = {name: len(frame) for name, frame in record.frames.items()}
        splits = prepare_splits(record, profile)
        scaler = scale_splits(record, splits, profile)
        summaries[record.entity] = save_entity(
            directory, record, splits, scaler, original
        )
    if hashes != {p.relative_to(root).as_posix(): file_hash(p) for p in sources}:
        raise ValueError(
            "Source contents changed during preprocessing; output remains incomplete"
        )
    files = {
        p.relative_to(directory).as_posix(): file_hash(p)
        for p in sorted(directory.glob("*/*"))
    }
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "pyyaml": yaml.__version__,
        "python": sys.version.split()[0],
    }
    write_json(
        directory / "manifest.json",
        {
            "status": "complete",
            "version_id": version,
            "identity": identity,
            "entities": summaries,
            "files": files,
            "libraries": versions,
        },
    )
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--profile", type=Path, help="Optional YAML profile overrides")
    parser.add_argument("--input-root", type=Path, help="Raw dataset directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        help=f"Parent of dataset/version directories (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--entities", nargs="+", help="Process only these source entity IDs"
    )
    parser.add_argument(
        "--show-profile", action="store_true", help="Print resolved settings and stop"
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.show_profile:
            profile = load_profile(arguments.dataset, arguments.profile)
            if arguments.input_root is not None:
                profile["input_root"] = str(arguments.input_root.resolve())
            if arguments.entities is not None:
                profile["entities"] = arguments.entities
            print(json.dumps(profile, indent=2, sort_keys=True))
            return 0
        directory = preprocess(
            arguments.dataset,
            arguments.profile,
            arguments.input_root,
            arguments.output_root,
            arguments.entities,
        )
        print(f"Complete, checked version: {directory}", flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f"Preprocessing stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from hashlib import sha256
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.prepared_data import iter_batches, load_entity


@pytest.fixture
def prepared(tmp_path):
    entity = tmp_path / "machine"
    entity.mkdir()
    features = ["z", "a"]
    files = {}
    splits = {}
    for split in ("train", "validation", "test"):
        columns = {
            "a": list(range(10, 18)),
            "z": list(range(8)),
            "source_file": ["missing/raw.csv"] * 8,
            "source_row": list(range(8)),
            "label": [None, 0, 0, 0, 0, 0, 1, None],
        }
        if split == "test":
            columns["timestamp"] = pa.array(
                np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype="datetime64[s]")
            )
        path = entity / f"{split}.parquet"
        pq.write_table(pa.table(columns), path)
        files[str(path.relative_to(tmp_path))] = sha256(path.read_bytes()).hexdigest()
        splits[split] = {"rows": 8}
    scaler = entity / "scaler.json"
    scaler.write_text(json.dumps({"features": features}))
    files["machine/scaler.json"] = sha256(scaler.read_bytes()).hexdigest()
    manifest = {
        "status": "complete",
        "version_id": tmp_path.name,
        "identity": {"settings": {"dataset": "smd"}},
        "files": files,
        "entities": {"machine": {"features": features, "splits": splits}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.mark.parametrize("name", ["train.parquet", "scaler.json"])
def test_rejects_hash_mismatch(prepared, name):
    (prepared / "machine" / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_entity(prepared, "machine")


def test_preserves_schema_and_missing_labels(prepared):
    entity = load_entity(prepared, "machine")
    assert entity["features"] == ["z", "a"]
    test = entity["splits"]["test"]
    np.testing.assert_array_equal(test["x"][0], [0, 10])
    assert np.isnan(test["labels"][0])
    assert test["labels"][6] == 1
    assert list(test["source_row"]) == list(range(8))
    assert test["source_file"][0] == "missing/raw.csv"
    assert "timestamps" in test and "timestamps" not in entity["splits"]["train"]


def settings():
    return {
        "window_size": 3,
        "batch_size": 4,
        "train_stride": 3,
        "validation_stride": 3,
        "test_stride": 3,
        "online_stride": 1,
        "shuffle_train": False,
        "tail_policy": {
            "train": "drop",
            "validation": "end_align",
            "test": "end_align",
            "online": "drop",
        },
        "gap_policy": "break_at_time_gaps",
        "expected_interval_seconds": 1,
    }


def test_gap_tail_and_online_order(prepared):
    split = load_entity(prepared, "machine")["splits"]["test"]
    offline = iter_batches(split, settings(), phase="offline", seed=6)
    assert iter(offline) is offline
    batch = next(offline)
    assert batch["x"].shape == (4, 3, 2)
    expected = [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]]
    assert batch["indices"].tolist() == expected
    assert next(offline, None) is None
    stream = list(iter_batches(split, settings(), phase="online", seed=6))
    assert [item["indices"].tolist()[0] for item in stream] == expected
    assert all(item["x"].shape == (1, 3, 2) for item in stream)


def test_short_splits_are_not_joined(prepared):
    train = load_entity(prepared, "machine")["splits"]["train"]
    data = settings() | {"window_size": 9}
    with pytest.raises(ValueError, match="no usable windows"):
        list(iter_batches(train, data, phase="offline", seed=6))


def test_does_not_read_source_or_transform_features(prepared):
    assert not (prepared / "missing").exists()
    data = load_entity(prepared, "machine")
    for split in data["splits"].values():
        np.testing.assert_array_equal(
            split["x"], np.column_stack((np.arange(8), np.arange(10, 18)))
        )


def test_discovery_prefers_valid_profile_and_orders_fallback(tmp_path):
    from os import utime

    from src.prepared_data import discover_versions

    def add(name, stamp, dataset="smd", status="complete", external=False):
        base = tmp_path / ("external" if external else "data/preprocessed/smd")
        path = base / name
        path.mkdir(parents=True)
        manifest = path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": status,
                    "version_id": name,
                    "identity": {"profile": {"dataset": dataset}},
                    "entities": {"machine": {}},
                }
            )
        )
        utime(manifest, (stamp, stamp))
        return path

    old = add("old", 1, external=True)
    first = add("a", 2)
    second = add("b", 2)
    add("wrong", 3, dataset="swat")
    add("incomplete", 4, status="partial")
    bad = add("bad", 5)
    (bad / "manifest.json").write_text("{")
    assert [path for path, _ in discover_versions(tmp_path, "smd", old)] == [
        old,
        first,
        second,
    ]
    assert [path for path, _ in discover_versions(tmp_path, "smd", bad)] == [
        first,
        second,
    ]

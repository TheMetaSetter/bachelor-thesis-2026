"""Check offline selection, validation calibration, and durable recovery."""

import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.artifacts import allocate_run, load_resume, save_checkpoint


class _Method:
    def __init__(self, config, events):
        self.config = config
        self.events = events
        self.weight = 0.0
        self.identity = {"method": "mustard", "variant": "O0"}

    def state_dict(self):
        return {"weight": self.weight}

    def load_state_dict(self, state):
        self.weight = state["weight"]

    def fit(
        self, train, validation, *, checkpoint_dir, resume=None, on_checkpoint=None
    ):
        assert train["identity"]["split"] == "train"
        assert validation["identity"]["split"] == "validation"
        initial = (
            resume["initial"]
            if resume
            else save_checkpoint(
                checkpoint_dir / "stage_a.initial.pt",
                self.state_dict(),
                identity=self.identity,
                role="stage_a.initial",
            )
        )
        if resume:
            self.load_state_dict(resume["method_state"])
        for epoch in range(resume["next_epoch"] if resume else 0, 2):
            if epoch == 1 and self.events.get("interrupt"):
                raise KeyboardInterrupt
            self.events["epochs"].append(epoch)
            self.weight += (float(torch.rand(())) + float(np.random.random())) / 10
            on_checkpoint(
                {
                    "next_epoch": epoch + 1,
                    "initial": initial,
                    "method_state": self.state_dict(),
                }
            )
        best = save_checkpoint(
            checkpoint_dir / "stage_a.best.pt",
            self.state_dict(),
            identity=self.identity,
            role="stage_a.best",
        )
        self.weight = -100.0
        return {
            "selected_checkpoint": best,
            "stages": [
                {
                    "name": "stage_a",
                    "initial_checkpoint": initial,
                    "best_checkpoint": best,
                }
            ],
        }

    def score(self, batch):
        assert "labels" not in batch
        assert self.weight >= 0
        self.events["scored"].append(batch["identity"]["split"])
        self.events["checkpoints"].append(self.weight)
        return {
            "score_kind": "window",
            "indices": batch["indices"],
            "raw_scores": batch["x"][..., 0] + self.weight,
        }


@pytest.fixture
def offline(tmp_path, monkeypatch):
    module = importlib.import_module("src.offline")
    identity = {
        "dataset": "smd",
        "entity": "machine",
        "version_id": "fixture",
        "features": ["x"],
        "window_size": 2,
    }
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        splits[split] = {
            "x": (np.arange(1, 7) + index * 100)[:, None].astype(float),
            "labels": np.array([0, 0, 0, np.nan, 1, 0]),
            "identity": dict(identity, split=split),
        }
    entity = {
        "identity": identity,
        "features": ["x"],
        "scaler": {"divisor": [1]},
        "splits": splits,
    }
    data = {
        "prepared_root": tmp_path,
        "window_size": 2,
        "batch_size": 8,
        "shuffle_train": False,
        "train_stride": 2,
        "validation_stride": 2,
        "test_stride": 2,
        "online_stride": 1,
        "gap_policy": "break_at_time_gaps",
        "expected_interval_seconds": 1,
        "tail_policy": dict.fromkeys(("train", "validation", "test", "online"), "drop"),
        "synthetic_profile": "data_profiles/synthetic_profiles/fixture.yaml",
    }
    events = {"epochs": [], "scored": [], "checkpoints": []}
    monkeypatch.setattr(module, "load_entity", lambda *_: entity)
    monkeypatch.setattr(module, "create_method", lambda config: _Method(config, events))
    synthetic = tmp_path / "configs/data_profiles/synthetic_profiles/fixture.yaml"
    synthetic.parent.mkdir(parents=True)
    synthetic.write_text("fixture: true\n")
    config = {
        "project_root": tmp_path,
        "dataset": "smd",
        "entity": "machine",
        "seed": 7,
        "device": "cpu",
        "phase": "offline",
        "method_name": "mustard",
        "variant": "O0",
        "prepared_identity": identity,
        "data": data,
        "limits": {},
        "method": {
            "phases": ["offline"],
            "training": {"stages": [{"name": "stage_a"}]},
        },
        "evaluation": {
            "score_space": "latent",
            "offline_threshold_split": "validation",
            "offline_threshold_quantile": 0.95,
            "metrics": ["fpr"],
            "offline_overlap_reduction": "last_ewma_estimate",
        },
    }
    return module, config, events


def _record(run, file):
    return json.loads((run["output_root"] / file).read_text())


@pytest.mark.parametrize("space", ["raw_input", "latent"])
def test_offline_restores_best_and_fits_q95_on_validation(offline, space):
    module, config, events = offline
    config["evaluation"]["score_space"] = space
    run = allocate_run(config, project_root=config["project_root"])
    result = module.run_offline(run)
    saved = _record(run, "state/calibration.json")
    assert result["status"] == "successful"
    assert saved["score_space"] == space
    assert saved["source_split"] == "validation"
    weight = events["checkpoints"][0]
    assert saved["point_threshold"] == pytest.approx(
        np.quantile([101, 102, 103, 104], 0.95) + weight
    )
    assert saved["unknown_label_windows"] == 1
    assert events["scored"][0] == "validation"
    assert _record(run, "test/result.json")["metrics"]["fpr"] == 1
    for role in ("initial", "best"):
        assert (run["output_root"] / f"state/stage_a.{role}.pt.json").is_file()


def test_synthetic_smd_uses_saved_validation_and_two_masks(offline, monkeypatch):
    module, config, events = offline
    config["evaluation"].update(
        score_space="normalized_input",
        offline_threshold_split="synthetic_validation",
        offline_point_threshold_source="synthetic_validation_normal",
        offline_window_threshold_source="synthetic_validation_normal",
        offline_threshold_quantile=0.99,
        offline_overlap_reduction="mean",
        triage={"window_quantile": 0.99},
    )

    def augment(x, settings, *, generator):
        assert settings["fixture"]
        assert x[:, 0, 0].tolist() == [101, 103, 105]
        mask = torch.zeros(x.shape[:2], dtype=torch.bool)
        mask[0, 0] = True
        return x, {"synthetic_anomaly_mask": mask, "class_rotation_offset": 0}

    monkeypatch.setattr(module, "augment", augment)
    run = allocate_run(config, project_root=config["project_root"])
    module.run_offline(run)
    saved = _record(run, "state/calibration.json")
    weight = events["checkpoints"][0]
    assert saved["source_split"] == "synthetic_validation_normal"
    assert saved["generation"]["source_split"] == "validation"
    assert saved["point_threshold"] == pytest.approx(
        np.quantile([102, 103, 104], 0.99) + weight
    )
    assert saved["window_threshold"] == pytest.approx(103.5 + weight)
    assert saved["unknown_label_windows"] == 1


def test_interrupted_offline_fit_matches_continuation(offline):
    module, config, events = offline
    paths = []
    for interrupt in (False, True):
        run = allocate_run(config, project_root=config["project_root"])
        if interrupt:
            events["epochs"].clear()
            events["interrupt"] = True
            with pytest.raises(KeyboardInterrupt):
                module.run_offline(run)
            assert _record(run, "state/run.json")["status"] == "interrupted"
            snapshot = load_resume(run)
            assert snapshot["method_state"]["next_epoch"] == 1
            events["interrupt"] = False
            module.run_offline(run, resume=snapshot)
            assert events["epochs"] == [0, 1]
        else:
            module.run_offline(run)
        paths.append(run["output_root"])

    def scores(path: Path):
        return pq.read_table(path / "test/point_scores.parquet")["score"].to_pylist()

    assert scores(paths[0]) == scores(paths[1])


def test_endpoint_calibration_checks_the_whole_source_window(offline, monkeypatch):
    module, config, events = offline
    data = module.load_entity()
    data["splits"]["validation"]["labels"][:] = [1, 0, np.nan, 0, 1, 0]
    original = _Method.score

    def score(method, batch):
        record = original(method, batch)
        return {
            "score_kind": "point",
            "indices": record["indices"][:, -1],
            "raw_scores": record["raw_scores"][:, -1],
        }

    monkeypatch.setattr(_Method, "score", score)
    run = allocate_run(config, project_root=config["project_root"])
    module.run_offline(run)
    calibration = _record(run, "state/calibration.json")
    assert calibration["point_threshold"] == pytest.approx(
        104 + events["checkpoints"][0]
    )
    assert calibration["excluded_windows"] == 2
    assert calibration["unknown_label_windows"] == 1


def test_epoch_monitor_splits_before_it_builds_windows(offline, monkeypatch):
    from src.mustard_validation import evaluate_epoch, split_validation
    from src.prepared_data import iter_batches

    module, config, _ = offline
    config["data"]["validation_stride"] = 1
    config["synthetic"] = {}
    config["evaluation"]["metrics"] = ["fpr", "segment_mean_affiliation_f1"]
    identity = config["prepared_identity"] | {"split": "validation"}
    split = {
        "x": np.array([0, 1, 2, 3, 4, 5, 6, 2, 8, 9, 10], dtype=float)[:, None],
        "labels": np.array([1, np.nan, 0, 0, 0, np.nan, 1, 0, 0, 0, 0]),
        "timestamps": np.array(
            [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype="datetime64[s]"
        ),
        "identity": identity,
    }
    calibration, held_out = split_validation(split)
    windows = [
        np.concatenate(
            [
                batch["x"][..., 0]
                for batch in iter_batches(
                    part, config["data"], phase="offline", seed=config["seed"]
                )
            ]
        ).tolist()
        for part in (calibration, held_out)
    ]
    assert windows == [
        [[0, 1], [2, 3], [3, 4]],
        [[5, 6], [6, 2], [2, 8], [8, 9], [9, 10]],
    ]
    assert calibration["identity"] == held_out["identity"] == identity

    masks = []

    def augment(values, settings, *, generator):
        del settings, generator
        mask = torch.zeros(values.shape[:2], dtype=torch.bool)
        mask[0, 1] = True
        masks.append(mask)
        return values, {"synthetic_anomaly_mask": mask, "class_rotation_offset": 1}

    def score(batch, **_):
        assert "labels" not in batch
        return {
            "score_kind": "window",
            "indices": batch["indices"],
            "raw_scores": np.asarray(batch["x"][..., 0]),
        }

    monkeypatch.setattr(module, "augment", augment)
    from types import SimpleNamespace

    method = SimpleNamespace(score=score, config=config)
    labels = split["labels"].copy()
    result = evaluate_epoch(method, split, config, stage="stage_a")
    assert result["threshold"] == pytest.approx(3.75)
    assert result["held_out_fpr"] == pytest.approx(8 / 10)
    assert result["synthetic_metrics"]["fpr"] == pytest.approx(7 / 9)
    assert result["synthetic_metrics"]["segment_mean_affiliation_f1"] is not None
    assert result["score_space"] == "raw_input"
    assert (
        evaluate_epoch(method, split, config, stage="stage_b")["score_space"]
        == "latent"
    )
    np.testing.assert_array_equal(split["labels"], labels)
    assert all(mask.sum() == 1 and mask[0, 1] for mask in masks)

    for positions, reason in [
        ([0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15], "no_eligible_calibration_scores"),
        ([0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16], "no_held_out_scores"),
    ]:
        timestamps = np.array(positions, dtype="datetime64[s]")
        result = evaluate_epoch(
            method, split | {"timestamps": timestamps}, config, stage="stage_a"
        )
        assert result["held_out_fpr"] is None
        assert result["reason"] == reason


def test_monitored_checkpoint_resume_matches_uninterrupted_run(offline, monkeypatch):
    from src.config import create_method, load_yaml
    from src.methods.mustard.model import Mustard

    module, config, _ = offline
    root = Path(__file__).resolve().parents[1] / "configs"
    method_config = load_yaml(root / "method_profiles/mustard_o0_offline.yaml")
    for stage in method_config["training"]["stages"]:
        stage["epochs"] = 2
    entity = module.load_entity()
    for split in entity["splits"].values():
        split["x"] = np.random.default_rng(7).normal(size=(144, 1))
        split["labels"] = np.zeros(144)
    config["data"].update(window_size=12, train_stride=12, validation_stride=12)
    config["prepared_identity"]["window_size"] = 12
    config.update(
        method=method_config,
        encoder={
            "input_features": 1,
            "hidden_channels": 2,
            "output_dim": 2,
            "kernel_size": 1,
            "num_layers": 2,
            "dropout": 0.0,
        },
        synthetic=load_yaml(
            root / "data_profiles/synthetic_profiles/redlamp_legacy.yaml"
        ),
    )
    config["evaluation"].update(
        offline_monitor="held_out_fpr", vus_max_buffer_size=2, vus_num_thresholds=8
    )
    monkeypatch.setattr(module, "create_method", create_method)
    original = Mustard._epoch
    remaining = None

    def epoch(method, split, settings, *, fitting):
        nonlocal remaining
        if fitting and remaining is not None:
            if remaining == 0:
                remaining = None
                raise KeyboardInterrupt
            remaining -= 1
        return original(method, split, settings, fitting=fitting)

    monkeypatch.setattr(Mustard, "_epoch", epoch)
    first = allocate_run(config, project_root=config["project_root"])
    expected = module.run_offline(first)
    second = allocate_run(config, project_root=config["project_root"])
    remaining = 1
    with pytest.raises(KeyboardInterrupt):
        module.run_offline(second)
    snapshot = load_resume(second)
    assert snapshot["method_state"]["next_epoch"] == 1
    actual = module.run_offline(second, resume=snapshot)
    for expected_stage, actual_stage in zip(
        expected["stages"], actual["stages"], strict=True
    ):
        assert actual_stage["epochs"] == expected_stage["epochs"]
        assert actual_stage["best_held_out_fpr"] == expected_stage["best_held_out_fpr"]
        space = "raw_input" if actual_stage["name"] == "stage_a" else "latent"
        assert all(
            row["monitor"]["score_space"] == space for row in actual_stage["epochs"]
        )
    for run in (first, second):
        assert _record(run, "state/calibration.json")["source_split"] == "validation"
    assert (first["output_root"] / "test/point_scores.parquet").read_bytes() == (
        second["output_root"] / "test/point_scores.parquet"
    ).read_bytes()


def test_final_uncertainty_uses_separate_validation_and_test_scores(
    offline, monkeypatch
):
    module, config, events = offline
    splits = module.load_entity()["splits"]
    original_labels = {name: split["labels"].copy() for name, split in splits.items()}
    field = "point_anomaly_score_variance"
    config["evaluation"].update(
        offline_monitor="held_out_fpr",
        uncertainty_fields=[field],
        uncertainty_statistics=["mean"],
    )
    config["synthetic"] = {}
    original_fit, original_score = _Method.fit, _Method.score

    def fit(self, *args, monitor_epoch, **kwargs):
        return original_fit(self, *args, **kwargs)

    def score(self, batch, **options):
        raw = original_score(self, batch)
        split = batch["identity"]["split"]
        value = 20 if options else {"train": 5, "validation": 10, "test": 30}[split]
        return raw | {"uncertainty": {field: np.full(raw["raw_scores"].shape, value)}}

    def augment(values, settings, *, generator):
        return values, {
            "synthetic_anomaly_mask": torch.zeros(values.shape[:2], dtype=torch.bool),
            "class_rotation_offset": 1,
        }

    monkeypatch.setattr(_Method, "fit", fit)
    monkeypatch.setattr(_Method, "score", score)
    monkeypatch.setattr(module, "augment", augment)
    run = allocate_run(config, project_root=config["project_root"])
    module.run_offline(run)
    calibration = _record(run, "state/calibration.json")
    assert calibration["selected_windows"] == 3
    assert calibration["normal_data_assumption"] is True
    assert calibration["validation_label_policy"] == "all_normal_pseudo_labels"
    assert calibration["point_threshold"] == pytest.approx(
        np.quantile(np.arange(101, 107), 0.95) + events["checkpoints"][0]
    )
    validation = _record(run, "validation/result.json")
    assert validation["eligibility"]["eligible_observations"] == 6
    assert validation["metrics"]["fpr"] == pytest.approx(1 / 6)
    test = _record(run, "test/result.json")
    assert test["eligibility"]["eligible_observations"] == 5
    for name, labels in original_labels.items():
        np.testing.assert_array_equal(splits[name]["labels"], labels)
    assert validation["metrics"][field + "_mean"] == 10
    assert validation["synthetic_validation_uncertainty"] == {
        "mean": 20,
        "value_count": 2,
    }
    assert _record(run, "test/result.json")["metrics"][field + "_mean"] == 30


def test_progress_uses_committed_epochs_and_final_results(offline, monkeypatch):
    module, config, _ = offline
    config["method"]["training"]["stages"][0]["epochs"] = 2
    original = _Method.fit

    def fit(self, *args, on_checkpoint, **kwargs):
        def checkpoint(state):
            epoch = state["next_epoch"]
            on_checkpoint(
                state
                | {
                    "stage": "stage_a",
                    "history": [
                        {
                            "epoch": epoch,
                            "train": {
                                "total_loss": 0.2,
                                "loss_terms": {"point_score": 0.1},
                            },
                            "validation": {"total_loss": 0.3},
                        }
                    ],
                }
            )

        return original(self, *args, on_checkpoint=checkpoint, **kwargs)

    monkeypatch.setattr(_Method, "fit", fit)
    run = allocate_run(config, project_root=config["project_root"])
    events = []

    def progress(**event):
        if event["unit"] == "epoch":
            assert load_resume(run)["method_state"]["next_epoch"] == event["completed"]
        events.append(event)

    module.run_offline(run, progress=progress)
    epochs = [event for event in events if event["unit"] == "epoch"]
    assert [event["completed"] for event in epochs] == [1, 2]
    assert all(event["total"] == 2 for event in epochs)
    assert epochs[-1]["stats"]["train point_score"] == 0.1
    assert epochs[-1]["stats"]["val loss"] == 0.3
    assert events[-1]["stage"] == "evaluate"
    assert (
        events[-1]["stats"]["FPR"] == _record(run, "test/result.json")["metrics"]["fpr"]
    )


def test_progress_uses_fit_and_evaluate_milestones_without_epochs(offline):
    module, config, _ = offline
    run = allocate_run(config, project_root=config["project_root"])
    events = []
    module.run_offline(run, progress=lambda **event: events.append(event))
    assert [(e["stage"], e["completed"]) for e in events] == [
        ("fit", 0),
        ("fit", 1),
        ("evaluate", 0),
        ("evaluate", 1),
    ]

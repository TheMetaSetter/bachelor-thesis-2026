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

    def fit(self, train, validation, *, checkpoint_dir, resume=None, on_checkpoint=None):
        assert train["identity"]["split"] == "train"
        assert validation["identity"]["split"] == "validation"
        initial = resume["initial"] if resume else save_checkpoint(
            checkpoint_dir / "stage_a.initial.pt", self.state_dict(),
            identity=self.identity, role="stage_a.initial",
        )
        if resume:
            self.load_state_dict(resume["method_state"])
        for epoch in range(resume["next_epoch"] if resume else 0, 2):
            if epoch == 1 and self.events.get("interrupt"):
                raise KeyboardInterrupt
            self.events["epochs"].append(epoch)
            self.weight += (float(torch.rand(())) + float(np.random.random())) / 10
            on_checkpoint({"next_epoch": epoch + 1, "initial": initial, "method_state": self.state_dict()})
        best = save_checkpoint(checkpoint_dir / "stage_a.best.pt", self.state_dict(),
                               identity=self.identity, role="stage_a.best")
        self.weight = -100.0
        return {"selected_checkpoint": best, "stages": [
            {"name": "stage_a", "initial_checkpoint": initial, "best_checkpoint": best}
        ]}

    def score(self, batch):
        assert "labels" not in batch
        assert self.weight >= 0
        self.events["scored"].append(batch["identity"]["split"])
        self.events["checkpoints"].append(self.weight)
        return {"score_kind": "window", "indices": batch["indices"],
                "raw_scores": batch["x"][..., 0] + self.weight}


@pytest.fixture
def offline(tmp_path, monkeypatch):
    module = importlib.import_module("src.offline")
    identity = {"dataset": "smd", "entity": "machine", "version_id": "fixture",
                "features": ["x"], "window_size": 2}
    splits = {}
    for index, split in enumerate(("train", "validation", "test")):
        splits[split] = {"x": (np.arange(1, 7) + index * 100)[:, None].astype(float),
                         "labels": np.array([0, 0, 0, np.nan, 1, 0]),
                         "identity": dict(identity, split=split)}
    entity = {"identity": identity, "features": ["x"], "scaler": {"divisor": [1]}, "splits": splits}
    data = {"prepared_root": tmp_path, "window_size": 2, "batch_size": 8, "shuffle_train": False,
            "train_stride": 2, "validation_stride": 2, "test_stride": 2, "online_stride": 1,
            "gap_policy": "break_at_time_gaps", "expected_interval_seconds": 1,
            "tail_policy": dict.fromkeys(("train", "validation", "test", "online"), "drop"),
            "synthetic_profile": "data_profiles/synthetic_profiles/fixture.yaml"}
    events = {"epochs": [], "scored": [], "checkpoints": []}
    monkeypatch.setattr(module, "load_entity", lambda *_: entity)
    monkeypatch.setattr(module, "create_method", lambda config: _Method(config, events))
    synthetic = tmp_path / "configs/data_profiles/synthetic_profiles/fixture.yaml"
    synthetic.parent.mkdir(parents=True)
    synthetic.write_text("fixture: true\n")
    config = {"project_root": tmp_path, "dataset": "smd", "entity": "machine", "seed": 7,
              "device": "cpu", "phase": "offline", "method_name": "mustard", "variant": "O0",
              "prepared_identity": identity, "data": data, "limits": {},
              "method": {"phases": ["offline"], "training": {"stages": [{"name": "stage_a"}]}},
              "evaluation": {"score_space": "latent", "offline_threshold_split": "validation",
                             "offline_threshold_quantile": 0.95, "metrics": ["fpr"],
                             "offline_overlap_reduction": "last_ewma_estimate"}}
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
    assert saved["point_threshold"] == pytest.approx(np.quantile([101, 102, 103, 104], 0.95) + weight)
    assert saved["unknown_label_windows"] == 1
    assert events["scored"][0] == "validation"
    assert _record(run, "test/result.json")["metrics"]["fpr"] == 1
    for role in ("initial", "best"):
        assert (run["output_root"] / f"state/stage_a.{role}.pt.json").is_file()


def test_synthetic_smd_uses_saved_validation_and_two_masks(offline, monkeypatch):
    module, config, events = offline
    config["evaluation"].update(score_space="normalized_input", offline_threshold_split="synthetic_validation",
                                offline_point_threshold_source="synthetic_validation_normal",
                                offline_window_threshold_source="synthetic_validation_normal",
                                offline_threshold_quantile=0.99, offline_overlap_reduction="mean",
                                triage={"window_quantile": 0.99})

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
    assert saved["point_threshold"] == pytest.approx(np.quantile([102, 103, 104], 0.99) + weight)
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
        return {"score_kind": "point", "indices": record["indices"][:, -1],
                "raw_scores": record["raw_scores"][:, -1]}

    monkeypatch.setattr(_Method, "score", score)
    run = allocate_run(config, project_root=config["project_root"])
    module.run_offline(run)
    calibration = _record(run, "state/calibration.json")
    assert calibration["point_threshold"] == pytest.approx(104 + events["checkpoints"][0])
    assert calibration["excluded_windows"] == 2
    assert calibration["unknown_label_windows"] == 1

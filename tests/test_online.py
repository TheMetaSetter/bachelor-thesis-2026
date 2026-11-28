"""Check causal scoring, online update limits, and exact continuation."""

import importlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.artifacts import allocate_run, load_resume


class _OnlineMethod:
    def __init__(self, events):
        self.events = events
        self.network = torch.nn.ModuleDict(
            {
                "projector": torch.nn.Linear(1, 1, bias=False),
                "encoder": torch.nn.Linear(1, 1, bias=False),
            }
        )
        self.optimizer = torch.optim.SGD(
            self.network["projector"].parameters(), lr=0.1, momentum=0.9
        )
        self.buffer = 0

    def state_dict(self):
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self.buffer,
        }

    def load_state_dict(self, state):
        self.network.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.buffer = state["buffer"]

    def score(self, batch):
        assert "labels" not in batch
        start = int(batch["indices"][0, 0])
        if start == self.events.get("interrupt"):
            raise KeyboardInterrupt
        self.events["order"].append(("score", start))
        weight = self.network["projector"].weight.item()
        raw = batch["x"][..., 0] + weight + torch.rand(()).item() + self.buffer
        return {
            "score_kind": "window",
            "raw_scores": raw,
            "indices": batch["indices"],
            "auxiliary": {"input_window_score": [1.0], "latent_window_score": [0.5]},
        }

    def update(self, batch, context):
        assert "labels" not in batch and np.isfinite(context["recorded_score"]).all()
        start = int(batch["indices"][0, 0])
        self.events["order"].append(("update", start))
        self.optimizer.zero_grad()
        loss = self.network["projector"].weight.square().sum()
        loss.backward()
        self.optimizer.step()
        self.buffer += 1
        if self.events.get("forbidden"):
            with torch.no_grad():
                self.network["encoder"].weight.add_(1)
        return {"did_update": True, "loss": float(loss.detach())}


@pytest.fixture
def online(tmp_path, monkeypatch):
    module = importlib.import_module("src.online")
    events = {"order": []}
    identity = {
        "dataset": "smd",
        "entity": "machine",
        "version_id": "fixture",
        "features": ["x"],
        "window_size": 2,
    }
    split = {
        "identity": dict(identity, split="test"),
        "x": np.arange(15, dtype=float)[:, None],
        "labels": np.zeros(15),
    }
    entity = {
        "identity": identity,
        "features": ["x"],
        "scaler": {"divisor": [1]},
        "splits": {"test": split},
    }
    config = {
        "project_root": tmp_path,
        "dataset": "smd",
        "entity": "machine",
        "seed": 7,
        "device": "cpu",
        "phase": "online",
        "method_name": "mustard",
        "variant": "A2",
        "prepared_identity": identity,
        "limits": {},
        "data": {
            "prepared_root": tmp_path,
            "window_size": 2,
            "online_stride": 1,
            "gap_policy": "break_at_time_gaps",
            "expected_interval_seconds": 1,
            "tail_policy": {"online": "drop"},
        },
        "method": {"phases": ["online"], "update": {"steps": 1}},
        "evaluation": {"score_space": "latent", "metrics": ["fpr"]},
    }
    calibration = {
        "point_threshold": 0.8,
        "score_space": "latent",
        "transform": {"name": "identity"},
        "source_split": "validation",
        "checkpoint_sha256": "fixture",
        "input_window_threshold": 0.5,
        "latent_window_low_threshold": 0.2,
        "latent_window_high_threshold": 0.7,
    }
    monkeypatch.setattr(module, "load_entity", lambda *_: entity)
    monkeypatch.setattr(module, "create_method", lambda _: _OnlineMethod(events))
    monkeypatch.setattr(module, "_initialize_online", lambda *_: (calibration, None))
    return module, config, events, split


def _allocate(config):
    return allocate_run(config, project_root=config["project_root"])


def _scores(run):
    path = run["output_root"] / "test/point_scores.parquet"
    return pq.read_table(path)["score"].to_numpy()


def _status(run):
    return json.loads((run["output_root"] / "state/run.json").read_text())["status"]


def test_online_scores_and_records_before_update(online, monkeypatch):
    module, config, events, _ = online
    original = module._record_window

    def record(scored, state):
        start = int(scored["indices"][0, 0])
        events["order"].append(("record", start))
        return original(scored, state)

    monkeypatch.setattr(module, "_record_window", record)
    run = _allocate(config | {"limits": {"max_online_steps": 2}})
    module.run_online(run)
    assert events["order"] == [
        (step, start) for start in range(2) for step in ("score", "record", "update")
    ]
    assert _status(run) == "successful"


def test_later_input_cannot_change_finalized_prefix(online):
    module, config, _, split = online
    first, second = _allocate(config), _allocate(config)
    module.run_online(first)
    split["x"][8:] += 1000
    module.run_online(second)
    np.testing.assert_array_equal(_scores(first)[:7], _scores(second)[:7])
    assert _scores(first)[8] != _scores(second)[8]


def test_online_recovers_cursor_rng_optimizer_buffer_and_scores(online):
    module, config, events, _ = online
    original, interrupted = _allocate(config), _allocate(config)
    module.run_online(original)
    events["interrupt"] = 10
    with pytest.raises(KeyboardInterrupt):
        module.run_online(interrupted)
    assert _status(interrupted) == "interrupted"
    state = load_resume(interrupted)
    assert state["cursor"] == 8
    assert state["method_state"]["buffer"] == 8
    assert state["method_state"]["optimizer"]["state"]
    events.clear()
    events["order"] = []
    module.run_online(interrupted, resume=state)
    assert events["order"][0] == ("score", 8)
    np.testing.assert_array_equal(_scores(original), _scores(interrupted))


def test_online_rejects_a_change_to_frozen_weights(online):
    module, config, events, _ = online
    events["forbidden"] = True
    run = _allocate(config)
    with pytest.raises(ValueError, match="allowed"):
        module.run_online(run)
    assert _status(run) == "failed"


@pytest.mark.parametrize("variant", ["A0", "fixed"])
def test_online_frozen_runs_do_not_update(online, variant):
    module, config, events, _ = online
    config.update(variant=variant, method={"phases": ["online"], "update": None})
    module.run_online(_allocate(config))
    assert not any(step == "update" for step, _ in events["order"])


def test_missing_parent_blocks_before_scoring(online, monkeypatch):
    module, config, events, _ = online

    def unavailable(*_):
        raise module.OfflineDependencyUnavailable(
            "no compatible successful offline parent"
        )

    monkeypatch.setattr(module, "_initialize_online", unavailable)
    run = _allocate(config)
    with pytest.raises(ValueError, match="no compatible"):
        module.run_online(run)
    assert _status(run) == "blocked"
    assert not events["order"]


def test_sigint_commits_latest_complete_window(online, monkeypatch):
    import signal

    module, config, _, _ = online
    original = _OnlineMethod.score

    def score(self, batch):
        if int(batch["indices"][0, 0]) == 9:
            signal.raise_signal(signal.SIGINT)
        return original(self, batch)

    monkeypatch.setattr(_OnlineMethod, "score", score)
    run = _allocate(config)
    with pytest.raises(KeyboardInterrupt):
        module.run_online(run)
    assert load_resume(run)["cursor"] == 10


def test_resume_reuses_recorded_parent(online, monkeypatch):
    module, config, events, _ = online
    initialize = module._initialize_online
    artifact = config["project_root"] / "parent.artifact"
    artifact.write_bytes(b"fixture")
    checksum = module._hash(artifact)
    parent = {
        "run_id": "first-parent",
        "checkpoint": {"path": artifact, "sha256": checksum},
        "calibration": artifact,
        "calibration_sha256": checksum,
    }

    monkeypatch.setattr(
        module, "_initialize_online", lambda *args: (initialize(*args)[0], parent)
    )
    events["interrupt"] = 8
    run = _allocate(config)
    with pytest.raises(KeyboardInterrupt):
        module.run_online(run)
    saved = load_resume(run)
    assert saved["offline_parent"] == parent
    events["interrupt"] = None

    def reject_lookup(*_):
        pytest.fail("resume looked for another parent")

    monkeypatch.setattr(module, "_initialize_online", reject_lookup)
    assert module.run_online(run, resume=saved)["offline_parent"] == parent


def _real_m2n2_pair(tmp_path):
    from pathlib import Path

    from src.config import resolve_run
    from src.offline import run_offline

    project = Path(__file__).resolve().parents[1]
    cell = {
        "data_profile": "data_profiles/smd.yaml",
        "method_profile": "method_profiles/baselines/m2n2.yaml",
        "evaluation_profile": "evaluation_profiles/native_q99_ewma075.yaml",
        "entity": "machine-1-1",
        "phase": "offline",
        "seed": 6,
        "device": "cpu",
        "encoder": {
            "hidden_channels": 64,
            "output_dim": 64,
            "kernel_size": 3,
            "num_layers": 3,
            "dropout": 0.1,
        },
        "limits": {
            "max_epochs_per_stage": 1,
            "max_train_windows": 64,
            "max_validation_windows": 32,
            "max_test_windows": 32,
        },
    }
    config = resolve_run(cell, project_root=project)
    config["evaluation"].update(metrics=["fpr"], uncertainty_fields=[])
    offline = allocate_run(config, project_root=tmp_path)
    run_offline(offline)
    cell.update(
        phase="online",
        offline_parent_dir=str(offline["output_root"]),
        limits={"max_online_steps": 32, "max_validation_windows": 32},
    )
    config = resolve_run(cell, project_root=project)
    config["evaluation"].update(metrics=["fpr"], uncertainty_fields=[])
    return [allocate_run(config, project_root=tmp_path) for _ in range(2)]


def test_real_m2n2_updates_match_after_resume(tmp_path, monkeypatch):
    import signal

    from src.artifacts import load_checkpoint
    from src.methods.m2n2 import M2N2
    from src.online import run_online

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original, interrupted = _real_m2n2_pair(tmp_path)
        update = M2N2.update
        counts = []

        def count_updates(self, *args):
            outcome = update(self, *args)
            counts.append(outcome["did_update"])
            return outcome

        monkeypatch.setattr(M2N2, "update", count_updates)
        complete = run_online(original)
        actual_updates = sum(counts)
        assert actual_updates > 0

        def interrupt_after_window(self, *args):
            outcome = count_updates(self, *args)
            if len(counts) == 10:
                signal.raise_signal(signal.SIGINT)
            return outcome

        counts.clear()
        with monkeypatch.context() as scoped:
            scoped.setattr(M2N2, "update", interrupt_after_window)
            with pytest.raises(KeyboardInterrupt):
                run_online(interrupted)
        saved = load_resume(interrupted)
        assert saved["cursor"] == 10
        resumed = run_online(interrupted, resume=saved)
        assert sum(counts) == actual_updates
        np.testing.assert_array_equal(_scores(original), _scores(interrupted))
        for key in ("network", "trend"):
            states = [
                load_checkpoint(
                    row["final_checkpoint"],
                    expected_identity=config["prepared_identity"],
                )[key]
                for row, config in ((complete, original), (resumed, interrupted))
            ]
            torch.testing.assert_close(*states, rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)


def test_stream_range_limits_metrics_and_label_counts(online):
    module, config, _, split = online
    split["labels"][:3] = np.nan
    split["labels"][9:] = np.nan
    run = _allocate(config | {"limits": {"stream_start": 3, "stream_end": 9}})
    result = module.run_online(run)
    record = json.loads(result["splits"]["test"].read_text())
    assert record["stream_range"] == [3, 9]
    assert record["eligibility"]["total_observations"] == 6
    assert record["eligibility"]["unknown_labels"] == 0
    assert len(_scores(run)) == 6


@pytest.mark.parametrize("previous,current,expected", [(0, 1, 6), (0.25, 0.75, 5)])
def test_online_reuses_the_offline_q95_and_score_reduction(
    monkeypatch, previous, current, expected
):
    from src import online
    from src.scoring import aggregate_points, calibrate

    settings = {
        "score_space": "raw_input",
        "online_point_threshold_source": "offline_parent",
        "offline_threshold_split": "validation",
        "offline_threshold_quantile": 0.95,
        "offline_overlap_reduction": "last_ewma_estimate",
        "ewma_previous_weight": previous,
        "ewma_current_weight": current,
    }
    windows = {
        "source_split": "validation",
        "score_kind": "window",
        "length": 3,
        "raw_scores": np.array([[1.0, 2.0], [6.0, 1.0]]),
        "indices": np.array([[0, 1], [1, 2]]),
        "labels": np.zeros((2, 2)),
    }
    cal = calibrate(windows, settings, identity={"checkpoint_sha256": "fixture"})
    assert cal["point_threshold"] == pytest.approx(np.quantile([1, expected, 1], 0.95))
    parent = {"checkpoint": {"sha256": "fixture"}}
    monkeypatch.setattr(online, "_load_parent", lambda *_: (cal, parent))
    config = {
        "method_name": "mustard",
        "variant": "A0",
        "required_offline_variant": "O0",
        "evaluation": settings,
    }
    actual, _ = online._initialize_online(None, None, config, {})
    assert actual["point_threshold"] == cal["point_threshold"]
    for phase in ("offline", "online"):
        points = aggregate_points(
            iter([windows]), length=3, settings=settings | {"phase": phase}
        )
        assert points["scores"][1] == expected
    for updates, match in [
        ({"score_space": "latent"}, "matching offline threshold"),
        (
            {"ewma_current_weight": 0.5, "ewma_previous_weight": 0.5},
            "different score reduction",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            online._initialize_online(
                None, None, config | {"evaluation": settings | updates}, {}
            )


def test_online_uncertainty_uses_actual_causal_score_records(online, monkeypatch):
    module, config, _, _ = online
    field = "point_anomaly_score_variance"
    config["evaluation"].update(
        uncertainty_fields=[field], uncertainty_statistics=["mean"]
    )
    config["limits"] = {"max_online_steps": 2, "stream_start": 3}
    original = _OnlineMethod.score

    def score(method, batch):
        record = original(method, batch)
        return record | {"uncertainty": {field: batch["indices"]}}

    monkeypatch.setattr(_OnlineMethod, "score", score)
    run = _allocate(config)
    module.run_online(run)
    result = json.loads((run["output_root"] / "test/result.json").read_text())
    assert result["metrics"][field + "_mean"] == 4.0
    assert result["metric_metadata"][field + "_mean"]["value_count"] == 4


def test_progress_counts_only_new_windows_after_resume(online):
    module, config, events, _ = online
    run = _allocate(config | {"limits": {"max_online_steps": 11}})
    events["interrupt"] = 9
    with pytest.raises(KeyboardInterrupt):
        module.run_online(run)
    snapshot = load_resume(run)
    assert snapshot["cursor"] == 8
    events.pop("interrupt")
    updates = []
    module.run_online(
        run, resume=snapshot, progress=lambda **event: updates.append(event)
    )
    assert [
        (event["unit"], event["completed"], event["total"]) for event in updates
    ] == [
        ("window", 8, 11),
        ("window", 9, 11),
        ("window", 10, 11),
        ("window", 11, 11),
        ("window", 11, 11),
    ]
    assert updates[-1]["stats"]["FPR"] is not None

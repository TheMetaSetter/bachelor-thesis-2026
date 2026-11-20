from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from src.artifacts import (
    allocate_run,
    find_runs,
    load_checkpoint,
    load_resume,
    result_directory,
    save_checkpoint,
    save_resume,
    select_offline_parent,
    write_result,
    write_run_status,
)


def config(**overrides):
    return {
        "dataset": "smd", "entity": "machine", "method_name": "mustard",
        "variant": "O0", "phase": "offline", "seed": 6,
        **overrides,
    }


def test_uuidv7_and_repeated_execution(tmp_path):
    run = config(experiment_id="example")
    first = allocate_run(run, project_root=tmp_path)
    second = allocate_run(run, project_root=tmp_path)
    for item in (first, second):
        assert UUID(item["run_id"]).version == 7
        assert (tmp_path / "outputs/run_ids" / f'{item["run_id"]}.json').is_file()
    assert first["run_id"] != second["run_id"]
    assert "run_id" not in run


def test_explicit_id_and_collision(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        allocate_run(config(run_id="../bad"), project_root=tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        allocate_run(config(experiment_id="example", run_id="manual"), project_root=tmp_path)
    standalone = config(run_id="manual")
    allocate_run(standalone, project_root=tmp_path)
    with pytest.raises((FileExistsError, ValueError), match="manual"):
        allocate_run(standalone, project_root=tmp_path)


def test_layout_and_exact_lookup(tmp_path):
    run = allocate_run(config(experiment_id="example", grouping={"method_name": True}), project_root=tmp_path)
    expected = tmp_path / "outputs/experiments/example/offline/mustard/test" / run["run_id"]
    assert result_directory(run, "test") == expected
    write_run_status(run, "running")
    write_result(run, "test", {"metrics": {"fpr": 0.1}})
    assert find_runs(tmp_path / "outputs/experiments", filters={"entity": "machine-test"}) == []
    assert find_runs(tmp_path / "outputs/experiments", filters={"entity": "machine"})[0]["run_id"] == run["run_id"]


def test_latest_parent_with_missing_artifact_never_falls_back(tmp_path):
    for day in (1, 2):
        run = allocate_run(config(experiment_id="example"), project_root=tmp_path)
        run["started_at"] = f"2026-11-0{day}T00:00:00.000000Z"
        state_dir = result_directory(run, "state")
        checkpoint = state_dir / "stage_b.best.pt"
        calibration = state_dir / "calibration.json"
        write_run_status(run, "running")
        checkpoint.write_bytes(b"example")
        calibration.write_text("{}")
        write_run_status(run, "successful", details={"selected_checkpoint": checkpoint, "calibration": calibration})
        if day == 2:
            checkpoint.unlink()
    child = config(experiment_id="example", phase="online", variant="A0", required_offline_variant="O0")
    with pytest.raises(FileNotFoundError, match="stage_b.best"):
        select_offline_parent(child, project_root=tmp_path)


def test_checkpoint_identity_round_trip_and_corruption(tmp_path):
    path = tmp_path / "value.pt"
    state = {"coefficients": np.array([1.1, 4.2], dtype=np.float64)}
    save_checkpoint(path, state, identity={"method": "mustard"}, role="fit.best")
    loaded = load_checkpoint(path, expected_identity={"method": "mustard", "role": "fit.best"})
    np.testing.assert_array_equal(state["coefficients"], loaded["coefficients"])
    assert loaded["coefficients"].dtype == np.float64
    with pytest.raises(ValueError, match="role"):
        load_checkpoint(path, expected_identity={"method": "mustard", "role": "fit.initial"})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(path, expected_identity={"method": "mustard"})


def test_resume_commits_complete_generations(tmp_path):
    run = allocate_run(config(), project_root=tmp_path)
    state = {"config": run, "cursor": 1, "method_state": {}, "rng_state": {}, "scoring_state": {}}
    pointer = save_resume(run, state)
    with pytest.raises(ValueError, match="cursor"):
        save_resume(run, {key: value for key, value in state.items() if key != "cursor"})
    assert load_resume(run)["cursor"] == 1
    save_resume(run, state | {"cursor": 2})
    assert load_resume(run)["cursor"] == 2
    payload = Path(pointer).parent / "resume-1.pt"
    payload.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="hash"):
        load_resume(run)


def test_parent_selection_excludes_newer_incompatible_runtime(tmp_path):
    current = None
    for day, identity in ((1, "same"), (2, "different")):
        run = allocate_run(config(experiment_id="example"), project_root=tmp_path)
        run["started_at"] = f"2026-11-0{day}T00:00:00.000000Z"
        write_run_status(run, "running")
        directory = result_directory(run, "state")
        checkpoint = directory / "stage_b.best.pt"
        calibration = directory / "calibration.json"
        checkpoint.write_bytes(b"example")
        calibration.write_text("{}")
        write_run_status(run, "successful", details={
            "selected_checkpoint": checkpoint, "calibration": calibration, "execution_identity": identity,
        })
        if day == 1:
            current = run["run_id"]
    child = config(experiment_id="example", phase="online", variant="A0",
                   required_offline_variant="O0", execution_identity="same")
    assert select_offline_parent(child, project_root=tmp_path)["run_id"] == current

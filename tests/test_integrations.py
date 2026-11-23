"""Check remote-provider boundaries with fakes."""

import importlib
import json
from types import SimpleNamespace

import pytest

from src.artifacts import allocate_run, write_result, write_run_status


@pytest.fixture
def integrations():
    return importlib.import_module("src.integrations")


def test_disabled_providers_do_not_load_dependencies(
    integrations, monkeypatch, tmp_path
):
    def unavailable(*args):
        pytest.fail("disabled provider was loaded")

    monkeypatch.setattr(integrations, "_wandb", unavailable)
    monkeypatch.setattr(integrations, "_kaggle", unavailable)
    assert integrations.log_wandb({"kind": "start", "enabled": False}) is None
    assert (
        integrations.backup_kaggle(tmp_path, {"kaggle_backup": False})["status"]
        == "disabled"
    )


def test_wandb_uses_native_session_and_sanitizes_failures(integrations, monkeypatch):
    calls = []
    native = SimpleNamespace(
        log=lambda metrics: calls.append(metrics), finish=lambda: calls.append("finish")
    )

    def init(**settings):
        assert (
            settings["id"] == "abc"
            and settings["group"] == "example"
            and settings["resume"] == "allow"
        )
        return native

    monkeypatch.setattr(integrations, "_wandb", lambda: SimpleNamespace(init=init))
    event = {
        "kind": "start",
        "enabled": True,
        "run_id": "abc",
        "experiment_id": "example",
    }
    session = integrations.log_wandb(event)
    assert session is native
    integrations.log_wandb(
        {"kind": "metrics", "enabled": True, "metrics": {"fpr": 0.1}}, session=session
    )
    assert (
        integrations.log_wandb({"kind": "finish", "enabled": True}, session=session)
        is None
    )
    assert calls == [{"fpr": 0.1}, "finish"]

    def fail(**_):
        raise RuntimeError("sensitive provider detail")

    monkeypatch.setattr(integrations, "_wandb", lambda: SimpleNamespace(init=fail))
    with pytest.raises(RuntimeError, match=r"^W&B start failed$"):
        integrations.log_wandb(event)


def test_backup_runs_after_success_failure_and_interrupt(
    integrations, monkeypatch, tmp_path
):
    from src import experiments

    calls = []
    config = {
        "project_root": tmp_path,
        "experiment_id": "sample",
        "phase": "offline",
        "kaggle_backup": True,
    }
    monkeypatch.setattr(experiments, "plan_experiment", lambda *_, **__: [config])

    def backup(root, settings):
        records = [json.loads(path.read_text()) for path in root.rglob("run.json")]
        record = max(records, key=lambda item: item["started_at"])
        calls.append((root.name, record["status"]))
        return {"status": "successful"}

    monkeypatch.setattr(experiments, "backup_kaggle", backup)
    for status in ("successful", "failed", "interrupted"):

        def run(config, *, resume=None):
            write_run_status(config, "running")
            write_run_status(config, status)
            if status == "interrupted":
                raise KeyboardInterrupt
            if status == "failed":
                raise RuntimeError("fixture failure")
            return {"status": "successful", "run_id": config["run_id"]}

        monkeypatch.setattr(experiments, "run_offline", run)
        if status == "interrupted":
            with pytest.raises(KeyboardInterrupt):
                experiments.run_experiment(
                    tmp_path / "sample.yaml", project_root=tmp_path
                )
        else:
            experiments.run_experiment(tmp_path / "sample.yaml", project_root=tmp_path)
    assert calls == [
        ("sample", status) for status in ("successful", "failed", "interrupted")
    ]


def test_backup_only_links_approved_artifacts_once(integrations, monkeypatch, tmp_path):
    config = {"experiment_id": "sample", "phase": "offline", "execution_mode": "normal"}
    run = allocate_run(config, project_root=tmp_path)
    write_run_status(run, "running")
    result = write_result(run, "test", {"metrics": {"fpr": 0.1}})
    write_run_status(run, "successful", details={"splits": {"test": result}})
    root = run["output_root"]
    (root / "kaggle.json").write_text("dummy")
    (root.parent / "unrelated").mkdir()
    (root.parent / "unrelated/sensitive.json").write_text("dummy")
    calls = []

    def upload(folder, **options):
        from pathlib import Path

        folder = Path(folder)
        assert options["dir_mode"] == "skip" and options["convert_to_csv"] is False
        contents = json.loads((folder / "backup-manifest.json").read_text())
        records = contents["files"]
        assert {entry["path"].split("/")[-1] for entry in records} == {
            "result.json",
            "run.json",
        }
        for entry in records:
            assert (folder / entry["file"]).stat().st_ino == (
                root / entry["path"]
            ).stat().st_ino
        calls.append(folder)
        return SimpleNamespace(status="ok", version_number=3)

    client = SimpleNamespace(
        dataset_list=lambda **_: [SimpleNamespace(ref="owner/target", is_private=True)],
        dataset_create_version=upload,
    )
    monkeypatch.setattr(integrations, "_kaggle", lambda: client)
    settings = {
        "kaggle_backup": True,
        "kaggle_dataset": "owner/target",
        "experiment_id": "sample",
    }
    assert integrations.backup_kaggle(root, settings)["remote_version"] == 3
    assert integrations.backup_kaggle(root, settings)["status"] == "already_saved"
    assert len(calls) == 1 and not calls[0].exists()
    with pytest.raises(ValueError, match="experiment root"):
        integrations.backup_kaggle(root.parent, settings)

"""Check command routing and the isolated command installation."""

from hashlib import sha256
import importlib
import json
from pathlib import Path
from subprocess import run

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


def prepared_version(root, name, dataset="smd"):
    path = root / "data/preprocessed" / dataset / name
    entity = path / "machine"
    entity.mkdir(parents=True)
    files = {}
    splits = {}
    for split in ("train", "validation", "test"):
        table = pa.table(
            {"value": [1.0], "label": [0], "source_row": [0], "source_file": ["source"]}
        )
        output = entity / f"{split}.parquet"
        pq.write_table(table, output)
        files[str(output.relative_to(path))] = sha256(output.read_bytes()).hexdigest()
        splits[split] = {"rows": 1}
    scaler = entity / "scaler.json"
    scaler.write_text('{"features": ["value"]}')
    files["machine/scaler.json"] = sha256(scaler.read_bytes()).hexdigest()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "version_id": name,
                "identity": {"profile": {"dataset": dataset}},
                "entities": {"machine": {"features": ["value"], "splits": splits}},
                "files": files,
            }
        )
    )
    return path


@pytest.fixture
def cli(tmp_path):
    module = importlib.import_module("src.cli")
    (tmp_path / "pyproject.toml").touch()
    for category in ("experiments", "data_profiles", "method_profiles"):
        (tmp_path / "configs" / category).mkdir(parents=True)
    version = prepared_version(tmp_path, "first")
    (tmp_path / "configs/data_profiles/smd.yaml").write_text(
        f"dataset: smd\nprepared_root: {version.relative_to(tmp_path)}\nwindow_size: 20\n"
    )
    (tmp_path / "configs/method_profiles/default.yaml").write_text("method: mustard\n")
    return module, tmp_path


def test_lists_edits_and_completes_names(cli, capsys):
    module, root = cli
    command = ["--project-root", str(root)]
    assert module.main(command + ["list", "datasets"]) == 0
    assert capsys.readouterr().out.strip() == "smd"
    assert module.main(command + ["new", "experiment", "--name", "example"]) == 0
    assert (
        module.main(
            command + ["experiment", "--name", "example", "methods", "add", "mustard"]
        )
        == 0
    )
    assert (
        module.main(
            command + ["experiment", "--name", "example", "datasets", "add", "smd"]
        )
        == 0
    )
    assert (
        module.main(
            command
            + ["experiment", "--name", "example", "set", "random_seeds", "1", "2", "3"]
        )
        == 0
    )
    assert module.complete_names("experiments", "exa", project_root=root) == ["example"]
    assert module.complete_names("methods", "mus", project_root=root) == ["mustard"]
    assert module.main(command + ["completion", "zsh"]) == 0
    assert "compdef" in capsys.readouterr().out


def test_show_validate_run_resume_and_report(cli, monkeypatch, capsys):
    module, root = cli
    base = ["--project-root", str(root)]
    path = root / "configs/experiments/example.yaml"
    path.touch()
    calls = []

    def plan(received, *, project_root, smoke=False):
        assert received == path and project_root == root
        calls.append("plan")
        return [{"entity": "machine", "profile_paths": {"method": "profile.yaml"}}]

    def execute(received, *, project_root, smoke=False, progress_factory):
        assert progress_factory is module.run_progress
        calls.append(("run", smoke))
        return [{"run_id": "abc", "status": "failed" if smoke else "successful"}]

    monkeypatch.setattr(module, "plan_experiment", plan)
    monkeypatch.setattr(module, "run_experiment", execute)
    monkeypatch.setattr(
        module, "resume_experiment", lambda *_, **__: [{"status": "successful"}]
    )
    monkeypatch.setattr(
        module, "report_experiment", lambda *_, **__: [root / "report.csv"]
    )
    for verb in ("show", "validate", "run", "resume", "report"):
        command = (
            ["experiment", "--name", "example", verb]
            if verb == "show"
            else [verb, "experiment", "--name", "example"]
        )
        assert module.main(base + command) == 0
    assert (
        module.main(base + ["run", "experiment", "--name", "example", "--smoke"]) != 0
    )
    assert calls == ["plan", "plan", ("run", False), ("run", True)]
    assert "report.csv" in capsys.readouterr().out


def test_cli_returns_an_error_for_missing_scope(cli, capsys):
    module, root = cli
    assert (
        module.main(
            ["--project-root", str(root), "validate", "experiment", "--name", "absent"]
        )
        != 0
    )
    assert "absent" in capsys.readouterr().err


def test_preprocessing_forwards_all_arguments(cli, monkeypatch):
    module, root = cli
    arguments = [
        "--dataset",
        "smd",
        "--profile",
        "example.yaml",
        "--input-root",
        "input",
        "--output-root",
        "output",
        "--entities",
        "first",
        "second",
        "--show-profile",
    ]
    passed = []
    monkeypatch.setattr(
        module, "_preprocess_main", lambda args: passed.append(args) or 0
    )
    assert module.main(["--project-root", str(root), "preprocess", *arguments]) == 0
    assert passed == [arguments]


def test_installed_entrypoint_from_another_directory(cli):
    _, root = cli
    checkout = Path(__file__).resolve().parents[1]
    command = checkout / ".venv/bin/tsad"
    result = run(
        [command, "--project-root", root, "list", "datasets"],
        cwd=root.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "smd"


def _add_datasets(cli, *names):
    module, root = cli
    base = ["--project-root", str(root)]
    path = root / "configs/experiments/example.yaml"
    if not path.exists():
        assert module.main(base + ["new", "experiment", "--name", "example"]) == 0
    return module.main(
        base + ["experiment", "--name", "example", "datasets", "add", *names]
    )


def test_save_updates_shared_profile_and_repeated_add_is_safe(cli, monkeypatch):
    module, root = cli
    second = prepared_version(root, "second")
    monkeypatch.setattr(
        module, "_choose_versions", lambda choices: {"smd": second}, raising=False
    )
    profile = root / "configs/data_profiles/smd.yaml"
    profile.chmod(0o664)
    assert _add_datasets(cli, "smd", "smd") == 0
    assert profile.stat().st_mode & 0o777 == 0o664
    definition = root / "configs/experiments/example.yaml"
    assert yaml.safe_load(profile.read_text()) == {
        "dataset": "smd",
        "prepared_root": str(second.relative_to(root)),
        "window_size": 20,
    }
    assert yaml.safe_load(definition.read_text())["datasets"] == ["smd"]
    before = (profile.stat().st_mtime_ns, definition.stat().st_mtime_ns)
    assert _add_datasets(cli, "smd") == 0
    assert (profile.stat().st_mtime_ns, definition.stat().st_mtime_ns) == before


@pytest.mark.parametrize("case", ["cancel", "missing", "corrupt"])
def test_add_without_save_keeps_files(cli, monkeypatch, case):
    module, root = cli
    profile = root / "configs/data_profiles/smd.yaml"
    version = root / "data/preprocessed/smd/first"
    before = profile.read_bytes()
    if case == "cancel":
        monkeypatch.setattr(
            module, "_choose_versions", lambda choices: None, raising=False
        )
    elif case == "missing":
        (version / "manifest.json").unlink()
    else:
        (version / "machine/test.parquet").write_bytes(b"corrupt")
    assert _add_datasets(cli, "smd") == (0 if case == "cancel" else 1)
    assert profile.read_bytes() == before
    assert (
        yaml.safe_load((root / "configs/experiments/example.yaml").read_text())[
            "datasets"
        ]
        == []
    )


def test_save_restores_profile_if_experiment_write_fails(cli, monkeypatch):
    from src import dataset_selection

    module, root = cli
    second = prepared_version(root, "second")
    monkeypatch.setattr(
        module, "_choose_versions", lambda choices: {"smd": second}, raising=False
    )
    write = dataset_selection._write_atomic

    def fail_on_experiment(path, writer):
        if path.parent.name == "experiments":
            raise OSError("test write failure")
        return write(path, writer)

    monkeypatch.setattr(dataset_selection, "_write_atomic", fail_on_experiment)
    profile = root / "configs/data_profiles/smd.yaml"
    profile.chmod(0o664)
    before = profile.read_bytes()
    assert _add_datasets(cli, "smd") != 0
    assert profile.read_bytes() == before
    assert profile.stat().st_mode & 0o777 == 0o664
    assert (
        yaml.safe_load((root / "configs/experiments/example.yaml").read_text())[
            "datasets"
        ]
        == []
    )


def test_bare_cli_launches_only_with_terminal(cli, monkeypatch):
    from src import tui

    module, root = cli
    calls = []
    monkeypatch.chdir(root)
    monkeypatch.setattr(module.sys, "argv", ["tsad"])
    monkeypatch.setattr(
        tui, "launch_tui", lambda directory: calls.append(directory) or 7
    )
    for input_terminal, output_terminal in ((True, False), (False, True), (True, True)):
        monkeypatch.setattr(module.sys.stdin, "isatty", lambda: input_terminal)
        monkeypatch.setattr(module.sys.stdout, "isatty", lambda: output_terminal)
        with pytest.raises(SystemExit, match="2"):
            module.main([])
        if input_terminal and output_terminal:
            assert module.main() == 7
        else:
            with pytest.raises(SystemExit, match="2"):
                module.main()
    assert calls == [root.resolve()]


@pytest.mark.parametrize("mode", ["normal", "smoke"])
@pytest.mark.parametrize("outcome", ["files", "empty", "error"])
def test_report_cli_mode_and_status(cli, monkeypatch, capsys, mode, outcome):
    module, root = cli
    paths = [root / name for name in ("test_detail.csv", "test_mean.csv", "test.html")]

    def report(path, *, project_root, smoke):
        assert path == root / "configs/experiments/example.yaml"
        assert project_root == root
        assert smoke == (mode == "smoke")
        if outcome == "error":
            raise ValueError("result identity mismatch")
        return paths if outcome == "files" else []

    monkeypatch.setattr(module, "report_experiment", report)
    command = ["--project-root", str(root), "report", "experiment", "--name", "example"]
    if mode == "smoke":
        command.append("--smoke")
    status = module.main(command)
    output = capsys.readouterr()
    if outcome == "error":
        assert status == 1
        assert "result identity mismatch" in output.err
    elif outcome == "empty":
        assert status == 0
        assert output.out.strip() == f"No completed runs to report in {mode} mode."
    else:
        assert status == 0
        assert output.out.splitlines() == list(map(str, paths))


@pytest.mark.parametrize("field", ["wandb_logging", "kaggle_backup"])
def test_set_remote_flag(cli, field):
    module, root = cli
    command = ["--project-root", str(root)]
    assert module.main(command + ["new", "experiment", "--name", "example"]) == 0
    path = root / "configs/experiments/example.yaml"
    for value, expected in (("true", True), ("false", False)):
        assert (
            module.main(
                command + ["experiment", "--name", "example", "set", field, value]
            )
            == 0
        )
        assert yaml.safe_load(path.read_text())[field] is expected


def test_dotenv_loads_missing_values_and_keeps_shell_precedence(
    cli, monkeypatch
):
    import os

    module, root = cli
    monkeypatch.delenv("TSAD_TEST_NEW_VAR", raising=False)
    monkeypatch.delenv("TSAD_TEST_QUOTED", raising=False)
    monkeypatch.setenv("TSAD_TEST_EXISTING", "from-shell")
    (root / ".env").write_text(
        "\n# comment line\n"
        "TSAD_TEST_NEW_VAR=from-dotenv\n"
        "export TSAD_TEST_QUOTED='quoted-value'\n"
        "TSAD_TEST_EXISTING=overridden\n",
        encoding="utf-8",
    )
    assert module.main(["--project-root", str(root), "list", "experiments"]) == 0
    assert os.environ["TSAD_TEST_NEW_VAR"] == "from-dotenv"
    assert os.environ["TSAD_TEST_QUOTED"] == "quoted-value"
    assert os.environ["TSAD_TEST_EXISTING"] == "from-shell"


def test_missing_dotenv_does_not_fail(cli):
    module, root = cli
    assert not (root / ".env").exists()
    assert module.main(["--project-root", str(root), "list", "experiments"]) == 0

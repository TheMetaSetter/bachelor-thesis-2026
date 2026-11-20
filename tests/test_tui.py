"""Check the slash command interface and its CLI processes."""

from shlex import split

import pytest


@pytest.mark.parametrize(
    ("command", "experiment", "expected"),
    [
        *[(f"/list {kind}", None, ("list", kind))
          for kind in ("experiments", "datasets", "methods", "standalones")],
        ("/new trial", None, ("new", "experiment", "--name", "trial")),
        *[(f"/{operation}", "demo", ("experiment", "--name", "demo", *operation.split()))
          for operation in (
              "show", "datasets add smd swat", "datasets remove swat",
              "methods add mustard candi m2n2", "methods remove candi",
              "set random_seeds 1 2", "set wandb_logging true", "set kaggle_backup false",
          )],
        *[(f"/{verb}", "demo", (verb, "experiment", "--name", "demo"))
          for verb in ("validate", "run", "resume", "report")],
        ("/run --smoke", "demo", ("run", "experiment", "--name", "demo", "--smoke")),
        ("/report --smoke", "demo", ("report", "experiment", "--name", "demo", "--smoke")),
        ("/run standalone trial", None, ("run", "standalone", "--name", "trial")),
        ('/preprocess --dataset smd --input-root "raw data"', None,
         ("preprocess", "--dataset", "smd", "--input-root", "raw data")),
        ("/completion zsh", None, ("completion", "zsh")),
        ("/_complete methods mus", None, ("_complete", "methods", "mus")),
    ],
)
def test_resolve_cli_arguments(command, experiment, expected):
    from src.tui import resolve_cli_arguments

    assert resolve_cli_arguments(split(command), experiment_name=experiment) == expected


@pytest.mark.parametrize("command", ["/unknown", "/show", "/new", "/run standalone"])
def test_resolve_cli_arguments_rejects_invalid_input(command):
    from src.tui import resolve_cli_arguments

    with pytest.raises(ValueError):
        resolve_cli_arguments(split(command), experiment_name=None)


def test_build_cli_process_arguments(tmp_path):
    import sys

    from src.tui import build_cli_process_arguments

    root = tmp_path / "project with spaces"
    assert build_cli_process_arguments(("list", "datasets"), project_root=root) == (
        sys.executable, "-u", "-m", "thesis_tsad.cli", "--project-root",
        str(root), "list", "datasets",
    )


async def wait_until(condition):
    import asyncio

    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.01)


def child_program(monkeypatch, program):
    import sys

    from src import tui

    monkeypatch.setattr(tui, "build_cli_process_arguments", lambda *_args, **_kwargs: (
        sys.executable, "-u", "-c", program,
    ))


def test_command_screen_streams_output(tmp_path, monkeypatch):
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Log

    from src.tui import CommandScreen

    child_program(monkeypatch, """
import pathlib, sys, time
print("standard output")
print("standard error", file=sys.stderr)
while not pathlib.Path("complete").is_file():
    time.sleep(0.01)
sys.exit(2)
""")

    async def check():
        app, results = App(), []
        screen = CommandScreen(("list", "datasets"), project_root=tmp_path)
        async with app.run_test() as pilot:
            await app.push_screen(screen, results.append)
            try:
                await wait_until(lambda: "standard error" in screen.query_one(Log).lines)
                assert "standard output" in screen.query_one(Log).lines
                assert screen.query_one(Button).disabled
                await pilot.press("esc")
                assert app.screen is screen
            finally:
                (tmp_path / "complete").touch()
            await wait_until(lambda: not screen.query_one(Button).disabled)
            await pilot.click("#back")
            assert results == [2]

    asyncio.run(check())


def test_command_screen_interrupt(tmp_path, monkeypatch):
    import asyncio

    from textual.app import App
    from textual.widgets import Button, Log

    from src.tui import CommandScreen

    child_program(monkeypatch, """
import signal, sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
print("ready")
signal.pause()
""")

    async def check():
        app, results = App(), []
        screen = CommandScreen(("run",), project_root=tmp_path)
        async with app.run_test() as pilot:
            await app.push_screen(screen, results.append)
            await wait_until(lambda: "ready" in screen.query_one(Log).lines)
            await pilot.press("ctrl+c")
            await wait_until(lambda: not screen.query_one(Button).disabled)
            await pilot.click("#back")
            assert results == [130]
            assert app.is_running

    asyncio.run(check())


@pytest.mark.parametrize("picker", [False, True])
def test_command_screen_recovers_from_start_failure(tmp_path, monkeypatch, picker):
    import asyncio
    from contextlib import contextmanager

    from textual.app import App, SuspendNotSupported
    from textual.widgets import Button, Log

    from src import tui

    @contextmanager
    def suspend():
        raise SuspendNotSupported("Terminal is unavailable.")
        yield

    def build(*_args, **_kwargs):
        if picker:
            pytest.fail("The picker started without terminal suspension.")
        return (str(tmp_path / "absent-python"),)

    monkeypatch.setattr(tui, "build_cli_process_arguments", build)

    async def check():
        app, results = App(), []
        monkeypatch.setattr(app, "suspend", suspend)
        arguments = ("experiment", "--name", "demo", "datasets", "add", "smd") if picker else ("list", "datasets")
        screen = tui.CommandScreen(arguments, project_root=tmp_path)
        async with app.run_test() as pilot:
            await app.push_screen(screen, results.append)
            await wait_until(lambda: not screen.query_one(Button).disabled)
            assert any("Could not start" in line for line in screen.query_one(Log).lines)
            await pilot.click("#back")
            assert results == [1]

    asyncio.run(check())


async def submit(pilot, command):
    from textual.widgets import Input

    text_bar = pilot.app.query_one(Input)
    text_bar.value = command
    await pilot.press("enter")


def test_session_switch_restores_history(tmp_path, monkeypatch):
    import asyncio

    from textual.widgets import Button, Input, Label, Log

    from src.tui import CommandScreen, TsadApp

    experiments = tmp_path / "configs/experiments"
    experiments.mkdir(parents=True)
    for name in ("first", "second"):
        (experiments / f"{name}.yaml").touch()
    child_program(monkeypatch, "print('list output')")
    app = TsadApp(tmp_path)

    async def check():
        async with app.run_test() as pilot:
            assert str(app.query_one("#session-title", Label).content) == "TSAD project"
            await submit(pilot, "/experiment first")
            await submit(pilot, "/list datasets")
            screen = app.screen
            assert isinstance(screen, CommandScreen)
            await wait_until(lambda: not screen.query_one(Button).disabled)
            await pilot.click("#back")
            assert app.focused is app.query_one(Input)
            assert "/list datasets" in "\n".join(app.query_one(Log).lines)
            assert "Exit status 0." in "\n".join(app.query_one(Log).lines)
            await submit(pilot, "/experiment second")
            assert "second" in str(app.query_one("#session-title", Label).content)
            assert "/list datasets" not in "\n".join(app.query_one(Log).lines)
            await submit(pilot, "/experiment first")
            assert "/list datasets" in "\n".join(app.query_one(Log).lines)

    asyncio.run(check())


def test_session_input_errors(tmp_path):
    import asyncio

    from textual.widgets import Log

    from src.tui import TsadApp

    app = TsadApp(tmp_path)

    async def check():
        async with app.run_test() as pilot:
            cases = (
                ("/experiment missing", "Experiment not found"),
                ("/unknown", "Unknown slash command"),
                ("/show", "Select an experiment"),
                ("/report", "Select an experiment"),
                ('/preprocess --input-root "raw data', "quotation"),
            )
            for command, message in cases:
                await submit(pilot, command)
                assert len(app.screen_stack) == 1
                assert message in "\n".join(app.query_one(Log).lines)

    asyncio.run(check())


def test_slash_help_and_quit(tmp_path):
    import asyncio

    from textual.widgets import Log

    from src.tui import TsadApp

    app = TsadApp(tmp_path)

    async def check():
        async with app.run_test() as pilot:
            await submit(pilot, "/help")
            help_text = "\n".join(app.query_one(Log).lines)
            for command in (
                "/experiment", "/list", "/new", "/show", "/datasets", "/methods", "/set",
                "/validate", "/run", "/resume", "/report", "/preprocess", "/completion", "/help", "/quit",
            ):
                assert command in help_text
            assert "/report [--smoke]" in help_text
            assert "/_complete" not in help_text
            await submit(pilot, "/quit")
        assert app.return_value == 0

    asyncio.run(check())


@pytest.mark.parametrize("command", [
    "/run --name second", "/resume --na=second", "/validate --n second",
    "/report --nam second", "/new first --name second", "/run standalone first --name second",
])
def test_slash_commands_reject_name_override(command):
    from src.tui import resolve_cli_arguments

    with pytest.raises(ValueError, match="name"):
        resolve_cli_arguments(split(command), experiment_name="first")


@pytest.mark.parametrize("prompt_error", [EOFError, KeyboardInterrupt])
def test_terminal_prompt_keeps_child_status(tmp_path, monkeypatch, prompt_error):
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace

    from textual.app import App
    from textual.widgets import Button

    from src import tui

    def dismiss_prompt(_prompt):
        raise prompt_error

    monkeypatch.setattr("builtins.input", dismiss_prompt)
    monkeypatch.setattr(tui.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=2))

    async def check():
        app, results = App(), []
        monkeypatch.setattr(app, "suspend", nullcontext)
        arguments = ("experiment", "--name", "demo", "datasets", "add", "smd")
        screen = tui.CommandScreen(arguments, project_root=tmp_path)
        async with app.run_test() as pilot:
            await app.push_screen(screen, results.append)
            await wait_until(lambda: not screen.query_one(Button).disabled)
            await pilot.click("#back")
            assert results == [2]

    asyncio.run(check())

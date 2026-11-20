"""Use slash commands to work on an experiment in a terminal."""

import asyncio
from codecs import getincrementaldecoder
from collections.abc import Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
from shlex import join, split
import signal
import subprocess
import sys

from textual.app import App, ComposeResult, SuspendNotSupported
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Log

from .cli import complete_names


PROJECT_COMMANDS = ("/list", "/preprocess", "/completion", "/_complete")
EDIT_COMMANDS = ("/show", "/datasets", "/methods", "/set")
EXPERIMENT_COMMANDS = ("/validate", "/run", "/resume", "/report")


def resolve_cli_arguments(
    words: Sequence[str], *, experiment_name: str | None
) -> tuple[str, ...]:
    if not words:
        raise ValueError("Enter a slash command. Use /help to see the commands.")
    command, *arguments = words
    verb = command.removeprefix("/")
    if command in PROJECT_COMMANDS:
        return (verb, *arguments)
    if command in EXPERIMENT_COMMANDS + ("/new",):
        # argparse also accepts abbreviated options such as --na.
        if any(value.partition("=")[0] in ("--n", "--na", "--nam", "--name") for value in arguments):
            raise ValueError("Enter the name in the slash command. Use /help for the syntax.")
    if command == "/new":
        if not arguments or arguments[0].startswith("-"):
            raise ValueError("Enter /new NAME.")
        name, *rest = arguments
        return ("new", "experiment", "--name", name, *rest)
    if command == "/run" and arguments[:1] == ["standalone"]:
        if len(arguments) < 2 or arguments[1].startswith("-"):
            raise ValueError("Enter /run standalone NAME.")
        _, name, *rest = arguments
        return ("run", "standalone", "--name", name, *rest)
    if command not in EDIT_COMMANDS + EXPERIMENT_COMMANDS:
        raise ValueError("Unknown slash command. Use /help to see the commands.")
    if experiment_name is None:
        raise ValueError("Select an experiment with /experiment NAME.")
    if command in EDIT_COMMANDS:
        return ("experiment", "--name", experiment_name, verb, *arguments)
    return (verb, "experiment", "--name", experiment_name, *arguments)


def build_cli_process_arguments(
    arguments: Sequence[str], *, project_root: Path
) -> tuple[str, ...]:
    return (
        sys.executable, "-u", "-m", "thesis_tsad.cli",
        "--project-root", str(project_root), *arguments,
    )


class CommandScreen(ModalScreen[int]):
    CSS = """
    CommandScreen { padding: 1 2; }
    CommandScreen Log { height: 1fr; border: round $surface-lighten-2; }
    """
    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, arguments: Sequence[str], *, project_root: Path) -> None:
        super().__init__()
        self.arguments = tuple(arguments)
        self.project_root = project_root
        self.process: asyncio.subprocess.Process | None = None
        self.exit_status: int | None = None

    def compose(self) -> ComposeResult:
        yield Label(join(("tsad", *self.arguments)), markup=False)
        yield Log(highlight=False, max_lines=2000)
        yield Label("Command in progress. Use Ctrl+C to interrupt.", id="command-status")
        yield Button("Back", id="back", disabled=True)

    def on_mount(self) -> None:
        self.call_after_refresh(self.run_worker, self.run_command)

    async def run_with_output(self) -> int:
        command = build_cli_process_arguments(self.arguments, project_root=self.project_root)
        process = await asyncio.create_subprocess_exec(
            *command, cwd=self.project_root, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        self.process = process
        log = self.query_one(Log)
        decoder = getincrementaldecoder("utf-8")(errors="replace")
        assert process.stdout is not None
        try:
            while chunk := await process.stdout.read(4096):
                log.write(decoder.decode(chunk))
            log.write(decoder.decode(b"", final=True))
            return await process.wait()
        finally:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                await process.wait()
            self.process = None

    def action_interrupt(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    def run_with_terminal(self) -> int:
        with self.app.suspend():
            try:
                command = build_cli_process_arguments(self.arguments, project_root=self.project_root)
                environment = dict(os.environ, TEXTUAL_ALLOW_SIGNALS="1")
                # The child receives Ctrl+C from the terminal. Keep the parent open.
                previous_handler = signal.signal(signal.SIGINT, lambda *_: None)
                try:
                    status = subprocess.run(command, cwd=self.project_root, env=environment).returncode
                finally:
                    signal.signal(signal.SIGINT, previous_handler)
            except KeyboardInterrupt:
                return 130
            except OSError as error:
                # Exit the suspension context normally so Textual restores the terminal.
                self.query_one(Log).write_line(f"Could not start the command. {error}")
                return 1
            try:
                input("Press Enter to return to TSAD.")
            except (KeyboardInterrupt, EOFError, OSError):
                pass
        return status

    async def run_command(self) -> None:
        try:
            if self.arguments[:2] == ("experiment", "--name") and self.arguments[3:5] == ("datasets", "add"):
                status = self.run_with_terminal()
            else:
                status = await self.run_with_output()
        except (OSError, SuspendNotSupported) as error:
            self.query_one(Log).write_line(f"Could not start the command. {error}")
            status = 1
        self.exit_status = 130 if status == -signal.SIGINT else status
        self.query_one("#command-status", Label).update(f"Exit status {self.exit_status}.")
        back = self.query_one("#back", Button)
        back.disabled = False
        back.focus()

    def action_back(self) -> None:
        if self.exit_status is not None:
            self.dismiss(self.exit_status)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()


HELP_TEXT = """/experiment NAME — Open an experiment session
/list KIND — List experiments, datasets, methods, or standalones
/new NAME — Create an experiment
/show — Show the experiment plan
/datasets add|remove NAME ... — Edit the datasets
/methods add|remove NAME ... — Edit the methods
/set FIELD VALUE ... — Set experiment values
/validate — Check the experiment
/run [--smoke] — Run the experiment
/run standalone NAME — Start a standalone run
/resume — Resume the experiment
/report [--smoke] — Make the experiment report
/preprocess OPTIONS — Prepare source data
/completion SHELL — Show the shell completion script
/help — Show these commands
/quit — Close TSAD"""


@dataclass
class ExperimentSession:
    name: str
    history: list[str] = field(default_factory=list)


class TsadApp(App[int]):
    TITLE = "TSAD"
    CSS = """
    TsadApp > Screen { padding: 1 2; }
    #session-title { text-style: bold; color: $accent; }
    #session-history { height: 1fr; border: round $surface-lighten-2; }
    """

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.sessions: dict[str, ExperimentSession] = {}
        self.active_session: ExperimentSession | None = None
        self.project_history: list[str] = []
        self.pending_command: tuple[list[str], str] | None = None

    def compose(self) -> ComposeResult:
        yield Label("TSAD project", id="session-title", markup=False)
        yield Log(id="session-history", highlight=False)
        yield Input(placeholder="/list experiments    /help", id="text-bar")

    def on_mount(self) -> None:
        self.query_one("#text-bar", Input).focus()

    def open_experiment(self, name: str) -> None:
        if name not in complete_names("experiments", name, project_root=self.project_root):
            raise ValueError(f"Experiment not found. {name}")
        if name not in self.sessions:
            self.sessions[name] = ExperimentSession(name)
        self.active_session = self.sessions[name]
        self.query_one("#session-title", Label).update(f"Experiment: {name}")
        log = self.query_one("#session-history", Log)
        log.clear()
        for entry in self.active_session.history:
            log.write_lines(entry.splitlines())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        try:
            words = split(text)
        except ValueError:
            self.write_history("Check the quotation marks in the command.")
            return
        command, *values = words
        try:
            if command == "/experiment":
                if len(values) != 1:
                    raise ValueError("Enter /experiment NAME.")
                self.open_experiment(values[0])
            elif command == "/help":
                self.write_history(HELP_TEXT)
            elif command == "/quit":
                self.exit(0)
            else:
                name = self.active_session.name if self.active_session else None
                arguments = resolve_cli_arguments(words, experiment_name=name)
                self.pending_command = (self.history, text)
                self.push_screen(
                    CommandScreen(arguments, project_root=self.project_root),
                    self.record_command_result,
                )
        except (OSError, ValueError) as error:
            self.write_history(str(error))

    @property
    def history(self) -> list[str]:
        return self.active_session.history if self.active_session else self.project_history

    def write_history(self, text: str) -> None:
        self.history.append(text)
        self.query_one("#session-history", Log).write_lines(text.splitlines())

    def record_command_result(self, exit_status: int | None) -> None:
        if self.pending_command is not None:
            history, command = self.pending_command
            status = "No exit status." if exit_status is None else f"Exit status {exit_status}."
            entry = f"{command}  {status}"
            history.append(entry)
            if history is self.history:
                self.query_one("#session-history", Log).write_line(entry)
            self.pending_command = None
        self.query_one("#text-bar", Input).focus()


def launch_tui(project_root: Path) -> int:
    return TsadApp(project_root).run() or 0

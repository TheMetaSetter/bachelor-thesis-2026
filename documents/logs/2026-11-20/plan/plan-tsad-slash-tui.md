# TSAD slash-command TUI implementation plan

Date 2026-11-20. The user approved the [TUI design](../detail/tsad-slash-tui-design.md) in chat. This checklist is for future execution.

Use Python 3.12 or later, the installed Textual 8.2.8 dependency, and pytest. Complete each stage in order. Record the result of each check before you start the next phase. For agent work, use the executing-plans skill for inline execution or the subagent-driven-development skill if the user selects that approach.

## Change boundary

| File | Change |
| --- | --- |
| `src/tui.py` | Add slash routing, experiment sessions, and one reusable command screen. Use the signatures in the approved design. |
| `src/cli.py` | Add bare interactive launch and a module entry point. Keep the installed `tsad` entry point and command handlers. |
| `tests/test_tui.py` | Check routing, sessions, command screens, and process behavior. |
| `tests/test_cli.py` | Add entry-point tests. Reuse the disposable project and prepared-data fixtures. |
| `README.md` | Add TUI start, session, and picker instructions. |

Use the existing CLI for project operations. The TUI keeps its session history in memory. Keep the existing data profiles, run schemas, picker, and source identity checks. Use ASD-STE100 for text in the app and documents. Follow the project limits of 50 lines per function or method and 500 lines per code file.

## Checks that need special attention

| Condition | Expected result | Phase |
| --- | --- | --- |
| The project path contains spaces. | The child receives one path argument. | 3 |
| Output arrives before the child exits. | The command screen shows the output during execution. | 3 |
| The selected experiment changes. | Each experiment keeps its own history. | 4 |
| Input or output is redirected. | Bare `tsad` uses the existing parser. | 5 |
| The user cancels the picker or interrupts a command. | The TUI returns control. Cancel preserves the YAML files. | 6 |

## Phase 1 — Record the starting state

### Stage 1. Read the requirements

1. [ ] Read the approved design and its slash command table. Tool: Markdown.
2. [ ] Read `python_semantics.md`. Tool: Markdown.
3. [ ] Record the current Git status in the thesis checkout. Tool: `git status --short`.
4. [ ] Record the baseline diff for the files in the change boundary. Include copies of applicable untracked files in a temporary directory. Tools: Git and Python.

### Stage 2. Check the existing commands

1. [ ] Run `.venv/bin/python -m pytest -q tests/test_cli.py tests/test_dataset_picker.py`. Expect the existing CLI and picker tests to pass. Tool: pytest.
2. [ ] Run `.venv/bin/python -m pytest -q`. Record the full-suite result and any existing failures. Tool: pytest.
3. [ ] Check that `pyproject.toml` maps `thesis_tsad` to `src` and includes Textual 8.2.8. Tool: TOML.
4. [ ] Check that `datasets add` selects the picker only when both terminal streams are interactive. Tool: `src/cli.py`.

## Phase 2 — Map slash commands to CLI arguments

### Stage 1. Specify the command mapping

1. [ ] Add `test_resolve_cli_arguments` in `tests/test_tui.py`. Use one case per CLI action in the approved table, including all list kinds, the shell helper, and standalone runs. Assert exact argument tuples. Tool: pytest.
2. [ ] Add `test_resolve_cli_arguments_rejects_invalid_input`. Check an unknown slash command, a required experiment with no selection, and a missing name in `/new` and `/run standalone`. Tool: pytest.
3. [ ] Run `.venv/bin/python -m pytest -q tests/test_tui.py -k resolve_cli_arguments`. Expect a failure until the resolver exists. Tool: pytest.

### Stage 2. Add the resolver

1. [ ] Implement `resolve_cli_arguments(words: Sequence[str], *, experiment_name: str | None) -> tuple[str, ...]` in `src/tui.py`. Use the approved table. Add `--name` where the CLI requires it. Pass the remaining CLI arguments in order. Tool: Python.
2. [ ] Run the resolver tests again. Expect all cases to pass. Tool: pytest.

The resolver maps syntax. Keep checks for dataset names, method names, profiles, and existing CLI flags in the current CLI. For example, resolve `/datasets add smd swat` in session `demo` to `("experiment", "--name", "demo", "datasets", "add", "smd", "swat")`.

## Phase 3 — Add the reusable command screen

### Stage 1. Specify process behavior

1. [ ] Add `test_build_cli_process_arguments`. Assert the current Python interpreter, `-u -m thesis_tsad.cli`, `--project-root`, and the exact argument list. Use a path with spaces. Tool: pytest.
2. [ ] Add `test_command_screen_streams_output`. Use a small child process that writes to both output streams before it exits with status 2. Assert that the log receives both streams before exit, **Back** is blocked during execution, and the screen returns status 2. Tool: pytest and Textual Pilot.
3. [ ] Add `test_command_screen_interrupt`. Use a child process with a SIGINT handler. Assert that Ctrl+C reaches the child, the screen shows its exit status, and the app remains open. Tool: pytest and Textual Pilot.
4. [ ] Add `test_command_screen_recovers_from_start_failure`. Assert that a process or terminal start error shows an error and enables **Back**. Assert that the command does not start without a terminal if suspension fails. Tool: pytest.
5. [ ] Run `.venv/bin/python -m pytest -q tests/test_tui.py -k 'command_screen or build_cli_process'`. Expect a failure until the process builder and screen exist. Tool: pytest.

### Stage 2. Implement the process and screen

1. [ ] Implement `build_cli_process_arguments(arguments: Sequence[str], *, project_root: Path) -> tuple[str, ...]`. Build `python -u -m thesis_tsad.cli --project-root PROJECT_ROOT ...` with `sys.executable`. Pass each argument directly to the process API. Tool: Python.
2. [ ] Implement `CommandScreen` with the constructor and method signatures in the approved design. Show the command, one scrollable log, its status, and **Back**. Start one command after Textual displays the screen. Tool: Textual.
3. [ ] Implement `async run_with_output(self) -> int`. Run the child in the project directory. Merge standard error into standard output. Read and display the output as it arrives. Use a bounded log. Tool: `asyncio` and Textual.
4. [ ] Implement `action_interrupt(self) -> None`. Send SIGINT to the active child on macOS and Linux. Wait for the child to exit and read its remaining output. Show CLI status 130 for an interrupt if the child ends by SIGINT. Tool: Python process API.
5. [ ] Implement `run_with_terminal(self) -> int`. Use `App.suspend()` for the resolved `experiment --name NAME datasets add ...` command. Give the child the terminal streams. Show the CLI output until the user presses Enter. Restore Textual on exit, error, or interruption. Tool: Textual and `subprocess`.
6. [ ] Implement `async run_command(self) -> None` to select the terminal mode, show the exit status or error, and enable **Back** after the child stops. Tool: Python and Textual.
7. [ ] Implement `action_back(self) -> None` to dismiss the finished screen with its status. Keep **Back** disabled during execution. Tool: Textual.
8. [ ] Run the screen and process tests again. Expect all cases to pass. Tool: pytest.

Use the actual CLI exit status. When the existing picker returns status 0, the terminal message gives the Save or Cancel result.

## Phase 4 — Add experiment sessions

### Stage 1. Specify the session behavior

1. [ ] Add `test_session_switch_restores_history`. Open two experiment sessions in a disposable project. Run a stub command in the first session. Assert that the history follows the selected experiment, that the command status appears after **Back**, and that the text bar has focus. Tool: pytest and Textual Pilot.
2. [ ] Add `test_session_input_errors`. Submit an unknown experiment, an unknown slash command, a command that requires a session, and an incomplete quote. Assert that the app displays an error and starts no command. Tool: pytest and Textual Pilot.
3. [ ] Add `test_slash_help_and_quit`. Assert that **Help** lists the public slash commands, hides `/_complete`, and that `/quit` exits. Tool: pytest and Textual Pilot.
4. [ ] Run `.venv/bin/python -m pytest -q tests/test_tui.py -k 'session or slash_help'`. Expect a failure until the app exists. Tool: pytest.

### Stage 2. Implement the session view

1. [ ] Add `ExperimentSession` with `name: str` and a separate `history: list[str]` for each instance. Tool: Python.
2. [ ] Implement `TsadApp` with the signatures in the approved design. Start with no selected experiment. Show the current project or experiment, its history, and one focused `Input`. Tool: Textual.
3. [ ] Implement `open_experiment(self, name: str) -> None`. Check the exact name with `complete_names("experiments", name, project_root=...)`. Load or create the in-memory session for that name. Tool: Python and the existing CLI.
4. [ ] Implement `on_input_submitted(self, event: Input.Submitted) -> None`. Split with `shlex.split`. Handle `/experiment`, `/help`, and `/quit` in the app. Route other slash commands to `CommandScreen`. Show local input errors in the session view. Tool: Python and Textual.
5. [ ] Implement `record_command_result(self, exit_status: int | None) -> None`. Record the command and the reported status in the session that started it. Restore focus to the text bar. Tool: Textual.
6. [ ] Implement `launch_tui(project_root: Path) -> int`. Start `TsadApp` and return its integer exit status. Use status 0 for a normal close. Tool: Textual.
7. [ ] Run `.venv/bin/python -m pytest -q tests/test_tui.py`. Expect all routing, command screen, and session tests to pass. Tool: pytest.

## Phase 5 — Connect the installed entry point

### Stage 1. Preserve the CLI contract

1. [ ] Add `test_bare_cli_launches_only_with_terminal` to `tests/test_cli.py`. With a stub launcher, test bare invocation with two terminal streams and each redirected stream. Check that explicit `main([])` keeps its parser behavior. Tool: pytest.
2. [ ] Run the new test. Expect the interactive case to fail before dispatch is added. Tool: pytest.
3. [ ] Add bare dispatch to `main(argv: Sequence[str] | None = None) -> int`. When `argv is None`, there are no command-line arguments, and both streams are terminals, load `launch_tui` and pass the current directory. Keep all explicit arguments on the existing parser path. Tool: Python.
4. [ ] Add the module entry point in `src/cli.py` so `python -m thesis_tsad.cli` calls `main()` and returns its status. Tool: Python.
5. [ ] Run `.venv/bin/python -m pytest -q tests/test_cli.py tests/test_dataset_picker.py tests/test_tui.py`. Expect all tests to pass. Tool: pytest.

### Stage 2. Check the wheel

1. [ ] Build a wheel in a temporary directory with `uv build --wheel --offline --no-build-isolation --python .venv/bin/python --out-dir OUT_DIR`. Expect a successful build. Tool: uv.
2. [ ] Install the wheel in a separate test environment with the project dependencies available. Use `uv pip install --offline --no-deps --python TEST_PYTHON WHEEL`. Expect a successful install. Tool: uv.
3. [ ] Run the installed `tsad --help` and `python -m thesis_tsad.cli --help` outside the source checkout. Expect the current CLI help. Tool: the installed wheel.
4. [ ] Run the installed bare `tsad` with redirected input and output. Expect the baseline parser output and exit status. Tool: shell redirection.

## Phase 6 — Verify the terminal flow and record evidence

### Stage 1. Check the installed interactive flow

1. [ ] Prepare a disposable project with two experiment definitions and at least two valid prepared versions per test dataset. Reuse the fixture format in `tests/test_cli.py`. Record the initial YAML bytes and prepared manifest hashes. Tool: Python.
2. [ ] Open the installed bare `tsad` in a terminal in the disposable project. Expect the project prompt. Tool: terminal or Python pseudo-terminal.
3. [ ] Submit `/list experiments`. Expect the configured names in the command screen. Tool: terminal.
4. [ ] Open each session with `/experiment NAME`. Expect the correct title and history after each change. Tool: terminal.
5. [ ] Run `/datasets add smd swat`. Choose Cancel with the keyboard. Expect the cancellation message and the original YAML bytes. Use Enter and **Back** to return to the same session. Tools: terminal and Python.
6. [ ] Run the same command. Choose a different version with the keyboard. Move to **Save** and press Enter. Expect both dataset names in the experiment and the selected paths in the shared profiles. Tools: terminal and Python.
7. [ ] Compare the prepared manifest hashes with the initial hashes. Expect equality. Tool: SHA-256.
8. [ ] Run `/methods add mustard` and inspect the experiment YAML. Expect the existing CLI to add the method. Tools: terminal and Python.
9. [ ] Submit an invalid CLI action, such as `/list unknown`. Expect the parser message and a nonzero status. Expect **Back** to restore the text bar. Tool: terminal.
10. [ ] Check Ctrl+C with the controlled child from Phase 3 in a terminal with a pseudo-terminal test runner. Also check Ctrl+C during the picker handoff. Expect the app to recover and no child to remain. Tool: terminal or Python pseudo-terminal.

### Stage 2. Complete the checks

1. [ ] Run one SMD MUSTARD combination through `/run --smoke` in a disposable project. Use the `machine-1-1` cell and the O0 offline profile from `configs/experiments/smd_comparison.yaml`. Expect a successful run and its checkpoint. Tool: installed `tsad`. Keep the full experiment out of this check.
2. [ ] Run `.venv/bin/python -m pytest -q`. Compare the result with the recorded baseline and report new failures. Tool: pytest.
3. [ ] Run `ruff check src/cli.py src/tui.py tests/test_cli.py tests/test_tui.py`. Expect no findings in these files. Tool: Ruff.
4. [ ] Add README steps for bare `tsad`, `/list experiments`, `/experiment NAME`, `/help`, the picker keyboard controls, and `/quit`. Tool: Markdown.
5. [ ] Run `git diff --check`. Expect no whitespace findings. Tool: Git.
6. [ ] Review all changes against the starting state, including new and untracked files. Check the approved design, file sizes, and documented compatibility. Tools: Git and Python.
7. [ ] Record commands, statuses, and any incomplete checks in the execution record. Tool: Markdown.

## Compatibility and recovery

1. The runtime hashes Python source paths and contents. Existing attempts can fail current identity checks after this change. Keep the checks. Report an identity mismatch with the original CLI message.
2. If a phase fails, diagnose the failure before the next phase. To remove the TUI change, reverse only this task's edits against the recorded starting state. Keep prepared data, experiment definitions, run files, and unrelated workspace changes.

"""Run thesis workflows from the command line."""

import argparse
from collections.abc import Sequence
from importlib import import_module
import json
from pathlib import Path
import sys

from .artifacts import allocate_run
from .config import list_names, load_yaml, resolve_run
from .dataset_selection import add_datasets, edit_lock
from .experiments import (
    _call_runner, _experiment_name, create_experiment, edit_experiment, list_experiments,
    plan_experiment, resume_experiment, run_experiment,
)
from .reporting import report_experiment


def complete_names(kind: str, prefix: str, *, project_root: Path) -> list[str]:
    names = list_experiments(project_root=project_root) if kind == "experiments" else list_names(
        kind, project_root=project_root
    )
    return [name for name in names if name.startswith(prefix)]


def _parser():
    parser = argparse.ArgumentParser(prog="tsad", description="Run thesis time-series experiments.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Thesis project directory")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List available names")
    listing.add_argument("kind", choices=("datasets", "methods", "experiments", "standalones"))
    for verb in ("new", "validate", "run", "resume", "report"):
        command = commands.add_parser(verb)
        command.add_argument("scope", choices=("experiment", "standalone") if verb == "run" else ("experiment",))
        command.add_argument("--name", required=True)
        if verb == "run":
            command.add_argument("--smoke", action="store_true", help="Use the approved small work limits")
        elif verb == "report":
            command.add_argument("--smoke", action="store_true", help="Read saved smoke runs")
    editor = commands.add_parser("experiment", help="Show or edit a definition")
    editor.add_argument("--name", required=True)
    operations = editor.add_subparsers(dest="operation", required=True)
    operations.add_parser("show")
    for field in ("methods", "datasets"):
        selection = operations.add_parser(field)
        selection.add_argument("action", choices=("add", "remove"))
        selection.add_argument("values", nargs="+")
    settings = operations.add_parser("set")
    settings.add_argument("field", choices=("random_seeds", "wandb_logging", "kaggle_backup"))
    settings.add_argument("values", nargs="+")
    commands.add_parser("preprocess", help="Prepare source data", add_help=False)
    completion = commands.add_parser("completion", help="Print shell completion setup")
    completion.add_argument("shell", choices=("bash", "zsh"))
    complete = commands.add_parser("_complete")
    complete.add_argument("kind", choices=("datasets", "methods", "experiments", "standalones"))
    complete.add_argument("prefix", nargs="?", default="")
    return parser


def _preprocess_main(arguments):
    package = "thesis_scripts" if __package__ == "thesis_tsad" else "scripts"
    return import_module(f"{package}.preprocess").main(arguments)


def _completion_script(shell):
    if shell == "zsh":
        return """# Run after compinit.
_tsad_names() {
    local kind=experiments
    case "$words[CURRENT-1]" in
        --dataset) kind=datasets ;;
        --name) [[ "$words[3]" == standalone ]] && kind=standalones ;;
        *) case " $words[*] " in
               *" methods "*) kind=methods ;;
               *" datasets "*) kind=datasets ;;
               *) return ;;
           esac ;;
    esac
    local -a names
    names=("${(@f)$(tsad _complete "$kind" "$words[CURRENT]")}")
    (( ${#names} )) && compadd -a names
}
compdef _tsad_names tsad"""
    return """_tsad_names() {
    local kind=experiments prev="${COMP_WORDS[COMP_CWORD-1]}"
    case "$prev" in
        --dataset) kind=datasets ;;
        --name) [[ "${COMP_WORDS[2]}" == standalone ]] && kind=standalones ;;
        *) case " ${COMP_WORDS[*]} " in
               *" methods "*) kind=methods ;;
               *" datasets "*) kind=datasets ;;
               *) return ;;
           esac ;;
    esac
    COMPREPLY=()
    local name
    while IFS= read -r name; do COMPREPLY+=("$name"); done < <(tsad _complete "$kind" "${COMP_WORDS[COMP_CWORD]}")
}
complete -F _tsad_names tsad"""


def _print(value):
    print(json.dumps(value, default=str, indent=2))


def _results_status(results):
    _print(results)
    for result in results:
        if result["status"] != "successful":
            return 1
        for provider in ("wandb", "kaggle"):
            if result.get(provider, {}).get("status") in {"failed", "unknown"}:
                return 1
    return 0


def _choose_versions(choices):
    if sys.stdin.isatty() and sys.stdout.isatty():
        from .dataset_picker import DatasetPicker

        return DatasetPicker(choices).run()
    return {name: versions[0][0] for name, versions in choices.items()}


def _edit_definition(command, path, root):
    if command.operation == "show":
        _print(plan_experiment(path, project_root=root))
    elif command.operation == "datasets" and command.action == "add":
        selected = add_datasets(path, command.values, root, _choose_versions)
        if selected is None:
            print("Selection canceled.")
        else:
            for dataset, version in selected.items():
                print(f"{dataset}: {version}")
    else:
        with edit_lock(root):
            if command.operation == "set":
                edit_experiment(path, "set", command.field, command.values, project_root=root)
            else:
                edit_experiment(path, command.action, command.operation, command.values, project_root=root)


def _execute(command, root):
    verb = command.command
    if verb in {"list", "_complete"}:
        print("\n".join(complete_names(command.kind, getattr(command, "prefix", ""), project_root=root)))
        return 0
    if verb == "new":
        with edit_lock(root):
            print(create_experiment(command.name, project_root=root))
        return 0
    name = _experiment_name(command.name)
    if verb == "run" and command.scope == "standalone":
        if command.smoke:
            raise ValueError("use --smoke with an experiment")
        cell = load_yaml(root / "configs/standalones" / f"{name}.yaml")
        config = resolve_run(cell, project_root=root)
        run = allocate_run(config, project_root=root)
        return _results_status([_call_runner(run)])
    path = root / "configs/experiments" / f"{name}.yaml"
    if verb == "experiment":
        _edit_definition(command, path, root)
    elif verb == "validate":
        print(f"Valid experiment. Planned runs: {len(plan_experiment(path, project_root=root))}")
    elif verb == "run":
        return _results_status(run_experiment(path, project_root=root, smoke=command.smoke))
    elif verb == "resume":
        return _results_status(resume_experiment(path, project_root=root))
    elif verb == "report":
        paths = report_experiment(path, project_root=root, smoke=command.smoke)
        for result in paths:
            print(result)
        if not paths:
            mode = "smoke" if command.smoke else "normal"
            print(f"No completed runs to report in {mode} mode.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and not sys.argv[1:] and sys.stdin.isatty() and sys.stdout.isatty():
        from .tui import launch_tui

        return launch_tui(Path.cwd())
    parser = _parser()
    command, rest = parser.parse_known_args(argv)
    if command.command == "preprocess":
        return _preprocess_main(rest)
    if rest:
        parser.error("unrecognized arguments: " + " ".join(rest))
    if command.command == "completion":
        print(_completion_script(command.shell))
        return 0
    root = command.project_root.resolve()
    try:
        if not (root / "pyproject.toml").is_file() or not (root / "configs").is_dir():
            raise ValueError(f"thesis project not found: {root}")
        return _execute(command, root)
    except KeyboardInterrupt:
        print("Interrupted. Run state is saved at the last complete step.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"tsad stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

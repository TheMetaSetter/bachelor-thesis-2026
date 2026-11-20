"""Check selected versions and save shared data profiles."""

from contextlib import contextmanager
from fcntl import LOCK_EX, flock
import os
from pathlib import Path
from stat import S_IMODE

import yaml

from .artifacts import _write_atomic
from .config import load_yaml
from .experiments import _dataset_profile, _definition, _edit_selection
from .prepared_data import discover_versions, load_entity


@contextmanager
def edit_lock(project):
    """Lock CLI edits to the configuration directory."""
    descriptor = os.open(project / "configs", os.O_RDONLY | os.O_DIRECTORY)
    try:
        flock(descriptor, LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _profile_choices(project, paths):
    choices, profiles = {}, {}
    for dataset, path in paths.items():
        profile = load_yaml(path)
        if profile.get("dataset") != dataset:
            raise ValueError(f"{path}: expected data profile for {dataset}")
        configured = profile.get("prepared_root")
        choices[dataset] = discover_versions(project, dataset, Path(configured) if configured else None)
        if not choices[dataset]:
            raise ValueError(f"{dataset}: no complete prepared version in {project / 'data/preprocessed' / dataset}")
        profiles[dataset] = profile
    return choices, profiles


def _check_selection(selected, choices):
    if selected.keys() != choices.keys():
        raise ValueError("Select one version for each dataset.")
    for dataset, path in selected.items():
        entities = dict(choices[dataset]).get(path)
        if entities is None:
            raise ValueError(f"{dataset}: select a version from the list")
        for entity in entities:
            prepared = load_entity(path, entity)
            if prepared["identity"]["dataset"] != dataset:
                raise ValueError(f"{path}: expected prepared data for {dataset}")


def _yaml_bytes(value):
    return yaml.safe_dump(value, sort_keys=False).encode("utf-8")


def _write_settings(path, contents, mode):
    def write(file):
        file.write(contents)
        os.fchmod(file.fileno(), mode)

    _write_atomic(path, write)


def _save(project, original, updates):
    with edit_lock(project):
        for path, before in original.items():
            if path.read_bytes() != before:
                raise ValueError(f"{path}: changed during selection. Run datasets add again.")
        modes = {path: S_IMODE(path.stat().st_mode) for path in updates}
        attempted = []
        try:
            for path, contents in updates.items():
                attempted.append(path)
                _write_settings(path, contents, modes[path])
        except BaseException:
            failed = []
            for path in reversed(attempted):
                try:
                    if path.read_bytes() != original[path]:
                        _write_settings(path, original[path], modes[path])
                except OSError as error:
                    failed.append(f"{path}: {error}")
            if failed:
                raise RuntimeError("Could not restore files:\n" + "\n".join(failed)) from None
            raise


def add_datasets(path, names, project, choose):
    """Add datasets and save their checked paths, or return None on cancel."""
    original = {path: path.read_bytes()}
    definition = _definition(path)
    names = list(dict.fromkeys(names))
    previous = definition.get("datasets", [])
    _edit_selection(definition, "add", "datasets", names, project)
    cells = definition.get("cells", [])
    paths = {name: _dataset_profile(project, name, cells) for name in names}
    original.update({profile: profile.read_bytes() for profile in paths.values()})
    choices, profiles = _profile_choices(project, paths)
    selected = choose(choices)
    if selected is None:
        return None
    _check_selection(selected, choices)
    updates = {}
    for dataset, version in selected.items():
        value = str(version.relative_to(project) if version.is_relative_to(project) else version)
        if profiles[dataset].get("prepared_root") != value:
            profiles[dataset]["prepared_root"] = value
            updates[paths[dataset]] = _yaml_bytes(profiles[dataset])
    if definition["datasets"] != previous:
        updates[path] = _yaml_bytes(definition)
    _save(project, original, updates)
    return selected

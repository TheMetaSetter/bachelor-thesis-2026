"""Send committed summaries and narrowly selected backup artifacts."""

from hashlib import sha256
from importlib import import_module
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .artifacts import (
    _hash,
    _json_bytes,
    _read_json,
    _write_json,
    find_runs,
    result_directory,
)


def _wandb():
    return import_module("wandb")


def log_wandb(event: dict, *, session=None):
    """Use the provider session for one approved lifecycle event."""
    if not event.get("enabled", False):
        return None
    kind = event["kind"]
    try:
        if kind == "start":
            return _wandb().init(
                id=event["run_id"],
                group=event.get("experiment_id"),
                project=event.get("project", "bachelor-thesis-2026"),
                resume="allow",
            )
        if kind == "metrics":
            session.log(
                {
                    key: value
                    for key, value in event["metrics"].items()
                    if type(value) in (int, float)
                }
            )
            return session
        if kind == "finish":
            session.finish()
            return None
    except Exception:
        raise RuntimeError(f"W&B {kind} failed") from None
    raise ValueError("unknown W&B event")


def _kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi

    client = KaggleApi()
    client.authenticate()
    return client


def _checked_file(path, root, checksum=None):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if (
        not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("invalid or missing backup artifact")
    if checksum and _hash(path) != checksum:
        raise ValueError("backup artifact checksum mismatch")
    return path


def _checkpoint_files(paths, directory, reference, root):
    file = _checked_file(reference, root)
    if file.parent != directory or file.suffix != ".pt":
        raise ValueError("checkpoint is outside its state directory")
    descriptor = _checked_file(file.with_suffix(".pt.json"), root)
    _checked_file(file, root, _read_json(descriptor)["sha256"])
    paths.update((file, descriptor))


def _selected_state(paths, run, root):
    directory = Path(run["state_dir"])
    detail = run.get("details", {})
    for name in ("selected_checkpoint", "final_checkpoint"):
        if detail.get(name):
            _checkpoint_files(paths, directory, detail[name], root)
    for stage in detail.get("stages", ()):
        for role in ("initial_checkpoint", "best_checkpoint"):
            _checkpoint_files(paths, directory, stage[role], root)
    if detail.get("calibration"):
        calibration = _checked_file(
            detail["calibration"], root, detail.get("calibration_sha256")
        )
        if calibration != directory / "calibration.json":
            raise ValueError("invalid calibration path")
        paths.add(calibration)
    pointer = directory / "resume.json"
    if pointer.exists():
        saved = _read_json(_checked_file(pointer, root))
        if saved.get("run_id") != run["run_id"] or saved.get("slot") not in (0, 1):
            raise ValueError("invalid committed recovery pointer")
        slot = _checked_file(
            directory / f"resume-{saved['slot']}.pt", root, saved["sha256"]
        )
        paths.update((pointer, slot))
        for role, reference in saved.get("role_references", {}).items():
            if role == "selected" and reference.get("path"):
                _checkpoint_files(paths, directory, reference["path"], root)
            elif role.endswith((".initial", ".best")):
                _checkpoint_files(paths, directory, reference, root)


def _backup_files(root, experiment):
    paths = set()
    for run in find_runs(root, filters={"experiment_id": experiment}):
        if "resolved_config" not in run:
            continue
        config = run["resolved_config"]
        state_path = result_directory(config, "state") / "run.json"
        if run["record_path"] != state_path:
            raise ValueError("run metadata is outside its recorded state directory")
        paths.add(_checked_file(state_path, root))
        _selected_state(paths, run, root)
        if run["status"] != "successful":
            continue
        for split, filename in run.get("details", {}).get("splits", {}).items():
            result = _checked_file(filename, root)
            if result != result_directory(config, split) / "result.json":
                raise ValueError("split result is outside its recorded directory")
            record = _read_json(result)
            if record.get("run_id") != run["run_id"] or record.get("split") != split:
                raise ValueError("invalid backup result identity")
            paths.add(result)
            if record.get("point_scores_sha256"):
                paths.add(
                    _checked_file(
                        result.parent / "point_scores.parquet",
                        root,
                        record["point_scores_sha256"],
                    )
                )
    for attempt in sorted((root / "attempts").glob("*.json")):
        saved = _read_json(_checked_file(attempt, root))
        if saved.get("schema_version") == 1 and all(
            item.get("experiment_id") == experiment
            and Path(item["output_root"]) == root
            for item in saved["runs"]
        ):
            paths.add(attempt)
    return sorted(paths)


def _private_dataset(client, dataset):
    _, slug = dataset.split("/")
    for page in range(1, 101):
        matches = client.dataset_list(search=slug, mine=True, page=page) or []
        if not matches:
            break
        for item in matches:
            if getattr(item, "ref", None) == dataset:
                return getattr(item, "is_private", None) is True
    return False


def _send_backup(root, dataset, files, manifest, receipt):
    try:
        client = _kaggle()
        if not _private_dataset(client, dataset):
            return {
                "status": "failed",
                "reason": "private Kaggle dataset could not be verified",
            }
        with TemporaryDirectory(prefix=".kaggle-upload-", dir=root) as temp:
            folder = Path(temp)
            for path, entry in zip(files, manifest["files"], strict=True):
                link = folder / entry["file"]
                if not link.exists():
                    os.link(path, link)
                if _hash(link) != entry["sha256"]:
                    raise ValueError("backup artifact changed")
            _write_json(folder / "backup-manifest.json", manifest)
            _write_json(folder / "dataset-metadata.json", {"id": dataset})
            pending = {
                "status": "unknown",
                "manifest_sha256": manifest["manifest_sha256"],
            }
            _write_json(receipt, pending)
            response = client.dataset_create_version(
                str(folder),
                version_notes="Thesis experiment backup",
                quiet=True,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="skip",
            )
            if getattr(response, "status", "").lower() != "ok":
                return pending
            confirmed = pending | {
                "status": "successful",
                "remote_version": getattr(response, "version_number", None),
            }
            _write_json(receipt, confirmed)
            return confirmed
    except Exception:
        if receipt.is_file():
            return {"status": "unknown", "manifest_sha256": manifest["manifest_sha256"]}
        return {"status": "failed", "reason": "Kaggle backup preparation failed"}


def backup_kaggle(experiment_root: Path, settings: dict) -> dict:
    """Back up one checked experiment to an existing private dataset."""
    if not settings.get("kaggle_backup", False):
        return {"status": "disabled"}
    root = Path(experiment_root).resolve()
    experiment = settings["experiment_id"]
    if root.name != experiment or root.parent.name not in {"experiments", "smoke"}:
        raise ValueError("expected the exact experiment root")
    dataset = settings.get("kaggle_dataset") or os.environ.get(
        "TSAD_KAGGLE_DATASET", ""
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", dataset):
        return {
            "status": "failed",
            "reason": "configure a private Kaggle dataset as owner/name",
        }
    try:
        files = _backup_files(root, experiment)
    except (OSError, KeyError, ValueError):
        return {
            "status": "failed",
            "reason": "Kaggle backup artifacts did not pass local checks",
        }
    if not files:
        return {
            "status": "failed",
            "reason": "the experiment has no checked backup artifacts",
        }
    entries = []
    for file in files:
        checksum = _hash(file)
        entries.append(
            {
                "path": file.relative_to(root).as_posix(),
                "file": checksum + file.suffix,
                "sha256": checksum,
            }
        )
    manifest = {"schema_version": 1, "experiment_id": experiment, "files": entries}
    checksum = sha256(_json_bytes(manifest)).hexdigest()
    manifest["manifest_sha256"] = checksum
    key = sha256(f"{dataset}:{checksum}".encode()).hexdigest()
    receipt = root / ".backup" / f"{key}.json"
    if receipt.is_file():
        saved = _read_json(receipt)
        return (
            saved | {"status": "already_saved"}
            if saved["status"] == "successful"
            else saved
        )
    return _send_backup(root, dataset, files, manifest, receipt)

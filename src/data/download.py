from __future__ import annotations

"""Packaged download helpers for the canonical SMD raw dataset tree.

The active parser expects the same directory layout used by the notebook
templates. This module therefore mirrors that layout exactly instead of adding a
second storage convention.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import requests

from src.core.console import console_print

DATASET_REPOSITORY_OWNER = "NetManAIOps"
DATASET_REPOSITORY_NAME = "OmniAnomaly"
DATASET_REPOSITORY_BRANCH = "master"
DATASET_DIRECTORY_IN_REPOSITORY = "ServerMachineDataset"
REQUIRED_DATASET_DIRECTORIES = ("train", "test", "test_label")
OPTIONAL_DATASET_CHILDREN = ("LICENSE", "interpretation_label")


@dataclass(frozen=True)
class DownloadedPathRecord:
    repository_path: str
    local_path: str
    item_type: str


def build_github_contents_api_url(path_in_repository: str) -> str:
    return (
        f"https://api.github.com/repos/"
        f"{DATASET_REPOSITORY_OWNER}/"
        f"{DATASET_REPOSITORY_NAME}/contents/"
        f"{path_in_repository}"
        f"?ref={DATASET_REPOSITORY_BRANCH}"
    )


def _build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "User-Agent": "smd-dataset-downloader",
        }
    )
    return session


def _find_repo_root(start_path: str | Path | None = None) -> Path:
    search_path = Path(start_path or __file__).expanduser().resolve()
    for candidate in [search_path, *search_path.parents]:
        if (candidate / ".git").exists() or (candidate / "AGENTS.md").exists():
            return candidate
    return search_path


def resolve_repo_relative_path(path: str | Path) -> Path:
    expanded_path = Path(os.path.expanduser(os.path.expandvars(str(path))))
    if expanded_path.is_absolute():
        return Path(os.path.abspath(str(expanded_path)))

    cwd_candidate = Path(os.path.abspath(str(Path.cwd() / expanded_path)))
    repo_root = _find_repo_root()
    repo_candidate = Path(os.path.abspath(str(repo_root / expanded_path)))
    repo_candidate_without_parent = None
    if expanded_path.parts and expanded_path.parts[0] == "..":
        stripped_parts = expanded_path.parts[1:]
        if stripped_parts:
            repo_candidate_without_parent = Path(
                os.path.abspath(str(repo_root / Path(*stripped_parts)))
            )

    for candidate in [cwd_candidate, repo_candidate, repo_candidate_without_parent]:
        if candidate is not None and candidate.exists():
            return candidate
    return cwd_candidate


def get_smd_dataset_root(root_dir: str | Path) -> Path:
    configured_root_dir = os.environ.get("SMD_ROOT_DIR", root_dir)
    root_path = resolve_repo_relative_path(configured_root_dir)
    resolved_from_env = "SMD_ROOT_DIR" in os.environ
    if root_path.name == DATASET_DIRECTORY_IN_REPOSITORY:
        console_print(
            "DATA",
            "Resolved SMD dataset root",
            input_root=root_dir,
            configured_root_dir=configured_root_dir,
            resolved_root=root_path,
            used_environment_override=resolved_from_env,
        )
        return root_path
    resolved_root = root_path / DATASET_DIRECTORY_IN_REPOSITORY
    console_print(
        "DATA",
        "Resolved SMD dataset root",
        input_root=root_dir,
        configured_root_dir=configured_root_dir,
        resolved_root=resolved_root,
        used_environment_override=resolved_from_env,
    )
    return resolved_root


def is_smd_dataset_present(root_dir: str | Path) -> bool:
    dataset_root = get_smd_dataset_root(root_dir)
    is_present = all(
        (dataset_root / directory_name).exists()
        for directory_name in REQUIRED_DATASET_DIRECTORIES
    )
    console_print(
        "DATA",
        "Checked SMD dataset presence",
        dataset_root=dataset_root,
        is_present=is_present,
    )
    return is_present


def ensure_smd_dataset_layout(root_dir: str | Path) -> Path:
    dataset_root = get_smd_dataset_root(root_dir)
    missing_directories = [
        directory_name
        for directory_name in REQUIRED_DATASET_DIRECTORIES
        if not (dataset_root / directory_name).exists()
    ]
    if missing_directories:
        raise FileNotFoundError(
            f"SMD dataset root is missing required directories under {dataset_root}: {missing_directories}"
        )
    console_print("DATA", "Verified SMD dataset layout", dataset_root=dataset_root)
    return dataset_root


def _request_github_listing(
    http_session: requests.Session,
    path_in_repository: str,
) -> list[dict[str, Any]]:
    response = http_session.get(
        build_github_contents_api_url(path_in_repository), timeout=30
    )
    response.raise_for_status()
    listing = response.json()
    if not isinstance(listing, list):
        raise TypeError(f"Expected a GitHub directory listing for {path_in_repository}")
    return listing


def list_smd_download_plan(root_dir: str | Path) -> list[DownloadedPathRecord]:
    """Return the minimal path plan needed to materialize the canonical SMD tree.

    This helper is intentionally network-free so tests can verify path planning
    without depending on Kaggle, GitHub, or a full dataset download.
    """

    dataset_root = get_smd_dataset_root(root_dir)
    return [
        DownloadedPathRecord(
            repository_path=f"{DATASET_DIRECTORY_IN_REPOSITORY}/{child_name}",
            local_path=str(dataset_root / child_name),
            item_type="dir",
        )
        for child_name in REQUIRED_DATASET_DIRECTORIES + OPTIONAL_DATASET_CHILDREN
    ]


def download_smd_dataset(
    root_dir: str | Path,
    *,
    skip_existing_download: bool = True,
) -> Path:
    dataset_root = get_smd_dataset_root(root_dir)
    console_print("DATA", "Preparing SMD dataset download", dataset_root=dataset_root)
    if skip_existing_download and is_smd_dataset_present(dataset_root):
        return ensure_smd_dataset_layout(dataset_root)

    http_session = _build_http_session()

    def download_binary_file(
        file_download_url: str, local_output_file_path: Path
    ) -> None:
        local_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_response = http_session.get(file_download_url, timeout=60)
        file_response.raise_for_status()
        local_output_file_path.write_bytes(file_response.content)

    def recursively_download_directory(
        repository_path: str, local_output_directory: Path
    ) -> None:
        github_items = _request_github_listing(http_session, repository_path)
        local_output_directory.mkdir(parents=True, exist_ok=True)
        for github_item in github_items:
            github_item_type = github_item["type"]
            github_item_name = github_item["name"]
            github_item_path = github_item["path"]
            if github_item_type == "dir":
                recursively_download_directory(
                    repository_path=github_item_path,
                    local_output_directory=local_output_directory / github_item_name,
                )
            elif github_item_type == "file":
                download_binary_file(
                    file_download_url=github_item["download_url"],
                    local_output_file_path=local_output_directory / github_item_name,
                )
            else:
                raise ValueError(f"Unsupported GitHub item type: {github_item_type}")

    recursively_download_directory(DATASET_DIRECTORY_IN_REPOSITORY, dataset_root)
    console_print("DATA", "Completed SMD dataset download", dataset_root=dataset_root)
    return ensure_smd_dataset_layout(dataset_root)

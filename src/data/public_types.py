from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublicDataBundle(Mapping[str, Any]):
    """Notebook-friendly wrapper around the existing dataset bundle dictionary.

    The wrapper is additive rather than replacement-oriented. It exposes
    attribute access for readability in notebooks while still behaving like a
    mapping so existing dictionary-shaped expectations remain easy to satisfy.
    """

    dataset_name: str
    parser: Any
    scaler: Any
    raw_sequences: dict[str, list[dict[str, Any]]]
    scaled_sequences: dict[str, list[dict[str, Any]]]
    datasets: dict[str, Any]
    loaders: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        yield from (
            "dataset_name",
            "parser",
            "scaler",
            "raw_sequences",
            "scaled_sequences",
            "datasets",
            "loaders",
        )

    def __len__(self) -> int:
        return 7

    def as_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}

from __future__ import annotations

from typing import Any, Callable


DATASET_BUILDERS: dict[str, Callable[..., Any]] = {}
MODEL_BUILDERS: dict[str, Callable[..., Any]] = {}


def register_dataset(name: str, builder: Callable[..., Any]) -> None:
    DATASET_BUILDERS[name] = builder


def register_model(name: str, builder: Callable[..., Any]) -> None:
    MODEL_BUILDERS[name] = builder


def build_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in DATASET_BUILDERS:
        raise KeyError(f"Unknown dataset builder: {name}")
    return DATASET_BUILDERS[name](*args, **kwargs)


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model builder: {name}")
    return MODEL_BUILDERS[name](*args, **kwargs)


def clear_registry() -> None:
    DATASET_BUILDERS.clear()
    MODEL_BUILDERS.clear()

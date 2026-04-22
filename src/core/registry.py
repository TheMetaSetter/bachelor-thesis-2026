from __future__ import annotations

from typing import Any, Callable

from src.core.console import console_print


DATASET_BUILDERS: dict[str, Callable[..., Any]] = {}
MODEL_BUILDERS: dict[str, Callable[..., Any]] = {}


def register_dataset(name: str, builder: Callable[..., Any]) -> None:
    """
    Hàm này nhận vào tên của một tập dữ liệu
    và đăng ký (register) builder tương ứng
    với tập dữ liệu đó vào danh sách dataset builder.
    Đây là mẫu thiết kế registry pattern.
    """

    DATASET_BUILDERS[name] = builder
    console_print(
        "REGISTRY", "Registered dataset builder", name=name, builder=builder.__name__
    )


def register_model(name: str, builder: Callable[..., Any]) -> None:
    MODEL_BUILDERS[name] = builder
    console_print(
        "REGISTRY", "Registered model builder", name=name, builder=builder.__name__
    )


def build_dataset(name: str, *args: Any, **kwargs: Any) -> Any:

    """
    Hàm này có nhiệm vụ build 
    Hàm này nhận vào tên của tập dữ liệu cần build
    và các tham số cấu hình tương ứng với tập dữ liệu đó.
    """
    if name not in DATASET_BUILDERS:
        raise KeyError(f"Unknown dataset builder: {name}")
    console_print("REGISTRY", "Building dataset", name=name)
    return DATASET_BUILDERS[name](*args, **kwargs)


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model builder: {name}")
    console_print("REGISTRY", "Building model", name=name)
    return MODEL_BUILDERS[name](*args, **kwargs)


def clear_registry() -> None:
    DATASET_BUILDERS.clear()
    MODEL_BUILDERS.clear()
    console_print("REGISTRY", "Cleared dataset and model registries")

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


def _build_registered_component(
    builders: dict[str, Callable[..., Any]],
    component_name: str,
    component_type: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if component_name not in builders:
        raise KeyError(f"Unknown {component_type} builder: {component_name}")
    console_print(
        "REGISTRY",
        f"Building {component_type}",
        name=component_name,
    )
    return builders[component_name](*args, **kwargs)


def build_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Hàm này có nhiệm vụ build
    Hàm này nhận vào tên của tập dữ liệu cần build
    và các tham số cấu hình tương ứng với tập dữ liệu đó.
    """
    return _build_registered_component(
        DATASET_BUILDERS,
        name,
        "dataset",
        *args,
        **kwargs,
    )


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    return _build_registered_component(
        MODEL_BUILDERS,
        name,
        "model",
        *args,
        **kwargs,
    )


def clear_registry() -> None:
    DATASET_BUILDERS.clear()
    MODEL_BUILDERS.clear()
    console_print("REGISTRY", "Cleared dataset and model registries")

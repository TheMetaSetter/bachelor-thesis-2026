from __future__ import annotations

from src.core.registry import (
    build_dataset,
    build_model,
    clear_registry,
    register_dataset,
    register_model,
)


def _dataset_builder() -> str:
    return "dataset"


def _model_builder() -> str:
    return "model"


def test_registry_resolves_dataset_and_model_builders() -> None:
    clear_registry()

    register_dataset("smd", _dataset_builder)
    register_model("reconstruction_mlp_ae", _model_builder)

    assert build_dataset("smd") == "dataset"
    assert build_model("reconstruction_mlp_ae") == "model"


def test_registry_can_register_multiple_model_variants() -> None:
    clear_registry()

    register_model("thesis_multitask", _model_builder)

    assert build_model("thesis_multitask") == "model"

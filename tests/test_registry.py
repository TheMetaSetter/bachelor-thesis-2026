from __future__ import annotations

from src.core.registry import (
    build_dataset,
    build_encoder,
    build_model,
    build_task,
    clear_registry,
    register_dataset,
    register_encoder,
    register_model,
    register_task,
)


def _dataset_builder() -> str:
    return "dataset"


def _model_builder() -> str:
    return "model"


def _task_builder() -> str:
    return "task"


def _encoder_builder() -> str:
    return "encoder"


def test_registry_resolves_dataset_model_task_and_encoder_builders() -> None:
    clear_registry()

    register_dataset("smd", _dataset_builder)
    register_model("reconstruction_mlp_ae", _model_builder)
    register_task("reconstruction", _task_builder)
    register_encoder("placeholder_encoder", _encoder_builder)

    assert build_dataset("smd") == "dataset"
    assert build_model("reconstruction_mlp_ae") == "model"
    assert build_task("reconstruction") == "task"
    assert build_encoder("placeholder_encoder") == "encoder"


def test_registry_can_register_phase_three_variants() -> None:
    clear_registry()

    register_model("thesis_multitask", _model_builder)
    register_task("multitask_tsad", _task_builder)

    assert build_model("thesis_multitask") == "model"
    assert build_task("multitask_tsad") == "task"

from __future__ import annotations

from src.core.registry import (
    DATASET_BUILDERS,
    MODEL_BUILDERS,
    build_dataset,
    build_model,
    clear_registry,
    register_dataset,
    register_model,
)
from src.core.runtime_components import (
    register_offline_runtime_components,
    register_online_runtime_components,
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


def test_registry_can_register_online_adaptation_model_variant() -> None:
    clear_registry()

    register_model("online_adaptation", _model_builder)

    assert build_model("online_adaptation") == "model"


def test_offline_runtime_component_registration_registers_canonical_builders() -> None:
    clear_registry()

    register_offline_runtime_components()

    assert "smd" in DATASET_BUILDERS
    assert "anomaly_archive" in DATASET_BUILDERS
    assert "reconstruction_mlp_ae" in MODEL_BUILDERS
    assert "thesis_multitask" in MODEL_BUILDERS
    assert "redlamp_baseline" in MODEL_BUILDERS


def test_online_runtime_component_registration_includes_online_model() -> None:
    clear_registry()

    register_online_runtime_components()

    assert "online_adaptation" in MODEL_BUILDERS

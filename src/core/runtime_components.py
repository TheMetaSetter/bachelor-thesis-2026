from __future__ import annotations

from src.core.console import console_print
from src.core.registry import register_dataset, register_model
from src.data.loaders import (
    build_anomaly_archive_dataset_bundle,
    build_smd_dataset_bundle,
)
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.redlamp_baseline import RedLampBaseline
from src.models.reconstruction_mlp_ae import ReconstructionMLPAutoencoder
from src.models.thesis_multitask import ThesisMultitaskModel


def register_shared_runtime_components() -> None:
    register_dataset("smd", build_smd_dataset_bundle)
    register_dataset("anomaly_archive", build_anomaly_archive_dataset_bundle)
    register_model("reconstruction_mlp_ae", ReconstructionMLPAutoencoder)
    register_model("thesis_multitask", ThesisMultitaskModel)
    register_model("redlamp_baseline", RedLampBaseline)


def register_offline_runtime_components() -> None:
    register_shared_runtime_components()
    console_print("REGISTRY", "Registered offline training runtime components")


def register_evaluation_runtime_components() -> None:
    register_shared_runtime_components()
    console_print("REGISTRY", "Registered evaluation runtime components")


def register_online_runtime_components() -> None:
    register_shared_runtime_components()
    register_model("online_adaptation", OnlineAdaptationModel)
    console_print("REGISTRY", "Registered online adaptation runtime components")

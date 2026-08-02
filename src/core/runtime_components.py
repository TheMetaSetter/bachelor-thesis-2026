from __future__ import annotations

from src.core.registry import register_dataset, register_model
from src.data.loaders import build_smd_dataset_bundle
from src.models.online_adaptation import OnlineAdaptationModel
from src.models.redlamp_baseline import RedLampBaseline
from src.models.thesis_multitask import ThesisMultitaskModel


def register_shared_runtime_components() -> None:
    register_dataset("smd", build_smd_dataset_bundle)
    register_model("thesis_multitask", ThesisMultitaskModel)
    register_model("redlamp_baseline", RedLampBaseline)


def register_offline_runtime_components() -> None:
    register_shared_runtime_components()


def register_evaluation_runtime_components() -> None:
    register_shared_runtime_components()


def register_online_runtime_components() -> None:
    # Đăng ký (register) builder cho để build dataset, model chính và baseline
    register_shared_runtime_components()
    register_model("online_adaptation", OnlineAdaptationModel)


def register_archival_dataset_components() -> None:
    """Register AnomalyArchive only for an explicit archival tool."""
    from src.data.loaders import build_anomaly_archive_dataset_bundle

    register_dataset("anomaly_archive", build_anomaly_archive_dataset_bundle)

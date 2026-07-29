from __future__ import annotations

from pathlib import Path

import torch
import pytest

from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta import checkpoint_resolution
from src.models.online_adaptation import (
    OnlineAdaptationModel,
    _resolve_reference_checkpoint_path,
)
from src.models.thesis_multitask import ThesisMultitaskModel


def test_reference_path_requires_an_existing_checkpoint(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "seed6" / "checkpoints" / "best.pt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _resolve_reference_checkpoint_path(requested)


def test_stage_b_checkpoint_resolver_prefers_metadata_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_resolution, "REPOSITORY_ROOT", tmp_path)
    resolved = (
        tmp_path
        / "outputs"
        / "benchmark"
        / "smd"
        / "thesis"
        / "O0"
        / "machine_1_6"
        / "seed6"
        / "two_stage"
        / "stage_b_fusion_finetuning"
        / "checkpoints"
        / "best.pt"
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(b"checkpoint")

    config = {
        "task": {
            "offline_variant": "O0",
            "entity_id": "machine_1_6",
            "seed": 6,
            "benchmark_mode": "main",
            "stage_name": "stage_b_fusion_finetuning",
        }
    }

    assert checkpoint_resolution.resolve_stage_b_checkpoint(config) == resolved


def test_stage_b_checkpoint_resolver_fails_when_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_resolution, "REPOSITORY_ROOT", tmp_path)
    config = {
        "task": {
            "offline_variant": "O0",
            "entity_id": "machine_1_6",
            "seed": 6,
            "benchmark_mode": "main",
            "stage_name": "stage_b_fusion_finetuning",
        }
    }

    with pytest.raises(FileNotFoundError, match="No Stage B checkpoint matches"):
        checkpoint_resolution.resolve_stage_b_checkpoint(config)


def test_stage_b_checkpoint_resolver_fails_on_ambiguous_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_resolution, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        checkpoint_resolution,
        "_find_stage_b_checkpoint_candidates",
        lambda stage_root: [
            tmp_path / "stage_b_fusion_finetuning" / "checkpoints" / "best.pt",
            tmp_path / "stage_b_fusion_finetuning" / "checkpoints" / "best.pt",
        ],
    )
    config = {
        "task": {
            "offline_variant": "O0",
            "entity_id": "machine_1_6",
            "seed": 6,
            "benchmark_mode": "main",
            "stage_name": "stage_b_fusion_finetuning",
        }
    }

    with pytest.raises(ValueError, match="Ambiguous Stage B checkpoint metadata"):
        checkpoint_resolution.resolve_stage_b_checkpoint(config)


def test_online_model_rejects_non_thesis_reference_checkpoint(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(38, 38)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="baseline.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={
            "model": {
                "model_name": "redlamp_baseline",
            },
            "task": {"task_name": "multitask_tsad"},
        },
        epoch=1,
        metric_history=[],
    )

    with pytest.raises(ValueError, match="thesis_multitask checkpoint"):
        OnlineAdaptationModel(
            input_dim=38,
            encoder_dim=64,
            hidden_dim=32,
            projector_hidden_dim=48,
            projector_dropout=0.0,
            enable_prototype_alignment=False,
            lambda_align=1.0,
            lambda_proto=0.1,
            lambda_anchor=0.001,
            score_source="projected_hidden",
            reference_checkpoint_path=str(checkpoint_path),
            warm_start_projector=False,
            target_param_group="projector_params",
            clean_stream_only=True,
        )


def test_online_model_accepts_multitask_reference_checkpoint_with_memory_extra_state(
    tmp_path: Path,
) -> None:
    reference_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=32,
        num_classes=2,
        dropout=0.0,
        bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    reference_model.mark_memories_initialized(initialization_epoch=3)
    optimizer = torch.optim.Adam(reference_model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="multitask_memory.pt",
        model=reference_model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={
            "model": {
                "model_name": "thesis_multitask",
                "input_dim": 38,
                "window_size": 100,
                "encoder_dim": 64,
                "hidden_dim": 32,
                "num_classes": 2,
                "dropout": 0.0,
                "continuous_enabled": True,
                "continuous_num_prototypes": 8,
                "discrete_enabled": True,
                "discrete_codebook_size": 16,
                "gumbel_temperature": 1.0,
                "temperature_start": 1.0,
                "temperature_end": 1.0,
                "temperature_anneal_fraction": 1.0,
                "alpha_logit_init": 0.0,
                "beta_logit_init": 0.0,
                "lambda_cls": 1.0,
                "lambda_div": 0.0,
                "lambda_var": 0.0,
                "lambda_cov": 0.0,
                "lambda_use": 0.0,
                "lambda_gate": 0.0,
                "bootstrap_encoder_epochs": 0,
            },
            "task": {
                "task_name": "multitask_tsad",
                "use_synthetic_augmentation": False,
                "anomaly_probability": 0.5,
                "min_segment_fraction": 0.1,
                "max_segment_fraction": 0.2,
                "spike_scale": 3.0,
            },
        },
        epoch=3,
        metric_history=[],
        extra_state=reference_model.get_checkpoint_extra_state(),
    )

    online_model = OnlineAdaptationModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=32,
        projector_hidden_dim=48,
        projector_dropout=0.0,
        enable_prototype_alignment=False,
        lambda_align=1.0,
        lambda_proto=0.1,
        lambda_anchor=0.001,
        score_source="projected_hidden",
        reference_checkpoint_path=str(checkpoint_path),
        warm_start_projector=False,
        target_param_group="projector_params",
        clean_stream_only=True,
    )

    assert online_model.reference_encoder.model.memory_initialized is True
    assert online_model.reference_encoder.model.memory_training_enabled is True

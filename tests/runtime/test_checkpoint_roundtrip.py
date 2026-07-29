from __future__ import annotations

from pathlib import Path

import torch

from src.data.scalers import SequenceStandardScaler
from src.engine.checkpoint import CheckpointManager
from src.models.thesis_multitask import ThesisMultitaskModel


def test_checkpoint_roundtrip_restores_model_optimizer_scaler_and_config(
    tmp_path: Path,
) -> None:
    model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = SequenceStandardScaler()
    scaler.feature_mean = torch.zeros(38)
    scaler.feature_std = torch.ones(38)
    config = {"experiment_name": "unit-test"}

    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="roundtrip.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state=scaler.state_dict(),
        config=config,
        epoch=3,
        metric_history=[{"val_loss": 1.0}],
    )

    reloaded_model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path, reloaded_model, reloaded_optimizer
    )

    assert loaded_checkpoint["config"] == config
    assert loaded_checkpoint["epoch"] == 3
    assert loaded_checkpoint["checkpoint_metadata"]["schema_version"] == 3
    assert loaded_checkpoint["checkpoint_metadata"]["experiment_name"] == "unit-test"
    assert (
        loaded_checkpoint["checkpoint_metadata"]["resolved_config_sha256"]
        == loaded_checkpoint["checkpoint_metadata"]["config_sha256"]
    )
    assert torch.equal(
        loaded_checkpoint["scaler_state_dict"]["feature_mean"], scaler.feature_mean
    )
    for parameter, reloaded_parameter in zip(
        model.parameters(), reloaded_model.parameters()
    ):
        assert torch.allclose(parameter, reloaded_parameter)
    assert "scheduler_state_dict" not in loaded_checkpoint


def test_checkpoint_roundtrip_restores_scheduler_state_when_present(
    tmp_path: Path,
) -> None:
    model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=0.0,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1.0e-5,
    )
    scaler = SequenceStandardScaler()
    scaler.feature_mean = torch.zeros(38)
    scaler.feature_std = torch.ones(38)
    config = {"experiment_name": "scheduler-roundtrip-test"}

    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)

    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="scheduler_roundtrip.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler_state=scaler.state_dict(),
        config=config,
        epoch=4,
        metric_history=[
            {"val_loss": 1.0, "optimizer_lr": optimizer.param_groups[0]["lr"]}
        ],
    )

    reloaded_model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    reloaded_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=reloaded_optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=0.0,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1.0e-5,
    )
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
        reloaded_scheduler,
    )

    assert "scheduler_state_dict" in loaded_checkpoint
    assert loaded_checkpoint["checkpoint_metadata"]["schema_version"] == 3
    assert reloaded_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert reloaded_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_roundtrip_restores_extra_memory_state(tmp_path: Path) -> None:
    model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)

    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="memory_extra_state.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={
            "feature_mean": torch.zeros(38),
            "feature_std": torch.ones(38),
        },
        config={"experiment_name": "memory-extra-state"},
        epoch=1,
        metric_history=[],
        extra_state={
            "memory_training_enabled": True,
            "memory_initialized": True,
            "bootstrap_encoder_epochs": 10,
        },
    )

    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path, model, optimizer
    )

    assert loaded_checkpoint["extra_state"] == {
        "memory_training_enabled": True,
        "memory_initialized": True,
        "bootstrap_encoder_epochs": 10,
    }
    assert loaded_checkpoint["extra_state"]["memory_training_enabled"] is True
    assert loaded_checkpoint["extra_state"]["memory_initialized"] is True
    assert loaded_checkpoint["extra_state"]["bootstrap_encoder_epochs"] == 10
    assert loaded_checkpoint["checkpoint_metadata"]["schema_version"] == 3


def test_multitask_checkpoint_roundtrip_restores_memory_buffers(tmp_path: Path) -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initialization_batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }
    model.set_epoch_context(epoch_index=1, total_epochs=2)
    model.maybe_initialize_memories_from_loader([initialization_batch], device="cpu")
    checkpoint_manager = CheckpointManager(tmp_path)

    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="multitask_memory.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={"experiment_name": "multitask-memory"},
        epoch=1,
        metric_history=[],
        extra_state=model.get_checkpoint_extra_state(),
    )

    reloaded_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=1,
    )
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
    )

    assert torch.allclose(
        model.continuous_prototype_bank,
        reloaded_model.continuous_prototype_bank,
    )


def test_multitask_checkpoint_roundtrip_repairs_verification_provenance(
    tmp_path: Path,
) -> None:
    from src.engine.online_tta.signature_verification import (
        PrototypeVerificationMetadata,
    )

    model = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=1,
    )
    model._initialize_memory_buffers_from_token_pool(
        continuous_hidden_tokens=torch.randn(4, 16),
        discrete_hidden_tokens_by_class={
            0: torch.randn(4, 16),
            1: torch.randn(4, 16),
        },
    )
    model.mark_memories_initialized(initialization_epoch=1)
    assert (
        model.get_checkpoint_extra_state()["verification_metadata_source"]
        == "train_anomaly_tokens_q99"
    )
    stale_state = model.get_checkpoint_extra_state()
    stale_state["verification_metadata_source"] = "uninitialized"

    restored = ThesisMultitaskModel(
        input_dim=38,
        window_size=100,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=1,
    )
    restored.load_checkpoint_extra_state(stale_state)

    repaired_state = restored.get_checkpoint_extra_state()

    assert restored.verification_metadata_source == "train_anomaly_tokens_q99"
    assert repaired_state["verification_metadata_source"] == "train_anomaly_tokens_q99"

    metadata = PrototypeVerificationMetadata.from_model(restored)
    assert metadata.source_split == "synthetic_train"


def test_stage_b_initialization_checkpoint_can_be_reloaded_with_stage_b_config(
    tmp_path: Path,
) -> None:
    checkpoint_manager = CheckpointManager(tmp_path)
    model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    stage_a_config = {
        "experiment_name": "stage-a",
        "model": {"model_name": "thesis_multitask"},
        "task": {"task_name": "multitask_tsad"},
        "seed": 8,
    }
    stage_b_config = {
        "experiment_name": "stage-b",
        "model": {"model_name": "thesis_multitask"},
        "task": {"task_name": "multitask_tsad"},
        "seed": 8,
    }

    stage_a_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="stage_a.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={
            "feature_mean": torch.zeros(38),
            "feature_std": torch.ones(38),
        },
        config=stage_a_config,
        epoch=1,
        metric_history=[{"train_loss": 1.0}],
    )
    payload = torch.load(stage_a_path, map_location="cpu")
    payload["config"] = stage_b_config
    payload["checkpoint_metadata"] = CheckpointManager._build_checkpoint_metadata(
        config=stage_b_config,
        epoch=int(payload["epoch"]),
        metric_history=list(payload["metric_history"]),
        extra_state=payload.get("extra_state"),
    )
    stage_b_init_path = tmp_path / "stage_b_init.pt"
    torch.save(payload, stage_b_init_path)

    reloaded_model = ThesisMultitaskModel(
        input_dim=38, window_size=100, encoder_dim=64, hidden_dim=16,
        num_classes=2, dropout=0.0, bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
    )
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        stage_b_init_path,
        reloaded_model,
        reloaded_optimizer,
    )

    assert loaded_checkpoint["config"] == stage_b_config
    assert loaded_checkpoint["checkpoint_metadata"]["experiment_name"] == "stage-b"
    assert loaded_checkpoint["checkpoint_metadata"]["schema_version"] == 3

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import scripts.experiments.run_two_stage_offline_pretraining as bridge_module
from src.models.thesis_multitask import ThesisMultitaskModel


def _build_model(training_phase: str) -> ThesisMultitaskModel:
    return ThesisMultitaskModel(
        input_dim=3,
        window_size=4,
        encoder_dim=8,
        hidden_dim=5,
        num_classes=2,
        continuous_num_prototypes=3,
        discrete_codebook_size=4,
        dropout=0.0,
        bootstrap_encoder_epochs=0,
        training_phase=training_phase,
        fusion_mode=(
            "direct_branch_routing"
            if training_phase == "stage_b_fusion_finetuning"
            else "learnable_sigmoid_scalars"
        ),
        use_synthetic_augmentation=False,
    )


def _normal_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 4, 3),
        "point_labels": torch.zeros(2, 4, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


@pytest.fixture
def bridge_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], ThesisMultitaskModel]:
    stage_a_model = _build_model("stage_a_multitask_pretraining")
    source_checkpoint_path = tmp_path / "stage_a_best.pt"
    torch.save(
        {
            "model_state_dict": stage_a_model.state_dict(),
            "extra_state": stage_a_model.get_checkpoint_extra_state(),
            "config": {"experiment_name": "stage-a-fixture"},
            "checkpoint_metadata": {},
            "epoch": 25,
            "metric_history": [],
        },
        source_checkpoint_path,
    )

    stage_b_model = _build_model("stage_b_fusion_finetuning")
    stage_b_config: dict[str, object] = {
        "experiment_name": "stage-b-fixture",
        "device": "cpu",
        "model": {
            "model_name": "thesis_multitask",
            "training_phase": "stage_b_fusion_finetuning",
            "fusion_mode": "direct_branch_routing",
        },
        "task": {"task_name": "multitask_tsad"},
        "data": {"dataset_name": "fixture"},
    }

    monkeypatch.setattr(bridge_module, "register_runtime_components", lambda: None)
    monkeypatch.setattr(
        bridge_module,
        "build_model_from_experiment_config",
        lambda config: stage_b_model,
    )
    monkeypatch.setattr(
        bridge_module,
        "build_dataset",
        lambda name, config: {"loaders": {"train": [_normal_batch()]}},
    )
    return source_checkpoint_path, stage_b_config, stage_b_model


def _run_bridge(
    fixture: tuple[Path, dict[str, object], ThesisMultitaskModel],
    output_path: Path,
) -> Path:
    source_path, stage_b_config, _ = fixture
    return bridge_module.prepare_stage_b_initialization_checkpoint(
        stage_b_config=stage_b_config,
        stage_a_checkpoint_path=source_path,
        initialization_checkpoint_path=output_path,
    )


@pytest.mark.parametrize(
    "allowed_key",
    ["discrete_assignment.weight", "discrete_assignment.bias"],
)
def test_bridge_allows_legacy_assignment_keys(
    bridge_fixture: tuple[Path, dict[str, object], ThesisMultitaskModel],
    tmp_path: Path,
    allowed_key: str,
) -> None:
    source_path, stage_b_config, _ = bridge_fixture
    payload = torch.load(source_path, map_location="cpu")
    payload["model_state_dict"][allowed_key] = torch.zeros_like(
        payload["model_state_dict"][allowed_key]
    )
    torch.save(payload, source_path)

    output_path = tmp_path / f"{allowed_key.rsplit('.', 1)[-1]}.pt"
    result = bridge_module.prepare_stage_b_initialization_checkpoint(
        stage_b_config=stage_b_config,
        stage_a_checkpoint_path=source_path,
        initialization_checkpoint_path=output_path,
    )

    assert result == output_path
    assert output_path.is_file()


def test_bridge_rejects_unexpected_checkpoint_keys(
    bridge_fixture: tuple[Path, dict[str, object], ThesisMultitaskModel],
    tmp_path: Path,
) -> None:
    source_path, stage_b_config, _ = bridge_fixture
    payload = torch.load(source_path, map_location="cpu")
    payload["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    torch.save(payload, source_path)

    with pytest.raises(RuntimeError, match="unexpected_keys"):
        bridge_module.prepare_stage_b_initialization_checkpoint(
            stage_b_config=stage_b_config,
            stage_a_checkpoint_path=source_path,
            initialization_checkpoint_path=tmp_path / "rejected.pt",
        )


def test_bridge_initializes_memory_banks(
    bridge_fixture: tuple[Path, dict[str, object], ThesisMultitaskModel],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "stage_b_init.pt"

    _run_bridge(bridge_fixture, output_path)

    payload = torch.load(output_path, map_location="cpu")
    assert payload["extra_state"]["memory_initialized"] is True
    assert payload["extra_state"]["verification_metadata_source"]


def test_bridge_output_loads_strictly(
    bridge_fixture: tuple[Path, dict[str, object], ThesisMultitaskModel],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "stage_b_init.pt"
    _, _, stage_b_model = bridge_fixture

    _run_bridge(bridge_fixture, output_path)

    loaded_payload = bridge_module.CheckpointManager(tmp_path).load_checkpoint(
        output_path,
        stage_b_model,
        strict=True,
    )
    assert loaded_payload["config"]["model"]["fusion_mode"] == ("direct_branch_routing")


def test_bridge_requires_a_normal_token(
    bridge_fixture: tuple[Path, dict[str, object], ThesisMultitaskModel],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, stage_b_model = bridge_fixture
    stage_b_model.use_synthetic_augmentation = True

    def all_anomalous_batch(batch: dict[str, object]) -> dict[str, object]:
        augmented_batch = dict(batch)
        batch_size, window_size = augmented_batch["x"].shape[:2]
        augmented_batch["classification_labels"] = torch.ones(
            batch_size,
            dtype=torch.long,
        )
        augmented_batch["synthetic_anomaly_mask"] = torch.ones(
            batch_size,
            window_size,
            dtype=torch.long,
        )
        return augmented_batch

    monkeypatch.setattr(
        stage_b_model.synthetic_anomaly_injector,
        "augment_batch",
        all_anomalous_batch,
    )
    monkeypatch.setattr(
        bridge_module,
        "build_dataset",
        lambda name, config: {
            "loaders": {
                "train": [
                    {
                        "x": torch.randn(1, 4, 3),
                        "point_labels": torch.ones(1, 4, dtype=torch.long),
                        "mask": None,
                        "timestamps": None,
                        "meta": [{"entity_id": "machine-a"}],
                    }
                ]
            }
        },
    )

    with pytest.raises(ValueError, match="at least one normal token"):
        _run_bridge(bridge_fixture, tmp_path / "no-normal-token.pt")

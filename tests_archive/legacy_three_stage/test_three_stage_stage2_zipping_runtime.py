from __future__ import annotations

import copy
from pathlib import Path

import torch

from scripts.run_three_stage_offline_pretraining import (
    _match_channel_signatures_by_cosine_similarity,
    _prepare_stage2_recovery_initialization_checkpoint,
    _zip_cnn_encoder_state_dicts_with_matches,
    materialize_three_stage_run_manifest,
)
from scripts.train import (
    build_model_from_experiment_config,
    register_runtime_components,
)
from src.core.config import load_experiment_config


def test_match_channel_signatures_uses_explicit_non_identity_pairing() -> None:
    classification_signatures = {
        "encoder.network.network.1": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
    }
    reconstruction_signatures = {
        "encoder.network.network.1": torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float32,
        )
    }

    matches = _match_channel_signatures_by_cosine_similarity(
        classification_signatures,
        reconstruction_signatures,
    )

    assert matches["encoder.network.network.1"] == [(0, 1), (1, 0)]


def test_zip_cnn_encoder_state_dicts_with_matches_reorders_next_layer_inputs() -> None:
    classification_state_dict = {
        "encoder.network.network.1.weight": torch.tensor(
            [[[1.0]], [[2.0]]],
            dtype=torch.float32,
        ),
        "encoder.network.network.1.bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "encoder.network.network.5.weight": torch.tensor(
            [
                [[1.0], [10.0]],
                [[2.0], [20.0]],
            ],
            dtype=torch.float32,
        ),
        "encoder.network.network.5.bias": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    reconstruction_state_dict = {
        "encoder.network.network.1.weight": torch.tensor(
            [[[2.0]], [[1.0]]],
            dtype=torch.float32,
        ),
        "encoder.network.network.1.bias": torch.tensor([0.2, 0.1], dtype=torch.float32),
        "encoder.network.network.5.weight": torch.tensor(
            [
                [[20.0], [2.0]],
                [[10.0], [1.0]],
            ],
            dtype=torch.float32,
        ),
        "encoder.network.network.5.bias": torch.tensor([2.0, 1.0], dtype=torch.float32),
    }
    channel_matches = {
        "encoder.network.network.1": [(0, 1), (1, 0)],
        "encoder.network.network.5": [(0, 1), (1, 0)],
    }

    zipped_encoder_state_dict = _zip_cnn_encoder_state_dicts_with_matches(
        classification_state_dict=classification_state_dict,
        reconstruction_state_dict=reconstruction_state_dict,
        channel_matches=channel_matches,
    )

    assert torch.allclose(
        zipped_encoder_state_dict["encoder.network.network.1.weight"],
        classification_state_dict["encoder.network.network.1.weight"],
    )
    assert torch.allclose(
        zipped_encoder_state_dict["encoder.network.network.5.weight"],
        classification_state_dict["encoder.network.network.5.weight"],
    )

    naive_identity_average = 0.5 * (
        classification_state_dict["encoder.network.network.5.weight"]
        + reconstruction_state_dict["encoder.network.network.5.weight"]
    )
    assert not torch.allclose(
        zipped_encoder_state_dict["encoder.network.network.5.weight"],
        naive_identity_average,
    )


def test_prepare_stage2_checkpoint_records_mtz_activation_matching_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")
    manifest = materialize_three_stage_run_manifest(experiment_config)
    stage_records_by_phase = {
        stage_record["phase_name"]: stage_record
        for stage_record in manifest["training_stages"]
    }

    register_runtime_components()

    for phase_name in ("stage1_classification", "stage1_reconstruction"):
        stage_config = load_experiment_config(
            stage_records_by_phase[phase_name]["config_path"]
        )
        model = build_model_from_experiment_config(stage_config)
        checkpoint_path = Path(
            stage_records_by_phase[phase_name]["best_checkpoint_path"]
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "scaler_state_dict": {"feature_mean": torch.zeros(38)},
                "config": stage_config,
                "epoch": 1,
                "metric_history": [],
            },
            checkpoint_path,
        )

    def _fake_build_stage2_mtz_encoder_state(
        classification_state_dict,  # type: ignore[no-untyped-def]
        reconstruction_state_dict,  # type: ignore[no-untyped-def]
        stage2_config,  # type: ignore[no-untyped-def]
        classification_config,  # type: ignore[no-untyped-def]
        reconstruction_config,  # type: ignore[no-untyped-def]
    ):
        del reconstruction_state_dict
        del stage2_config
        del classification_config
        del reconstruction_config
        encoder_state_dict = {
            key: value.clone()
            for key, value in classification_state_dict.items()
            if key.startswith("encoder.")
        }
        return encoder_state_dict, {
            "zipping_strategy": "mtz_approximation_activation_matching",
            "matching_policy": "greedy_cosine_channel_matching",
            "shared_scope": "encoder_only",
            "reused_head_policy": "stage1_task_specific_heads",
        }

    monkeypatch.setattr(
        "scripts.run_three_stage_offline_pretraining._build_stage2_mtz_approximation_encoder_state_dict",
        _fake_build_stage2_mtz_encoder_state,
    )

    initialization_checkpoint_path = _prepare_stage2_recovery_initialization_checkpoint(
        manifest
    )
    saved_payload = torch.load(initialization_checkpoint_path, map_location="cpu")

    assert saved_payload["extra_state"]["stage2_zip_metadata"] == {
        "zipping_strategy": "mtz_approximation_activation_matching",
        "matching_policy": "greedy_cosine_channel_matching",
        "shared_scope": "encoder_only",
        "reused_head_policy": "stage1_task_specific_heads",
    }
    assert saved_payload["extra_state"]["memory_initialized"] is False

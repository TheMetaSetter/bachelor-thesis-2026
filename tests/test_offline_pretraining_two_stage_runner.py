from __future__ import annotations

from pathlib import Path

from src.core.config import load_experiment_config
from src.models.thesis_multitask import ThesisMultitaskModel

from scripts.run_two_stage_offline_pretraining import (
    build_two_stage_training_plan,
    compute_two_stage_total_training_epochs,
    execute_two_stage_plan,
    materialize_two_stage_run_manifest,
)


def test_two_stage_training_plan_matches_80_plus_20_stage_contract() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp4/"
        "smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20"
        "__w20__seed11__rtx3090.yaml"
    )

    assert compute_two_stage_total_training_epochs(loaded_config["two_stage"]) == 100
    assert build_two_stage_training_plan(loaded_config) == [
        {
            "stage_name": "stage_a_multitask_pretraining",
            "epochs": 80,
            "global_epoch_start": 1,
            "global_epoch_end": 80,
        },
        {
            "stage_name": "stage_b_fusion_finetuning",
            "epochs": 20,
            "global_epoch_start": 81,
            "global_epoch_end": 100,
        },
    ]


def test_two_stage_manifest_and_stage_configs_use_stage_name_keys(tmp_path) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/"
        "smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20"
        "-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")

    manifest = materialize_two_stage_run_manifest(experiment_config)

    assert (
        manifest["training_stages"][0]["stage_name"] == "stage_a_multitask_pretraining"
    )
    assert manifest["training_stages"][1]["stage_name"] == "stage_b_fusion_finetuning"
    assert manifest["training_stages"][0]["global_epoch_start"] == 1
    assert manifest["training_stages"][1]["global_epoch_end"] == 5

    first_stage_config = load_experiment_config(
        manifest["training_stages"][0]["config_path"]
    )
    second_stage_config = load_experiment_config(
        manifest["training_stages"][1]["config_path"]
    )

    assert first_stage_config["stage_name"] == "stage_a_multitask_pretraining"
    assert first_stage_config["stage_global_epoch_start"] == 1
    assert first_stage_config["stage_global_epoch_end"] == 4
    assert first_stage_config["model"]["stage_name"] == "stage_a_multitask_pretraining"
    assert second_stage_config["stage_name"] == "stage_b_fusion_finetuning"
    assert second_stage_config["stage_global_epoch_start"] == 5
    assert second_stage_config["stage_global_epoch_end"] == 5
    assert second_stage_config["model"]["stage_name"] == "stage_b_fusion_finetuning"


def test_two_stage_execution_prepares_stage_b_after_stage_a_training(
    tmp_path, monkeypatch
) -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/thesis/exp4/"
        "smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20"
        "-smoke__w20__seed11__smoke.yaml"
    )
    experiment_config["output_dir"] = str(tmp_path / "outputs")
    experiment_config["checkpoint_dir"] = str(tmp_path / "outputs" / "checkpoints")
    manifest = materialize_two_stage_run_manifest(experiment_config)

    called_commands: list[list[str]] = []

    class _CompletedProcess:
        def __init__(self) -> None:
            self.returncode = 0

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        called_commands.append(list(args[0]))
        return _CompletedProcess()

    def _fake_prepare_stage_b_initialization_checkpoint(manifest_payload):  # type: ignore[no-untyped-def]
        checkpoint_path = Path(
            manifest_payload["training_stages"][1]["initialization_checkpoint_path"]
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"stage-b-init")
        return checkpoint_path

    monkeypatch.setattr(
        "scripts.run_two_stage_offline_pretraining.subprocess.run",
        _fake_run,
    )
    monkeypatch.setattr(
        "scripts.run_two_stage_offline_pretraining._prepare_stage_b_initialization_checkpoint",
        _fake_prepare_stage_b_initialization_checkpoint,
    )

    execution_report = execute_two_stage_plan(manifest, dry_run=False)

    assert len(called_commands) == 3
    assert called_commands[0][1].endswith("scripts/train.py")
    assert called_commands[0][3].endswith("01_stage_a_multitask_pretraining.yaml")
    assert called_commands[1][1].endswith("scripts/train.py")
    assert called_commands[1][3].endswith("02_stage_b_fusion_finetuning.yaml")
    assert called_commands[2][1].endswith("scripts/evaluate.py")
    assert execution_report["executed_stage_names"] == [
        "stage_a_multitask_pretraining",
        "stage_b_fusion_finetuning",
        "evaluation",
    ]
    assert Path(execution_report["stage_b_initialization_checkpoint_path"]).exists()


def test_thesis_multitask_two_stage_stages_switch_trainable_surface() -> None:
    stage_a_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=12,
        continuous_enabled=True,
        continuous_num_prototypes=32,
        discrete_enabled=True,
        discrete_codebook_size=60,
        training_phase="stage_a_multitask_pretraining",
        discrete_query_mode="cosine_topk",
        classification_label_mode="redlamp_multiclass",
    )
    stage_b_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=12,
        continuous_enabled=True,
        continuous_num_prototypes=32,
        discrete_enabled=True,
        discrete_codebook_size=60,
        training_phase="stage_b_fusion_finetuning",
        discrete_query_mode="cosine_topk",
        classification_label_mode="redlamp_multiclass",
        freeze_memories_after_initialization=True,
    )

    assert stage_a_model._phase_uses_prototype_path() is False
    assert stage_a_model._phase_uses_contrastive_objective() is True
    assert all(
        parameter.requires_grad for parameter in stage_a_model.encoder.parameters()
    )

    assert stage_b_model._phase_uses_prototype_path() is True
    assert stage_b_model._phase_uses_contrastive_objective() is False
    assert all(
        parameter.requires_grad is False
        for parameter in stage_b_model.encoder.parameters()
    )

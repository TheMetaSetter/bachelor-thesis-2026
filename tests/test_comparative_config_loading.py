from __future__ import annotations

from src.core.config import load_experiment_config, load_yaml_config
from scripts.run_comparative_smd_experiments import resolve_stage_family
from scripts.run_three_stage_offline_pretraining import (
    compute_three_stage_total_training_epochs,
)


OFFICIAL_ENTITY_IDS = ["machine-1-6", "machine-3-1", "machine-3-9"]
COMPARATIVE_SEEDS = [6, 36, 68]


def _entity_token(entity_id: str) -> str:
    return entity_id.replace("-", "_")


def _baseline_main_config_path(entity_id: str, seed: int) -> str:
    return (
        "configs/experiment/comparative/baseline/"
        f"smd__redlamp_baseline__comparative-single-stage-{_entity_token(entity_id)}"
        f"__w20__seed{seed}__main.yaml"
    )


def _thesis_main_config_path(entity_id: str, seed: int) -> str:
    return (
        "configs/experiment/comparative/thesis/"
        f"smd__thesis_multitask__comparative-three-stage-{_entity_token(entity_id)}"
        f"__w20__seed{seed}__main.yaml"
    )


def test_comparative_model_and_task_configs_lock_shared_semantics() -> None:
    baseline_model_config = load_yaml_config(
        "configs/model/redlamp_baseline_comparative_smd.yaml"
    )
    thesis_model_config = load_yaml_config(
        "configs/model/thesis_multitask_three_stage_comparative_smd.yaml"
    )
    task_config = load_yaml_config(
        "configs/task/multitask_tsad_redlamp_multiclass_window20_comparative.yaml"
    )

    assert baseline_model_config["model_name"] == "redlamp_baseline"
    assert baseline_model_config["encoder_family"] == "cnn_simple"
    assert baseline_model_config["lambda_recon"] == 0.9
    assert baseline_model_config["lambda_cls"] == 0.1
    assert baseline_model_config["use_label_refurbishment"] is True
    assert baseline_model_config["num_classes"] == 12

    assert thesis_model_config["model_name"] == "thesis_multitask"
    assert thesis_model_config["encoder_family"] == "cnn_simple"
    assert thesis_model_config["lambda_recon"] == 0.9
    assert thesis_model_config["lambda_cls"] == 0.1
    assert thesis_model_config["use_label_refurbishment"] is True
    assert thesis_model_config["num_classes"] == 12
    assert thesis_model_config["freeze_memories_after_initialization"] is True
    assert thesis_model_config["freeze_recovered_zipped_encoder_during_warmup"] is True

    assert task_config["classification_label_mode"] == "redlamp_multiclass"
    assert task_config["train_balance_classes"] is True
    assert task_config["use_synthetic_augmentation"] is True
    assert task_config["use_synthetic_validation"] is True
    assert "val_realistic" not in task_config
    assert "val_realistic_source" not in task_config


def test_all_comparative_main_configs_resolve_with_exact_stage_contracts() -> None:
    baseline_config_paths = [
        _baseline_main_config_path(entity_id, seed)
        for entity_id in OFFICIAL_ENTITY_IDS
        for seed in COMPARATIVE_SEEDS
    ]
    thesis_config_paths = [
        _thesis_main_config_path(entity_id, seed)
        for entity_id in OFFICIAL_ENTITY_IDS
        for seed in COMPARATIVE_SEEDS
    ]

    assert len(baseline_config_paths) == 9
    assert len(thesis_config_paths) == 9

    for config_path in baseline_config_paths:
        loaded_config = load_experiment_config(config_path)

        assert resolve_stage_family(loaded_config) == "baseline_single_stage"
        assert loaded_config["epochs"] == 300
        assert loaded_config["model"]["lambda_recon"] == 0.9
        assert loaded_config["model"]["lambda_cls"] == 0.1
        assert loaded_config["model"]["use_label_refurbishment"] is True
        assert loaded_config["task"]["train_balance_classes"] is True
        assert "val_realistic" not in loaded_config["task"]
        assert len(loaded_config["data"]["entity_ids"]) == 1
        assert loaded_config["data"]["window_size"] == 20
        assert loaded_config["data"]["stride"] == 1
        assert loaded_config["evaluation"]["vus_max_buffer_size"] == 20
        assert loaded_config["optimizer"]["scheduler"]["scheduler_name"] == "cosine"
        assert loaded_config["logging"]["use_wandb"] is True

    for config_path in thesis_config_paths:
        loaded_config = load_experiment_config(config_path)

        assert resolve_stage_family(loaded_config) == "thesis_three_stage"
        assert loaded_config["epochs"] == 300
        assert (
            compute_three_stage_total_training_epochs(loaded_config["three_stage"])
            == 300
        )
        assert loaded_config["three_stage"]["expected_total_training_epochs"] == 300
        assert loaded_config["model"]["lambda_recon"] == 0.9
        assert loaded_config["model"]["lambda_cls"] == 0.1
        assert loaded_config["model"]["use_label_refurbishment"] is True
        assert loaded_config["task"]["train_balance_classes"] is True
        assert "val_realistic" not in loaded_config["task"]
        assert len(loaded_config["data"]["entity_ids"]) == 1
        assert loaded_config["data"]["window_size"] == 20
        assert loaded_config["data"]["stride"] == 1
        assert loaded_config["evaluation"]["vus_max_buffer_size"] == 20
        assert loaded_config["optimizer"]["scheduler"]["scheduler_name"] == "cosine"
        assert loaded_config["logging"]["use_wandb"] is True


def test_comparative_smoke_configs_disable_online_wandb_and_reduce_runtime() -> None:
    smoke_config_paths = [
        "configs/experiment/comparative/baseline/"
        "smd__redlamp_baseline__comparative-single-stage-machine_1_6"
        "__w20__seed6__smoke.yaml",
        "configs/experiment/comparative/thesis/"
        "smd__thesis_multitask__comparative-three-stage-machine_1_6"
        "__w20__seed6__smoke.yaml",
    ]

    for config_path in smoke_config_paths:
        loaded_config = load_experiment_config(config_path)

        assert loaded_config["logging"]["use_wandb"] is False
        assert loaded_config["device"] == "cpu"
        assert loaded_config["data"]["num_workers"] == 0
        assert len(loaded_config["data"]["entity_ids"]) == 1

    baseline_smoke_config = load_experiment_config(smoke_config_paths[0])
    thesis_smoke_config = load_experiment_config(smoke_config_paths[1])

    assert baseline_smoke_config["epochs"] < 300
    assert thesis_smoke_config["epochs"] < 300
    assert (
        compute_three_stage_total_training_epochs(thesis_smoke_config["three_stage"])
        == thesis_smoke_config["epochs"]
    )


def test_comparative_stress_smoke_configs_use_cuda_and_nonzero_workers() -> None:
    stress_smoke_config_paths = [
        "configs/experiment/comparative_stress_smoke/baseline/"
        "smd__redlamp_baseline__comparative-single-stage-machine_1_6"
        "__w20__seed6__stress-smoke.yaml",
        "configs/experiment/comparative_stress_smoke/thesis/"
        "smd__thesis_multitask__comparative-three-stage-machine_1_6"
        "__w20__seed6__stress-smoke.yaml",
    ]

    baseline_stress_config = load_experiment_config(stress_smoke_config_paths[0])
    thesis_stress_config = load_experiment_config(stress_smoke_config_paths[1])

    for loaded_config in [baseline_stress_config, thesis_stress_config]:
        assert loaded_config["device"] == "cuda"
        assert loaded_config["data"]["num_workers"] > 0
        assert loaded_config["logging"]["use_wandb"] is True
        assert loaded_config["logging"]["wandb_mode"] == "offline"
        assert len(loaded_config["data"]["entity_ids"]) == 1

    assert baseline_stress_config["epochs"] < 300
    assert thesis_stress_config["epochs"] < 300
    assert (
        compute_three_stage_total_training_epochs(thesis_stress_config["three_stage"])
        == thesis_stress_config["epochs"]
    )

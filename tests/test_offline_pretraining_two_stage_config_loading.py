from __future__ import annotations

from src.core.config import load_experiment_config, validate_experiment_config


def test_two_stage_machine_3_4_experiment_config_matches_exact_100_epoch_stage_budget() -> (
    None
):
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp4/"
        "smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20"
        "__w20__seed11__rtx3090.yaml"
    )
    validate_experiment_config(loaded_config)

    assert loaded_config["data"]["entity_ids"] == ["machine-3-4"]
    assert loaded_config["epochs"] == 100
    assert loaded_config["two_stage"]["expected_total_training_epochs"] == 100
    assert loaded_config["two_stage"]["stage_a_multitask_epochs"] == 80
    assert loaded_config["two_stage"]["stage_b_fusion_finetuning_epochs"] == 20
    assert loaded_config["model"]["training_phase"] == "stage_a_multitask_pretraining"
    assert loaded_config["model"]["continuous_num_prototypes"] == 32
    assert loaded_config["model"]["discrete_codebook_size"] == 60
    assert loaded_config["model"]["discrete_query_mode"] == "cosine_topk"
    assert loaded_config["two_stage"]["freeze_encoder_and_memories_in_stage_b"] is True
    assert (
        loaded_config["two_stage"]["discrete_memory_label_source"]
        == "synthetic_train_labels"
    )


def test_two_stage_smoke_config_matches_exact_5_epoch_stage_budget() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp4/"
        "smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke"
        "__w20__seed11__smoke.yaml"
    )
    validate_experiment_config(loaded_config)

    assert loaded_config["epochs"] == 5
    assert loaded_config["two_stage"]["expected_total_training_epochs"] == 5
    assert loaded_config["two_stage"]["stage_a_multitask_epochs"] == 4
    assert loaded_config["two_stage"]["stage_b_fusion_finetuning_epochs"] == 1

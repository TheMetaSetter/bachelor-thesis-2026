from __future__ import annotations

from src.core.config import load_experiment_config, load_yaml_config


def test_machine_3_4_stride1_data_config_matches_target_contract() -> None:
    loaded_config = load_yaml_config("configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml")

    assert loaded_config["dataset_name"] == "smd"
    assert loaded_config["root_dir"] == "data/ServerMachineDataset"
    assert loaded_config["entity_ids"] == ["machine-3-4"]
    assert loaded_config["window_size"] == 20
    assert loaded_config["stride"] == 1
    assert loaded_config["batch_size"] == 256
    assert loaded_config["validation_split_ratio"] == 0.2


def test_three_stage_machine_3_4_experiment_config_matches_exact_300_epoch_budget() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml"
    )

    assert loaded_config["data"]["entity_ids"] == ["machine-3-4"]
    assert loaded_config["data"]["window_size"] == 20
    assert loaded_config["data"]["stride"] == 1
    assert loaded_config["epochs"] == 300
    assert loaded_config["three_stage"]["expected_total_training_epochs"] == 300
    assert loaded_config["three_stage"]["stage1_classification_epochs"] == 50
    assert loaded_config["three_stage"]["stage1_reconstruction_epochs"] == 70
    assert loaded_config["three_stage"]["stage2_recovery_epochs"] == 20
    assert loaded_config["three_stage"]["stage3_prototype_warmup_epochs"] == 20
    assert loaded_config["three_stage"]["multitask_pretraining_epochs"] == 140
    assert loaded_config["model"]["training_phase"] == "multitask_pretraining"
    assert loaded_config["model"]["fusion_mode"] == "task_specific_concat_projection"
    assert loaded_config["model"]["discrete_query_mode"] == "cosine_topk"
    assert loaded_config["model"]["discrete_topk"] == 3
    assert loaded_config["model"]["discrete_query_temperature"] == 0.1
    assert loaded_config["model"]["continuous_num_prototypes"] == 16
    assert loaded_config["model"]["discrete_codebook_size"] == 60
    assert loaded_config["model"]["freeze_memories_after_initialization"] is True
    assert (
        loaded_config["model"]["freeze_recovered_zipped_encoder_during_warmup"] is True
    )
    assert loaded_config["model"]["discrete_memory_label_source"] == "synthetic_train_labels"

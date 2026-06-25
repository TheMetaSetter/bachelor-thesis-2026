from __future__ import annotations

from src.core.config import load_yaml_config


def test_official_smd_entity_data_configs_use_exactly_one_entity_each() -> None:
    expected_entity_ids_by_config_path = {
        "configs/data/smd_rtx3090_machine_1_6_20_stride1.yaml": ["machine-1-6"],
        "configs/data/smd_rtx3090_machine_3_1_20_stride1.yaml": ["machine-3-1"],
        "configs/data/smd_rtx3090_machine_3_9_20_stride1.yaml": ["machine-3-9"],
    }

    for config_path, expected_entity_ids in expected_entity_ids_by_config_path.items():
        loaded_config = load_yaml_config(config_path)

        assert loaded_config["dataset_name"] == "smd"
        assert loaded_config["root_dir"] == "data/ServerMachineDataset"
        assert loaded_config["entity_ids"] == expected_entity_ids
        assert len(loaded_config["entity_ids"]) == 1
        assert loaded_config["window_size"] == 20
        assert loaded_config["stride"] == 1
        assert loaded_config["batch_size"] == 256
        assert loaded_config["num_workers"] == 16
        assert loaded_config["validation_split_ratio"] == 0.2

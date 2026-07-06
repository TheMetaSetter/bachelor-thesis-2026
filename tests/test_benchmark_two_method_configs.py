from __future__ import annotations

from src.core.config import load_experiment_config, validate_experiment_config


BENCHMARK_SEEDS = (6, 8, 36)
BENCHMARK_ENTITY_IDS = ("machine_1_6", "machine_3_4", "machine_3_9")
BENCHMARK_METHODS = ("baseline", "thesis_base", "thesis_point_score")


def _benchmark_config_path(method: str, entity_id: str, seed: int) -> str:
    if method == "baseline":
        return (
            "configs/experiment/benchmark/baseline/"
            f"smd__redlamp_baseline__benchmark-{entity_id}__w20__seed{seed}__main.yaml"
        )
    if method == "thesis_point_score":
        return (
            "configs/experiment/benchmark/thesis/"
            f"smd__thesis_multitask__benchmark-two-stage-point-score-{entity_id}"
            f"__w20__seed{seed}__main.yaml"
        )
    return (
        "configs/experiment/benchmark/thesis/"
        f"smd__thesis_multitask__benchmark-two-stage-{entity_id}"
        f"__w20__seed{seed}__main.yaml"
    )


def test_benchmark_two_method_configs_share_the_same_windowing_contract() -> None:
    for method in BENCHMARK_METHODS:
        for entity_id in BENCHMARK_ENTITY_IDS:
            for seed in BENCHMARK_SEEDS:
                loaded_config = load_experiment_config(
                    _benchmark_config_path(method, entity_id, seed)
                )
                validate_experiment_config(loaded_config)

                assert loaded_config["seed"] == seed
                assert loaded_config["data"]["window_size"] == 20
                assert loaded_config["data"]["stride"] == 1
                assert loaded_config["data"]["train_stride"] == 1
                assert loaded_config["data"]["val_stride"] == 20
                assert loaded_config["data"]["test_stride"] == 20
                assert loaded_config["data"]["shuffle_train"] is False
                assert loaded_config["data"]["batch_size"] == 512
                assert loaded_config["checkpoint_monitor_metric"] == "val_synth_vus_pr"
                assert loaded_config["evaluation"]["vus_max_buffer_size"] == 20
                assert loaded_config["evaluation"]["vus_num_thresholds"] == 200
                assert loaded_config["logging"]["log_hard_prediction_ratio"] is True
                assert (
                    loaded_config["logging"]["log_row_normalized_confusion_matrix"]
                    is True
                )
                assert loaded_config["logging"][
                    "diagnostics_stages_for_classification"
                ] == [
                    "train",
                    "val_synth",
                ]
                if method == "thesis_base":
                    assert loaded_config["experiment_variant"] == "two_stage_base_v1"
                    assert loaded_config["model"]["model_name"] == "thesis_multitask"
                    assert (
                        loaded_config["two_stage"]["expected_total_training_epochs"]
                        == 100
                    )
                    assert loaded_config["two_stage"]["stage_a_multitask_epochs"] == 80
                    assert (
                        loaded_config["two_stage"]["stage_b_fusion_finetuning_epochs"]
                        == 20
                    )
                if method == "thesis_point_score":
                    assert (
                        loaded_config["experiment_variant"]
                        == "two_stage_point_score_supervised_v1"
                    )
                    assert loaded_config["model"]["model_name"] == "thesis_multitask"
                    assert loaded_config["model"]["enable_score_loss"] is True
                    assert (
                        loaded_config["model"]["score_loss_type"]
                        == "pointwise_balanced_reconstruction_score"
                    )

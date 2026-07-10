from __future__ import annotations

from pathlib import Path

from scripts.generate_smd_benchmark_configs import (
    BENCHMARK_ENTITY_IDS,
    BENCHMARK_SEEDS,
    BENCHMARK_VARIANTS,
    generate_thesis_offline_benchmark_configs,
)
from src.core.config import load_experiment_config, validate_experiment_config


def _config_path(variant: str, entity_id: str, seed: int, smoke: bool) -> str:
    mode = "smoke" if smoke else "main"
    return (
        "configs/experiment/offline_benchmark/thesis/"
        f"smd__thesis__offline__{variant}__{entity_id.replace('-', '_')}"
        f"__w20__seed{seed}__{mode}.yaml"
    )


def test_generate_thesis_offline_benchmark_configs_writes_all_expected_files() -> None:
    generated_paths = generate_thesis_offline_benchmark_configs()

    assert (
        len(generated_paths)
        == len(BENCHMARK_VARIANTS)
        * len(BENCHMARK_ENTITY_IDS)
        * len(BENCHMARK_SEEDS)
        * 2
    )
    for variant in BENCHMARK_VARIANTS:
        for entity_id in BENCHMARK_ENTITY_IDS:
            for seed in BENCHMARK_SEEDS:
                for smoke in (False, True):
                    config_path = Path(_config_path(variant, entity_id, seed, smoke))
                    assert config_path.exists()
                    loaded_config = load_experiment_config(config_path)
                    validate_experiment_config(loaded_config)
                    assert loaded_config["seed"] == seed
                    assert loaded_config["data"]["window_size"] == 20
                    assert loaded_config["data"]["stride"] == 1
                    assert loaded_config["data"]["train_stride"] == 1
                    assert loaded_config["data"]["val_stride"] == 20
                    assert loaded_config["data"]["test_stride"] == 20
                    assert loaded_config["checkpoint_monitor_metric"] == (
                        "val_synth_vus_pr"
                    )
                    assert loaded_config["two_stage"][
                        "discrete_memory_label_source"
                    ] == ("synthetic_train_labels")
                    if smoke:
                        assert loaded_config["device"] == "cpu"
                        assert loaded_config["epochs"] == 2
                    else:
                        assert loaded_config["device"] == "cuda"
                        assert loaded_config["epochs"] == 30
                        assert loaded_config["output_dir"].startswith(
                            "outputs/benchmark/smd/thesis/"
                        )
                    if variant == "O0":
                        assert (
                            loaded_config["experiment_variant"] == "two_stage_base_v1"
                        )
                        assert "model_overrides" not in loaded_config
                    else:
                        assert (
                            loaded_config["experiment_variant"]
                            == "two_stage_point_score_supervised_v1"
                        )
                        assert (
                            loaded_config["model_overrides"]["score_loss_type"]
                            == "pointwise_balanced_bce_logits"
                        )
                        assert (
                            loaded_config["model_overrides"]["score_loss_target"]
                            == "synthetic_anomaly_mask"
                        )

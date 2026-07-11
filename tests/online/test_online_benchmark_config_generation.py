from __future__ import annotations

from pathlib import Path

from scripts.generate_online_benchmark_configs import (
    BENCHMARK_ENTITY_IDS,
    BENCHMARK_OFFLINE_VARIANTS,
    BENCHMARK_ONLINE_VARIANTS,
    BENCHMARK_SEEDS,
    generate_thesis_online_benchmark_configs,
)
from src.core.config import load_experiment_config, validate_experiment_config


def _config_path(
    offline_variant: str,
    online_variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> str:
    mode = "smoke" if smoke else "main"
    return (
        "configs/experiment/online_benchmark/thesis/"
        f"smd__thesis__online__{offline_variant}_{online_variant}"
        f"__{entity_id.replace('-', '_')}__w20__seed{seed}__{mode}.yaml"
    )


def test_generate_thesis_online_benchmark_configs_writes_all_expected_files() -> None:
    generated_paths = generate_thesis_online_benchmark_configs()

    assert len(generated_paths) == (
        len(BENCHMARK_OFFLINE_VARIANTS)
        * len(BENCHMARK_ONLINE_VARIANTS)
        * len(BENCHMARK_ENTITY_IDS)
        * len(BENCHMARK_SEEDS)
        * 2
    )
    for offline_variant in BENCHMARK_OFFLINE_VARIANTS:
        for online_variant in BENCHMARK_ONLINE_VARIANTS:
            for entity_id in BENCHMARK_ENTITY_IDS:
                for seed in BENCHMARK_SEEDS:
                    for smoke in (False, True):
                        config_path = Path(
                            _config_path(
                                offline_variant,
                                online_variant,
                                entity_id,
                                seed,
                                smoke,
                            )
                        )
                        assert config_path.exists()
                        loaded_config = load_experiment_config(config_path)
                        validate_experiment_config(loaded_config)
                        assert loaded_config["seed"] == seed
                        assert loaded_config["data"]["window_size"] == 20
                        assert loaded_config["data"]["stride"] == 1
                        assert loaded_config["data"]["train_stride"] == 1
                        assert loaded_config["data"]["val_stride"] == 20
                        assert loaded_config["data"]["test_stride"] == 20
                        assert loaded_config["data"]["shuffle_train"] is False
                        assert loaded_config["task"]["target_param_group"] == (
                            "projector_params"
                        )
                        assert loaded_config["task"]["clean_stream_only"] is True
                        assert loaded_config["task"]["warm_start_projector"] is False
                        assert loaded_config["task"][
                            "reference_checkpoint_path"
                        ].endswith(
                            f"outputs/{'benchmark_smoke' if smoke else 'benchmark'}/smd/thesis/{offline_variant}/{entity_id.replace('-', '_')}/seed{seed}/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt"
                        )
                        if smoke:
                            assert loaded_config["device"] == "cpu"
                            assert loaded_config["task"]["max_online_steps"] == 16
                        else:
                            assert loaded_config["device"] == "cuda"
                            assert loaded_config["task"]["max_online_steps"] is None
                        assert (
                            loaded_config["model"]["model_name"] == "online_adaptation"
                        )
                        assert (
                            loaded_config["model"]["score_source"] == "projected_hidden"
                        )
                        assert loaded_config["optimizer"]["optimizer_name"] == "adamw"

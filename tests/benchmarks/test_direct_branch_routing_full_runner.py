from __future__ import annotations

from scripts.run_direct_branch_routing_full import (
    OFFLINE_VARIANTS,
    SEEDS,
    ENTITIES,
    build_run_configs,
    build_direct_experiment_config,
    build_stage_a_source_checkpoint_path,
    build_stage_b_initialization_checkpoint_path,
)


def test_full_runner_builds_all_18_baseline_configs() -> None:
    configs = build_run_configs()

    assert len(configs) == 18
    assert {path.name for path in configs} == {
        f"smd__thesis__offline__{variant}__{entity}__w20__seed{seed}__main.yaml"
        for variant in OFFLINE_VARIANTS
        for entity in ENTITIES
        for seed in SEEDS
    }


def test_direct_config_uses_bridge_checkpoint_and_output() -> None:
    baseline_config = build_run_configs()[0]

    direct_config = build_direct_experiment_config(baseline_config)

    assert direct_config["device"] == "cuda"
    assert direct_config["epochs"] == 5
    assert "two_stage" not in direct_config
    assert direct_config["model"]["training_phase"] == "stage_b_fusion_finetuning"
    assert direct_config["model"]["fusion_mode"] == "direct_branch_routing"
    assert direct_config["initialization_checkpoint_path"] == (
        "outputs/benchmark/smd/machine_1_6/seed6/"
        "thesis_direct_branch_routing_O0/offline/stage_b/"
        "initializations/stage_b_init.pt"
    )
    assert direct_config["output_dir"] == (
        "outputs/benchmark/smd/"
        "machine_1_6/seed6/thesis_direct_branch_routing_O0/offline/stage_b"
    )


def test_all_direct_configs_have_unique_stage_a_and_output_paths() -> None:
    direct_configs = [
        build_direct_experiment_config(config_path)
        for config_path in build_run_configs()
    ]

    initialization_paths = {
        str(build_stage_b_initialization_checkpoint_path(config))
        for config in direct_configs
    }
    stage_a_paths = {
        str(build_stage_a_source_checkpoint_path(config_path))
        for config_path in build_run_configs()
    }
    output_paths = {config["output_dir"] for config in direct_configs}

    assert len(initialization_paths) == 18
    assert len(stage_a_paths) == 18
    assert len(output_paths) == 18
    assert all(config["epochs"] == 5 for config in direct_configs)
    assert all("two_stage" not in config for config in direct_configs)
    assert all(
        config["model"]["fusion_mode"] == "direct_branch_routing"
        for config in direct_configs
    )

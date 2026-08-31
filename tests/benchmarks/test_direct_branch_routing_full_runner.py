from __future__ import annotations

from scripts.run_direct_branch_routing_full import (
    OFFLINE_VARIANTS,
    SEEDS,
    ENTITIES,
    build_run_configs,
    build_direct_experiment_config,
)


def test_full_runner_builds_all_18_baseline_configs() -> None:
    configs = build_run_configs()

    assert len(configs) == 18
    assert {
        path.name
        for path in configs
    } == {
        f"smd__thesis__offline__{variant}__{entity}__w20__seed{seed}__main.yaml"
        for variant in OFFLINE_VARIANTS
        for entity in ENTITIES
        for seed in SEEDS
    }


def test_direct_config_uses_matching_stage_a_best_checkpoint_and_output() -> None:
    baseline_config = build_run_configs()[0]

    direct_config = build_direct_experiment_config(baseline_config)

    assert direct_config["device"] == "cuda"
    assert direct_config["epochs"] == 5
    assert "two_stage" not in direct_config
    assert direct_config["model"]["training_phase"] == "stage_b_fusion_finetuning"
    assert direct_config["model"]["fusion_mode"] == "direct_branch_routing"
    assert direct_config["initialization_checkpoint_path"] == (
        "outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/"
        "stage_a_multitask_pretraining/checkpoints/best.pt"
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
        config["initialization_checkpoint_path"] for config in direct_configs
    }
    output_paths = {config["output_dir"] for config in direct_configs}

    assert len(initialization_paths) == 18
    assert len(output_paths) == 18
    assert all(config["epochs"] == 5 for config in direct_configs)
    assert all("two_stage" not in config for config in direct_configs)
    assert all(
        config["model"]["fusion_mode"] == "direct_branch_routing"
        for config in direct_configs
    )

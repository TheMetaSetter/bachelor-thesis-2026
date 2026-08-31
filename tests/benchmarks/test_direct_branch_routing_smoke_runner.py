from __future__ import annotations

from scripts.run_direct_branch_routing_smoke import build_smoke_experiment_config


def test_smoke_runner_builds_one_gpu_direct_routing_config() -> None:
    config = build_smoke_experiment_config()

    assert config["device"] == "cuda"
    assert config["epochs"] == 1
    assert "two_stage" not in config
    assert config["model"]["training_phase"] == "stage_b_fusion_finetuning"
    assert config["model"]["fusion_mode"] == "direct_branch_routing"
    assert config["initialization_checkpoint_path"] == (
        "outputs/benchmark_smoke/smd/machine_1_6/seed6/"
        "thesis_direct_branch_routing_O0/offline/stage_b/"
        "initializations/stage_b_init.pt"
    )
    assert config["output_dir"] == (
        "outputs/benchmark_smoke/smd/machine_1_6/seed6/"
        "thesis_direct_branch_routing_O0/offline/stage_b"
    )
    assert config["data"]["batch_size"] == 256
    assert config["data"]["max_train_windows"] == 2048
    assert config["data"]["max_val_windows"] == 2048
    assert config["data"]["max_test_windows"] == 2048
    assert config["logging"]["use_wandb"] is False
    assert config["logging"]["wandb_mode"] == "disabled"

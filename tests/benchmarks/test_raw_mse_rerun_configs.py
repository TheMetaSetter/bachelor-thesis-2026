from __future__ import annotations

from pathlib import Path

from src.core.config import load_experiment_config
from scripts.experiments.run_two_stage_offline_pretraining import materialize_two_stage_run_manifest


def test_generated_matrix_preserves_budget_and_raw_stage_contract(tmp_path):
    from scripts.ops.prepare_raw_mse_offline_rerun import prepare_configs

    paths = prepare_configs(tmp_path / "configs", "unit_raw_mse")
    assert len(paths) == 18
    for path in paths:
        config = load_experiment_config(path)
        assert config["reconstruction_loss_space"] == "raw_input"
        assert config["evaluation"]["score_space"] == "raw_input"
        assert config["evaluation"]["point_score_transform"] == "identity"
        assert config["two_stage"]["stage_a_multitask_epochs"] == 25
        assert config["two_stage"]["stage_b_fusion_finetuning_epochs"] == 5
        assert config["model"]["monte_carlo_samples"] == 10
    config["output_dir"] = str(tmp_path / "offline")
    manifest = materialize_two_stage_run_manifest(config)
    for stage in manifest["training_stages"]:
        stage_config = load_experiment_config(stage["config_path"])
        assert stage_config["reconstruction_loss_space"] == "raw_input"
        assert Path(stage_config["output_dir"]).parent == tmp_path / "offline"
        assert stage_config["checkpoint_dir"] == stage_config["output_dir"]
    assert manifest["evaluation"]["checkpoint_path"] == manifest["training_stages"][1]["best_checkpoint_path"]

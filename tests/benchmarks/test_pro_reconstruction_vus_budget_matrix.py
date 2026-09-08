from __future__ import annotations

from pathlib import Path

from scripts.benchmarks.generate_pro_reconstruction_vus_budget_matrix import (
    generate_configs,
)
from scripts.experiments.run_two_stage_offline_pretraining import (
    materialize_two_stage_run_manifest,
)
from src.core.config import load_experiment_config


def test_generator_writes_54_budget_specific_configs(tmp_path: Path) -> None:
    generated_paths = generate_configs(tmp_path / "generated_configs")

    assert len(generated_paths) == 54
    assert len(set(generated_paths)) == 54
    assert len({path.name for path in generated_paths}) == 54

    monitors = set()
    output_dirs = set()
    run_names = set()
    for config_path in generated_paths:
        config = load_experiment_config(config_path)
        monitors.add(config["checkpoint_monitor_metric"])
        output_dirs.add(config["output_dir"])
        run_names.add(config["logging"]["wandb_run_name"])
        assert config["reconstruction_loss_space"] == "normalized_input"
        assert config["evaluation"]["score_space"] == "normalized_input"
        assert config["data_overrides"]["num_workers"] == 10
        assert config["model_overrides"]["fusion_mode"] == "direct_branch_routing"

    assert monitors == {
        "val_synth_vus_pr_at_fpr_budget_0_001",
        "val_synth_vus_pr_at_fpr_budget_0_005",
        "val_synth_vus_pr_at_fpr_budget_0_01",
    }
    assert len(output_dirs) == 54
    assert len(run_names) == 54


def test_budget_configs_materialize_108_unique_stage_runs(tmp_path: Path) -> None:
    generated_paths = generate_configs(tmp_path / "generated_configs")
    stage_run_names = set()
    for config_path in generated_paths:
        config = load_experiment_config(config_path)
        manifest = materialize_two_stage_run_manifest(config)
        for stage in manifest["training_stages"]:
            stage_config = load_experiment_config(Path(stage["config_path"]))
            stage_run_names.add(stage_config["logging"]["wandb_run_name"])
            assert (
                stage_config["checkpoint_monitor_metric"]
                == config["checkpoint_monitor_metric"]
            )
            assert stage_config["model_overrides"]["fusion_mode"] == (
                "direct_branch_routing"
            )

    assert len(stage_run_names) == 108

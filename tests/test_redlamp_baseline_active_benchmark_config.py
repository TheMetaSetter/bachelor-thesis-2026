from __future__ import annotations

from scripts.train import build_model_from_experiment_config, register_runtime_components
from src.core.config import load_experiment_config
from src.models.redlamp_baseline import RedLampBaseline


def test_active_benchmark_baseline_config_resolves_to_redlamp_baseline() -> None:
    experiment_config = load_experiment_config(
        "configs/experiment/benchmark/baseline/"
        "smd__redlamp_mlp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml"
    )

    register_runtime_components()
    model = build_model_from_experiment_config(experiment_config)

    assert experiment_config["model"]["model_name"] == "redlamp_baseline"
    assert "redlamp_baseline" in experiment_config["experiment_name"]
    assert "redlamp_baseline" in experiment_config["output_dir"]
    assert "redlamp_baseline" in experiment_config["checkpoint_dir"]
    assert "redlamp_baseline" in experiment_config["logging"]["wandb_run_name"]
    assert isinstance(model, RedLampBaseline)

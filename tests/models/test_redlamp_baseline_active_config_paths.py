from __future__ import annotations

import pytest

from src.core.config import load_experiment_config, load_yaml_config


@pytest.mark.parametrize(
    (
        "experiment_config_path",
        "expected_model_config_path",
        "expected_runtime_token",
    ),
    [
        (
            "configs/experiment/baseline/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed36__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed68__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__smoke.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_3_1__w20__seed6__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_3_1__w20__seed36__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_3_1__w20__seed68__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_3_9__w20__seed6__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_3_9__w20__seed36__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_3_9__w20__seed68__main.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/comparative_stress_smoke/baseline/"
            "smd__redlamp_baseline__comparative-single-stage-machine_1_6__w20__seed6__stress-smoke.yaml",
            "configs/model/redlamp_baseline_comparative_smd.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/scale/"
            "anomaly_archive__redlamp_baseline__staffiii-window20-adamw-cosine-"
            "warmup10-vus-pr-confmat__w20__seed11__default.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/scale/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-"
            "adamw-cosine-alt__w20__seed11__default.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/scale/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-"
            "adamw-cosine-lr1e-3__w20__seed11__default.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/scale/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-"
            "adamw-cosine-val-vus-pr__w20__seed68__default.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/scale/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-"
            "adamw-cosine-val-vus-pr-gradconf__w20__seed68__default.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/scale/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-"
            "adamw-cosine-val-vus-pr-gradconf-redlamp-aligned__w20__seed68__default.yaml",
            "configs/model/redlamp_baseline_redlamp_aligned.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/smoke/"
            "anomaly_archive__redlamp_baseline__staffiii-window20__w20__seed11__smoke.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
        (
            "configs/experiment/smoke/"
            "smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-"
            "adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
            "configs/model/redlamp_baseline.yaml",
            "redlamp_baseline",
        ),
    ],
)
def test_active_redlamp_experiment_configs_use_canonical_runtime_surface(
    experiment_config_path: str,
    expected_model_config_path: str,
    expected_runtime_token: str,
) -> None:
    root_config = load_yaml_config(experiment_config_path)
    resolved_config = load_experiment_config(experiment_config_path)

    assert root_config["model_config_path"] == expected_model_config_path
    assert expected_runtime_token in root_config["experiment_name"]
    assert expected_runtime_token in root_config["output_dir"]
    assert expected_runtime_token in root_config["checkpoint_dir"]
    assert expected_runtime_token in root_config["logging"]["wandb_run_name"]
    assert "redlamp_mlp_baseline" not in root_config["experiment_name"]
    assert "redlamp_mlp_baseline" not in root_config["output_dir"]
    assert "redlamp_mlp_baseline" not in root_config["checkpoint_dir"]
    assert "redlamp_mlp_baseline" not in root_config["logging"]["wandb_run_name"]
    assert resolved_config["model"]["model_name"] == "redlamp_baseline"

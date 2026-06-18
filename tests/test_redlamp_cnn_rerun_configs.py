from __future__ import annotations

from src.core.config import load_experiment_config, validate_experiment_config
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline


def _build_model_from_experiment_config(
    experiment_config: dict[str, object],
) -> RedLampMLPBaseline:
    model_kwargs = dict(experiment_config["model"])
    model_kwargs.pop("model_name", None)
    model_kwargs.update(
        {
            key: value
            for key, value in experiment_config["task"].items()
            if key != "task_name"
        }
    )
    return RedLampMLPBaseline(**model_kwargs)


def test_redlamp_cnn_rerun_configs_load_and_enable_balanced_sampling() -> None:
    experiment_config_paths = [
        "configs/experiment/baseline/smd__redlamp_cnn_baseline__redlamp-cnn-baseline-window20-balanced__w20__seed11__default.yaml",
        "configs/experiment/scale/smd__redlamp_cnn_baseline__redlamp-cnn-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr__w20__seed68__default.yaml",
        "configs/experiment/smoke/smd__redlamp_cnn_baseline__redlamp-cnn-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
    ]

    for experiment_config_path in experiment_config_paths:
        experiment_config = load_experiment_config(experiment_config_path)
        validate_experiment_config(experiment_config)
        model = _build_model_from_experiment_config(experiment_config)

        assert model.encoder_family == "cnn_simple"
        assert model.synthetic_anomaly_injector.train_balance_classes is True
        assert model.synthetic_validation_injector.train_balance_classes is True

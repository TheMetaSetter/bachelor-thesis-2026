from __future__ import annotations

from src.core.config import load_experiment_config, validate_experiment_config
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline
from src.models.thesis_multitask import ThesisMultitaskModel


def test_baseline_and_thesis_models_default_to_enabled_label_refurbishment() -> None:
    baseline_model = RedLampMLPBaseline(
        input_dim=38,
        window_size=20,
    )
    thesis_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=32,
        num_classes=12,
    )

    assert baseline_model.use_label_refurbishment is True
    assert thesis_model.use_label_refurbishment is True


def test_baseline_and_thesis_models_default_to_shared_loss_weights() -> None:
    baseline_model = RedLampMLPBaseline(
        input_dim=38,
        window_size=20,
    )
    thesis_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=32,
        num_classes=12,
    )

    assert baseline_model.lambda_recon == 0.9
    assert baseline_model.lambda_cls == 0.1
    assert thesis_model.lambda_recon == 0.9
    assert thesis_model.lambda_cls == 0.1


def test_redlamp_aligned_experiment_configs_preserve_shared_semantics() -> None:
    experiment_config_paths = [
        (
            "configs/experiment/scale/"
            "smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-"
            "window20-adamw-cosine-val-vus-pr-gradconf-redlamp-aligned__w20__seed68__default.yaml",
            "redlamp_mlp_baseline",
        ),
        (
            "configs/experiment/thesis/exp3/"
            "smd__thesis_multitask__thesis-multitask-redlamp-multiclass-"
            "window20-redlamp-aligned__w20__seed11__default.yaml",
            "thesis_multitask",
        ),
    ]

    for experiment_config_path, expected_model_name in experiment_config_paths:
        experiment_config = load_experiment_config(experiment_config_path)
        validate_experiment_config(experiment_config)

        assert experiment_config["model"]["model_name"] == expected_model_name
        assert experiment_config["model"]["lambda_recon"] == 0.9
        assert experiment_config["model"]["lambda_cls"] == 0.1
        assert experiment_config["model"]["refurbishment_alpha"] == 0.1
        assert experiment_config["model"]["refurbishment_beta"] == 0.01
        assert experiment_config["task"]["classification_label_mode"] == (
            "redlamp_multiclass"
        )
        assert experiment_config["task"]["train_balance_classes"] is True
        if expected_model_name == "redlamp_mlp_baseline":
            assert (
                experiment_config["model"]["enable_gradient_conflict_profiling"] is True
            )
        else:
            assert (
                experiment_config["model"]["enable_gradient_conflict_profiling"] is True
            )


def test_redlamp_aligned_model_and_thesis_model_build_with_shared_task_semantics() -> (
    None
):
    baseline_config = load_experiment_config(
        "configs/experiment/scale/"
        "smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-"
        "window20-adamw-cosine-val-vus-pr-gradconf-redlamp-aligned__w20__seed68__default.yaml"
    )
    thesis_config = load_experiment_config(
        "configs/experiment/thesis/exp3/"
        "smd__thesis_multitask__thesis-multitask-redlamp-multiclass-"
        "window20-redlamp-aligned__w20__seed11__default.yaml"
    )

    baseline_model_kwargs = dict(baseline_config["model"])
    baseline_model_kwargs.pop("model_name", None)
    baseline_model_kwargs.update(
        {
            key: value
            for key, value in baseline_config["task"].items()
            if key != "task_name"
        }
    )
    thesis_model_kwargs = dict(thesis_config["model"])
    thesis_model_kwargs.pop("model_name", None)
    thesis_model_kwargs.update(
        {
            key: value
            for key, value in thesis_config["task"].items()
            if key != "task_name"
        }
    )

    baseline_model = RedLampMLPBaseline(**baseline_model_kwargs)
    thesis_model = ThesisMultitaskModel(**thesis_model_kwargs)

    assert baseline_model.lambda_recon == 0.9
    assert baseline_model.lambda_cls == 0.1
    assert baseline_model.refurbishment_alpha == 0.1
    assert baseline_model.refurbishment_beta == 0.01
    assert baseline_model.synthetic_anomaly_injector.train_balance_classes is True
    assert baseline_model.enable_gradient_conflict_profiling is True
    assert thesis_model.lambda_recon == 0.9
    assert thesis_model.lambda_cls == 0.1
    assert thesis_model.refurbishment_alpha == 0.1
    assert thesis_model.refurbishment_beta == 0.01
    assert thesis_model.synthetic_anomaly_injector.train_balance_classes is True
    assert thesis_model.enable_gradient_conflict_profiling is True

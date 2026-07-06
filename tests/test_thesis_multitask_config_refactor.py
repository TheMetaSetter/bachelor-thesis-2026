from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.train import (
    build_model_from_experiment_config,
    register_runtime_components,
)
from src.core.registry import clear_registry
from src.engine.checkpoint import CheckpointManager
from src.models.thesis_multitask import ThesisMultitaskModel


def _get_config_class() -> type:
    # Import lazily so pytest can report the planned API as a normal failing
    # assertion before the refactor introduces the configuration dataclasses.
    from src.models import thesis_multitask

    config_class = getattr(thesis_multitask, "ThesisMultitaskModelConfig", None)
    assert config_class is not None, (
        "ThesisMultitaskModelConfig must be defined in src/models/thesis_multitask.py"
    )
    return config_class


def _flat_model_kwargs(**overrides: object) -> dict[str, object]:
    model_kwargs: dict[str, object] = {
        "input_dim": 38,
        "window_size": 100,
        "encoder_dim": 64,
        "hidden_dim": 16,
        "mlp_num_linear_layers": 3,
        "num_classes": 2,
        "dropout": 0.0,
        "continuous_enabled": True,
        "continuous_num_prototypes": 4,
        "discrete_enabled": True,
        "discrete_codebook_size": 8,
        "gumbel_temperature": 1.5,
        "temperature_start": 1.5,
        "temperature_end": 0.7,
        "temperature_anneal_fraction": 0.8,
        "temperature_hold_fraction": 0.1,
        "alpha_logit_init": 0.25,
        "beta_logit_init": -0.25,
        "use_label_refurbishment": True,
        "refurbishment_alpha": 0.2,
        "refurbishment_beta": 0.1,
        "reconstruction_normal_only": True,
        "lambda_recon": 0.9,
        "lambda_cls": 1.25,
        "enable_diversity_loss": False,
        "enable_variance_loss": False,
        "enable_covariance_loss": False,
        "enable_usage_loss": False,
        "enable_gate_loss": False,
        "lambda_div": 0.0,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "lambda_use": 0.03,
        "lambda_gate": 0.0,
        "enable_score_loss": False,
        "score_loss_granularity": "point",
        "score_loss_type": "pointwise_balanced_bce_logits",
        "score_loss_target": "synthetic_anomaly_mask",
        "score_loss_normalization": "train_batch_normal_tokens_detached_mean_std",
        "score_loss_reduction": "pointwise_binary_balanced_mean",
        "usage_lambda_schedule_fraction": 0.75,
        "variance_floor_gamma": 1.1,
        "gate_barrier_margin": 0.2,
        "bootstrap_encoder_epochs": 2,
        "discrete_ema_decay": 0.98,
        "memory_norm_epsilon": 1.0e-5,
        "memory_initialization_batches": 3,
        "memory_initialization_with_synthetic_windows": False,
        "training_phase": "multitask_pretraining",
        "fusion_mode": "task_specific_concat_projection",
        "discrete_query_mode": "cosine_topk",
        "discrete_topk": 3,
        "discrete_query_temperature": 0.1,
        "freeze_memories_after_initialization": True,
        "freeze_recovered_zipped_encoder_during_warmup": True,
        "discrete_memory_label_source": "synthetic_train_labels",
        "use_synthetic_augmentation": False,
        "use_synthetic_validation": True,
        "synthetic_validation_seed": 11,
        "freeze_fusion_for_epochs": 1,
        "warmup_alpha_value": 0.4,
        "warmup_beta_value": 0.6,
        "anomaly_probability": 0.75,
        "min_segment_fraction": 0.15,
        "max_segment_fraction": 0.25,
        "spike_scale": 4.0,
        "train_balance_classes": True,
        "anomaly_families": ["spike", "noise", "scale"],
    }
    model_kwargs.update(overrides)
    return model_kwargs


def _build_batch() -> dict[str, Any]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def _assert_common_model_contract(model: ThesisMultitaskModel) -> None:
    outputs = model(_build_batch())

    assert outputs["hidden"].shape == (2, 100, 16)
    assert outputs["pooled"].shape == (2, 100 * 16)
    assert outputs["recon"].shape == (2, 100, 38)
    assert outputs["logits"].shape == (2, 2)
    assert outputs["point_scores"].shape == (2, 100)
    assert outputs["window_scores"].shape == (2,)
    assert "continuous_branch" in outputs["aux"]
    assert "discrete_branch" in outputs["aux"]
    assert "fusion" in outputs["aux"]


def test_flat_kwargs_are_grouped_into_readable_config_sections() -> None:
    config_class = _get_config_class()

    config = config_class.from_flat_kwargs(_flat_model_kwargs())

    assert is_dataclass(config)
    assert config.architecture.input_dim == 38
    assert config.architecture.window_size == 100
    assert config.architecture.encoder_dim == 64
    assert config.architecture.hidden_dim == 16
    assert config.architecture.mlp_num_linear_layers == 3
    assert config.prototypes.continuous_num_prototypes == 4
    assert config.prototypes.discrete_codebook_size == 8
    assert config.prototypes.gumbel_temperature == 1.5
    assert config.schedule.temperature_start == 1.5
    assert config.schedule.temperature_end == 0.7
    assert config.schedule.freeze_fusion_for_epochs == 1
    assert config.objective.use_label_refurbishment is True
    assert config.objective.reconstruction_normal_only is True
    assert config.objective.lambda_recon == 0.9
    assert config.objective.lambda_cls == 1.25
    assert config.objective.enable_score_loss is False
    assert config.objective.score_loss_granularity == "point"
    assert config.objective.score_loss_type == "pointwise_balanced_bce_logits"
    assert config.objective.score_loss_target == "synthetic_anomaly_mask"
    assert config.memory.bootstrap_encoder_epochs == 2
    assert config.memory.memory_initialization_batches == 3
    assert config.synthetic.use_synthetic_augmentation is False
    assert config.synthetic.synthetic_validation_seed == 11
    assert config.synthetic.anomaly_families == ("spike", "noise", "scale")
    assert config.runtime.training_phase == "multitask_pretraining"
    assert config.runtime.fusion_mode == "task_specific_concat_projection"
    assert config.runtime.discrete_query_mode == "cosine_topk"
    assert config.runtime.discrete_topk == 3
    assert config.runtime.discrete_query_temperature == 0.1
    assert config.runtime.freeze_memories_after_initialization is True
    assert config.runtime.freeze_recovered_zipped_encoder_during_warmup is True
    assert config.runtime.discrete_memory_label_source == "synthetic_train_labels"


def test_config_sections_are_immutable_after_construction() -> None:
    config_class = _get_config_class()
    config = config_class.from_flat_kwargs(_flat_model_kwargs())

    with pytest.raises(FrozenInstanceError):
        config.architecture.hidden_dim = 32

    with pytest.raises(FrozenInstanceError):
        config.synthetic.anomaly_families = ("spike",)


def test_unknown_flat_kwargs_fail_before_model_construction() -> None:
    config_class = _get_config_class()
    bad_kwargs = _flat_model_kwargs(unexpected_constructor_key=True)

    with pytest.raises(ValueError, match="unexpected_constructor_key"):
        config_class.from_flat_kwargs(bad_kwargs)


def test_model_rejects_mixed_config_object_and_flat_kwargs() -> None:
    config_class = _get_config_class()
    config = config_class.from_flat_kwargs(_flat_model_kwargs())

    with pytest.raises(ValueError, match="either config or flat keyword"):
        ThesisMultitaskModel(config, input_dim=38)


def test_config_object_and_flat_kwargs_build_equivalent_runtime_contracts() -> None:
    config_class = _get_config_class()
    flat_kwargs = _flat_model_kwargs()

    torch.manual_seed(123)
    flat_model = ThesisMultitaskModel(**flat_kwargs)
    torch.manual_seed(123)
    config_model = ThesisMultitaskModel(config_class.from_flat_kwargs(flat_kwargs))

    assert flat_model.hidden_dim == config_model.hidden_dim == 16
    assert flat_model.mlp_num_linear_layers == config_model.mlp_num_linear_layers == 3
    assert (
        flat_model.continuous_num_prototypes == config_model.continuous_num_prototypes
    )
    assert flat_model.discrete_codebook_size == config_model.discrete_codebook_size
    assert flat_model.use_synthetic_augmentation is False
    assert tuple(config_model.synthetic_anomaly_injector.anomaly_families) == (
        "spike",
        "noise",
        "scale",
    )
    assert list(flat_model.state_dict().keys()) == list(
        config_model.state_dict().keys()
    )

    _assert_common_model_contract(flat_model)
    _assert_common_model_contract(config_model)


def test_usage_lambda_defaults_still_fall_back_to_lambda_use() -> None:
    config_class = _get_config_class()
    kwargs = _flat_model_kwargs(
        lambda_use=0.07,
        usage_lambda_start=None,
        usage_lambda_end=None,
    )

    model = ThesisMultitaskModel(config_class.from_flat_kwargs(kwargs))

    assert model.usage_lambda_start == pytest.approx(0.07)
    assert model.usage_lambda_end == pytest.approx(0.07)
    assert model.current_usage_lambda == pytest.approx(0.07)


def test_registry_runtime_still_accepts_flat_resolved_experiment_config() -> None:
    clear_registry()
    register_runtime_components()
    experiment_config = {
        "model": {
            "model_name": "thesis_multitask",
            **_flat_model_kwargs(),
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": False,
            "use_synthetic_validation": True,
            "synthetic_validation_seed": 11,
            "freeze_fusion_for_epochs": 1,
            "warmup_alpha_value": 0.4,
            "warmup_beta_value": 0.6,
            "anomaly_probability": 0.75,
            "min_segment_fraction": 0.15,
            "max_segment_fraction": 0.25,
            "spike_scale": 4.0,
            "train_balance_classes": True,
            "anomaly_families": ["spike", "noise", "scale"],
        },
    }

    model = build_model_from_experiment_config(experiment_config)

    assert isinstance(model, ThesisMultitaskModel)
    assert model.hidden_dim == 16
    assert model.use_synthetic_augmentation is False
    _assert_common_model_contract(model)


def test_checkpoint_style_flat_config_reconstructs_same_state_dict_keys() -> None:
    config_class = _get_config_class()
    saved_config = {
        "model": {
            "model_name": "thesis_multitask",
            **_flat_model_kwargs(),
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": False,
            "use_synthetic_validation": True,
            "synthetic_validation_seed": 11,
            "freeze_fusion_for_epochs": 1,
            "warmup_alpha_value": 0.4,
            "warmup_beta_value": 0.6,
            "anomaly_probability": 0.75,
            "min_segment_fraction": 0.15,
            "max_segment_fraction": 0.25,
            "spike_scale": 4.0,
            "train_balance_classes": True,
            "anomaly_families": ["spike", "noise", "scale"],
        },
    }
    checkpoint_kwargs = {
        key: value
        for key, value in saved_config["model"].items()
        if key != "model_name"
    }
    checkpoint_kwargs.update(
        {
            key: value
            for key, value in saved_config["task"].items()
            if key != "task_name"
        }
    )

    flat_model = ThesisMultitaskModel(**checkpoint_kwargs)
    config_model = ThesisMultitaskModel(
        config_class.from_flat_kwargs(checkpoint_kwargs)
    )

    assert list(flat_model.state_dict().keys()) == list(
        config_model.state_dict().keys()
    )
    assert (
        flat_model.get_memory_lifecycle_state()
        == config_model.get_memory_lifecycle_state()
    )


def test_config_object_survives_training_step_and_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    config_class = _get_config_class()
    kwargs = _flat_model_kwargs(
        bootstrap_encoder_epochs=0,
        use_synthetic_augmentation=False,
        use_label_refurbishment=False,
        reconstruction_normal_only=False,
    )
    model = ThesisMultitaskModel(config_class.from_flat_kwargs(kwargs))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    step_output = model.training_step(_build_batch())
    optimizer.zero_grad()
    step_output["loss"].backward()
    optimizer.step()

    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="config_refactor_roundtrip.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={
            "feature_mean": torch.zeros(38),
            "feature_std": torch.ones(38),
        },
        config={
            "model": {"model_name": "thesis_multitask", **kwargs},
            "task": {"task_name": "multitask_tsad"},
        },
        epoch=1,
        metric_history=[{"train_loss": float(step_output["loss"].detach())}],
        extra_state=model.get_checkpoint_extra_state(),
    )

    reloaded_model = ThesisMultitaskModel(config_class.from_flat_kwargs(kwargs))
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1.0e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
    )

    for parameter, reloaded_parameter in zip(
        model.parameters(),
        reloaded_model.parameters(),
    ):
        assert torch.allclose(parameter, reloaded_parameter)
    assert loaded_checkpoint["extra_state"] == model.get_checkpoint_extra_state()

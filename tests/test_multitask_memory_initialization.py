from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_initialization_model() -> ThesisMultitaskModel:
    model = ThesisMultitaskModel(
        input_dim=4,
        window_size=3,
        encoder_dim=4,
        hidden_dim=4,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=2,
        discrete_enabled=True,
        discrete_codebook_size=3,
        gumbel_temperature=1.0,
        temperature_start=1.0,
        temperature_end=1.0,
        temperature_anneal_fraction=1.0,
        alpha_logit_init=0.0,
        beta_logit_init=0.0,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        bootstrap_encoder_epochs=1,
        memory_initialization_batches=1,
        memory_initialization_with_synthetic_windows=True,
        use_synthetic_augmentation=True,
        use_synthetic_validation=False,
        anomaly_probability=1.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
    )

    def _identity_encoder_forward(
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hidden = batch["x"]
        return {
            "hidden": hidden,
            "pooled": hidden.mean(dim=1),
            "aux": {"encoder_name": "identity"},
        }

    model.encoder.forward = _identity_encoder_forward  # type: ignore[method-assign]
    return model


def _build_raw_batch() -> dict[str, object]:
    x_tensor = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
        ]
    )
    return {
        "x": x_tensor,
        "point_labels": torch.zeros(2, 3, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def _build_synthetic_batch(raw_batch: dict[str, object]) -> dict[str, object]:
    synthetic_batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in raw_batch.items()
    }
    synthetic_batch["classification_labels"] = torch.tensor([0, 1], dtype=torch.long)
    synthetic_batch["synthetic_anomaly_mask"] = torch.zeros(2, 3, dtype=torch.long)
    synthetic_batch["synthetic_anomaly_mask"][1, 1] = 1
    synthetic_batch["augmentation_metadata"] = [
        {"is_synthetic_anomaly": False},
        {"is_synthetic_anomaly": True},
    ]
    synthetic_batch["x"][1, 1] = torch.tensor([9.0, 9.0, 9.0, 9.0])
    return synthetic_batch


def test_initialization_pool_keeps_only_normal_synthetic_timesteps() -> None:
    model = _build_initialization_model()
    raw_batch = _build_raw_batch()
    anomaly_vector = torch.tensor([9.0, 9.0, 9.0, 9.0])

    model.synthetic_anomaly_injector.augment_batch = (  # type: ignore[method-assign]
        lambda batch: _build_synthetic_batch(batch)
    )

    token_pool = model._collect_memory_initialization_token_pool_from_loader(
        [raw_batch],
        device="cpu",
    )

    assert token_pool["num_batches_used"] == 1
    assert token_pool["num_clean_tokens"] == 6
    assert token_pool["num_synthetic_normal_tokens"] == 5
    assert token_pool["hidden_tokens"].shape == (11, 4)
    assert not torch.any(
        torch.all(token_pool["hidden_tokens"] == anomaly_vector, dim=1)
    )


def test_memory_initialization_marks_model_initialized_and_reseeds_buffers() -> None:
    model = _build_initialization_model()
    raw_batch = _build_raw_batch()
    initial_memory_state = model.get_memory_tensor_state()

    model.synthetic_anomaly_injector.augment_batch = (  # type: ignore[method-assign]
        lambda batch: _build_synthetic_batch(batch)
    )
    model.set_epoch_context(epoch_index=1, total_epochs=2)

    was_initialized = model.maybe_initialize_memories_from_loader(
        [raw_batch],
        device="cpu",
    )

    assert was_initialized is True
    assert model.memory_initialized is True
    assert model.memory_training_enabled is True
    assert model.memory_ready_for_initialization is False
    assert model.memory_initialization_epoch == 2
    assert not torch.equal(
        initial_memory_state["continuous_prototype_bank"],
        model.continuous_prototype_bank,
    )
    assert not torch.equal(
        initial_memory_state["discrete_codebook"],
        model.discrete_codebook,
    )
    assert torch.allclose(
        model.continuous_prototype_bank.norm(dim=-1),
        torch.ones(2),
        atol=1.0e-5,
    )
    assert torch.allclose(
        model.discrete_codebook.norm(dim=-1),
        torch.ones(3),
        atol=1.0e-5,
    )

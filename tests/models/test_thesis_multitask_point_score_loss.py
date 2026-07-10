from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_model(enable_score_loss: bool) -> ThesisMultitaskModel:
    return ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=12,
        continuous_enabled=True,
        continuous_num_prototypes=32,
        discrete_enabled=True,
        discrete_codebook_size=60,
        training_phase="stage_a_multitask_pretraining",
        discrete_query_mode="cosine_topk",
        classification_label_mode="redlamp_multiclass",
        use_synthetic_augmentation=False,
        use_synthetic_validation=False,
        enable_score_loss=enable_score_loss,
        score_loss_granularity="point",
        score_loss_type="pointwise_balanced_bce_logits",
        score_loss_target="synthetic_anomaly_mask",
        score_loss_normalization="train_batch_normal_tokens_detached_mean_std",
        score_loss_reduction="pointwise_binary_balanced_mean",
    )


def _build_batch() -> dict[str, torch.Tensor | list[dict[str, str]] | None]:
    return {
        "x": torch.tensor(
            [
                [[0.0] for _ in range(20)],
                [[1.0] for _ in range(20)],
            ],
            dtype=torch.float32,
        ).repeat(1, 1, 38),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-3-4"}, {"entity_id": "machine-3-4"}],
        "classification_labels": torch.tensor([0, 1], dtype=torch.long),
        "synthetic_anomaly_mask": torch.tensor(
            [
                [0] * 10 + [1] * 10,
                [0] * 5 + [1] * 15,
            ],
            dtype=torch.long,
        ),
        "augmentation_metadata": [
            {
                "is_synthetic_anomaly": False,
                "anomaly_family": "clean",
                "anomaly_family_index": None,
                "start_index": None,
                "end_index": None,
                "affected_channels": [],
                "family_parameters_by_channel": {},
            },
            {
                "is_synthetic_anomaly": True,
                "anomaly_family": "spike",
                "anomaly_family_index": 0,
                "start_index": 5,
                "end_index": 20,
                "affected_channels": [0],
                "family_parameters_by_channel": {},
            },
        ],
    }


def test_stage_a_point_score_loss_is_enabled_only_for_point_score_variant() -> None:
    base_model = _build_model(enable_score_loss=False)
    base_step = base_model.training_step(_build_batch())
    assert torch.is_tensor(base_step["loss_terms"]["score_loss"])
    assert base_step["loss_terms"]["score_loss"].item() == 0.0

    point_score_model = _build_model(enable_score_loss=True)
    point_score_step = point_score_model.training_step(_build_batch())

    assert point_score_step["loss_terms"]["score_loss"].item() > 0.0
    assert point_score_step["log"]["train_score_loss"] > 0.0
    assert point_score_step["log"]["train_score_loss_skipped_batches"] == 0.0

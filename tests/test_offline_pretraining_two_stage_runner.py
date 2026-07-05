from __future__ import annotations

from src.core.config import load_experiment_config
from src.models.thesis_multitask import ThesisMultitaskModel

from scripts.run_two_stage_offline_pretraining import (
    build_two_stage_training_plan,
    compute_two_stage_total_training_epochs,
)


def test_two_stage_training_plan_matches_80_plus_20_stage_contract() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp4/"
        "smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20"
        "__w20__seed11__rtx3090.yaml"
    )

    assert compute_two_stage_total_training_epochs(loaded_config["two_stage"]) == 100
    assert build_two_stage_training_plan(loaded_config) == [
        {
            "phase_name": "stage_a_multitask_pretraining",
            "epochs": 80,
            "global_epoch_start": 1,
            "global_epoch_end": 80,
        },
        {
            "phase_name": "stage_b_fusion_finetuning",
            "epochs": 20,
            "global_epoch_start": 81,
            "global_epoch_end": 100,
        },
    ]


def test_thesis_multitask_two_stage_stages_switch_trainable_surface() -> None:
    stage_a_model = ThesisMultitaskModel(
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
    )
    stage_b_model = ThesisMultitaskModel(
        input_dim=38,
        window_size=20,
        encoder_dim=64,
        hidden_dim=16,
        num_classes=12,
        continuous_enabled=True,
        continuous_num_prototypes=32,
        discrete_enabled=True,
        discrete_codebook_size=60,
        training_phase="stage_b_fusion_finetuning",
        discrete_query_mode="cosine_topk",
        classification_label_mode="redlamp_multiclass",
        freeze_memories_after_initialization=True,
    )

    assert stage_a_model._phase_uses_prototype_path() is False
    assert stage_a_model._phase_uses_contrastive_objective() is True
    assert all(parameter.requires_grad for parameter in stage_a_model.encoder.parameters())

    assert stage_b_model._phase_uses_prototype_path() is True
    assert stage_b_model._phase_uses_contrastive_objective() is False
    assert all(
        parameter.requires_grad is False
        for parameter in stage_b_model.encoder.parameters()
    )

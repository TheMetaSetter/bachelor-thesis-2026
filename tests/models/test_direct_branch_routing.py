from __future__ import annotations

from pathlib import Path

import torch

from src.core.config import load_experiment_config
from src.engine.checkpoint import CheckpointManager
from src.models.thesis_multitask import ThesisMultitaskModel


def _build_direct_model(*, stochastic_inference: bool = False) -> ThesisMultitaskModel:
    return ThesisMultitaskModel(
        input_dim=3,
        window_size=4,
        encoder_dim=8,
        hidden_dim=5,
        num_classes=2,
        continuous_num_prototypes=3,
        discrete_codebook_size=4,
        stochastic_inference=stochastic_inference,
        training_phase="stage_b_fusion_finetuning",
        fusion_mode="direct_branch_routing",
    )


def _build_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 4, 3),
        "point_labels": torch.zeros(2, 4, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def test_direct_mode_routes_each_branch_to_its_matching_task() -> None:
    model = _build_direct_model()
    continuous_hidden = torch.ones(2, 4, 5)
    discrete_hidden = torch.full((2, 4, 5), 2.0)

    fusion_outputs = model._compute_fusion_outputs(
        continuous_hidden=continuous_hidden,
        discrete_hidden=discrete_hidden,
    )

    assert torch.equal(fusion_outputs["hidden_reconstruction"], continuous_hidden)
    assert torch.equal(fusion_outputs["hidden_classification"], discrete_hidden)
    assert fusion_outputs["aux"]["fusion_mode"] == "direct_branch_routing"
    assert torch.equal(fusion_outputs["alpha"], torch.zeros(2))
    assert torch.equal(fusion_outputs["beta"], torch.zeros(2))


def test_direct_mode_routes_each_sampled_branch_to_its_matching_task() -> None:
    model = _build_direct_model()
    continuous_samples = torch.ones(2, 3, 4, 5)
    discrete_samples = torch.full((2, 3, 4, 5), 2.0)

    sampled_reconstruction, sampled_classification = model._build_sampled_fusion_hidden(
        continuous_samples,
        discrete_samples,
        alpha=torch.zeros(2),
        beta=torch.zeros(2),
    )

    assert torch.equal(sampled_reconstruction, continuous_samples)
    assert torch.equal(sampled_classification, discrete_samples)


def test_direct_stage_b_freezes_legacy_fusion_modules() -> None:
    model = _build_direct_model()

    for module in (
        model.reconstruction_concat_projection,
        model.classification_concat_projection,
        model.reconstruction_fusion_gate,
        model.classification_fusion_gate,
    ):
        assert all(
            parameter.requires_grad is False for parameter in module.parameters()
        )
    assert model.alpha_logit.requires_grad is False
    assert model.beta_logit.requires_grad is False


def test_direct_forward_keeps_branch_identity_before_task_heads() -> None:
    model = _build_direct_model()

    outputs = model(_build_batch())

    assert torch.equal(
        outputs["aux"]["hidden_reconstruction"],
        outputs["aux"]["continuous_branch"]["prototype_context"],
    )
    assert torch.equal(
        outputs["aux"]["hidden_classification"],
        outputs["aux"]["discrete_branch"]["quantized_hidden"],
    )


def test_direct_eval_uses_direct_routing_for_monte_carlo_outputs() -> None:
    model = _build_direct_model(stochastic_inference=True)
    model.eval()

    outputs = model(_build_batch())

    assert outputs["aux"]["stochastic_query"]["enabled"] is True
    assert outputs["aux"]["uncertainty"] is not None
    assert outputs["aux"]["fusion"]["fusion_mode"] == "direct_branch_routing"
    assert outputs["recon"].shape == (2, 4, 3)
    assert outputs["logits"].shape == (2, 2)


def test_direct_model_loads_legacy_fusion_checkpoint_strictly(tmp_path: Path) -> None:
    legacy_model = ThesisMultitaskModel(
        input_dim=3,
        window_size=4,
        encoder_dim=8,
        hidden_dim=5,
        num_classes=2,
        continuous_num_prototypes=3,
        discrete_codebook_size=4,
        stochastic_inference=False,
        training_phase="stage_b_fusion_finetuning",
        fusion_mode="task_specific_concat_projection",
        discrete_query_mode="cosine_topk",
        discrete_topk=3,
    )
    direct_model = ThesisMultitaskModel(
        input_dim=3,
        window_size=4,
        encoder_dim=8,
        hidden_dim=5,
        num_classes=2,
        continuous_num_prototypes=3,
        discrete_codebook_size=4,
        stochastic_inference=False,
        training_phase="stage_b_fusion_finetuning",
        fusion_mode="direct_branch_routing",
        discrete_query_mode="cosine_topk",
        discrete_topk=3,
    )
    config = {
        "experiment_name": "direct-branch-routing-checkpoint-test",
        "model": {"model_name": "thesis_multitask"},
        "task": {"task_name": "multitask_tsad"},
    }
    checkpoint_manager = CheckpointManager(tmp_path)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="legacy.pt",
        model=legacy_model,
        optimizer=None,
        scheduler=None,
        scaler_state={},
        config=config,
        epoch=1,
        metric_history=[],
    )

    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        direct_model,
        strict=True,
    )

    assert (
        loaded_checkpoint["model_state_dict"].keys() == direct_model.state_dict().keys()
    )


def test_direct_stage_b_config_is_standalone() -> None:
    config_path = (
        Path(__file__).parents[2] / "configs/experiment/offline_ablation/thesis/"
        "smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml"
    )

    experiment_config = load_experiment_config(config_path)

    assert experiment_config["epochs"] == 5
    assert experiment_config["model"]["fusion_mode"] == "direct_branch_routing"
    assert experiment_config["model"]["training_phase"] == ("stage_b_fusion_finetuning")
    assert experiment_config["output_dir"].endswith(
        "outputs/benchmark/smd/machine_1_6/seed6/thesis_direct_branch_routing/offline/stage_b"
    )
    assert experiment_config["initialization_checkpoint_path"].endswith(
        "outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt"
    )
    assert "two_stage" not in experiment_config


def test_direct_stage_b_runs_one_forward_and_backward_step() -> None:
    model = _build_direct_model()

    step_output = model.training_step(_build_batch())
    step_output["loss"].backward()

    assert torch.isfinite(step_output["loss"])
    assert any(
        parameter.grad is not None
        for parameter in model.reconstruction_head.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.classification_head.parameters()
    )
    for module in (
        model.reconstruction_concat_projection,
        model.classification_concat_projection,
        model.reconstruction_fusion_gate,
        model.classification_fusion_gate,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert model.alpha_logit.grad is None
    assert model.beta_logit.grad is None

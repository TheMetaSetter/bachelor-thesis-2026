from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.config import load_yaml
from src.methods.mustard.losses import compute_loss
from src.methods.mustard.model import Mustard
from src.methods.mustard.retrieval import retrieve


ROOT = Path(__file__).resolve().parents[1]


def configuration(variant):
    phase = "offline" if variant.startswith("O") else "online"
    path = ROOT / f"configs/method_profiles/mustard_{variant.lower()}_{phase}.yaml"
    method = load_yaml(path)
    return {
        "project_root": ROOT,
        "method": method,
        "variant": variant,
        "phase": phase,
        "seed": 6,
        "device": "cpu",
        "data": {"window_size": 2},
        "prepared_identity": {"features": ["x", "y"]},
        "encoder": {
            "input_features": 2,
            "hidden_channels": 2,
            "output_dim": 2,
            "kernel_size": 1,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "evaluation": {"score_space": "latent"},
    }


def test_dense_sparse_and_no_retrieval():
    hidden = torch.tensor([[[1.0, 0.0]]])
    bank = torch.eye(2)
    settings = {
        "continuous": {"temperature": 1.0},
        "discrete": {"temperature": 1.0, "top_k": 1},
        "memory": {"normalization_epsilon": 1e-6},
    }
    result = retrieve(hidden, {"continuous": bank, "discrete": bank}, settings)
    weights = torch.tensor([2**-0.5, 0.0]).softmax(0)
    torch.testing.assert_close(result["continuous"][0, 0], F.normalize(weights, dim=0))
    torch.testing.assert_close(result["discrete"], hidden)
    torch.testing.assert_close(retrieve(hidden, {}, settings)["continuous"], hidden)


def test_initialization_uses_kmeans_and_routes_each_branch():
    model = Mustard(configuration("O0"), encoder=nn.Identity())
    model.parameters_config["continuous"]["num_prototypes"] = 1
    model.parameters_config["discrete"]["codebook_size"] = 12
    model.parameters_config["discrete"]["top_k"] = 1
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[-1.0, 0.0], [-1.0, 0.0]]])
    model._initialize_memory(
        tokens, torch.tensor([0, 1]), torch.tensor([[False, False], [True, False]])
    )
    torch.testing.assert_close(
        model.memory["continuous"], F.normalize(torch.tensor([[0.5, 0.5]]), dim=-1)
    )
    model._set_stage("stage_b")
    model.network.eval()
    result = model._forward(tokens[:1])
    torch.testing.assert_close(
        result["reconstruction"], model.reconstruction_head(result["continuous"])
    )
    torch.testing.assert_close(
        result["logits"], model.classification_head(result["discrete"].flatten(1))
    )


@pytest.mark.parametrize("mask", [[[False, True]], [[False, False]], [[True, True]]])
def test_o1_point_score_loss_reduction(mask):
    parameters = configuration("O1")["method"]["parameters"]
    x = torch.zeros(1, 2, 2)
    logits = torch.zeros(1, 12)
    outputs = {
        "reconstruction": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        "logits": logits,
    }
    targets = {
        "x": x,
        "classification_labels": torch.tensor([0]),
        "synthetic_anomaly_mask": torch.tensor(mask),
    }
    losses = compute_loss(
        outputs, targets, parameters | {"variant": "O1", "stage": "stage_a"}
    )
    classification = torch.log(torch.tensor(12.0))
    if any(mask[0]) and not all(mask[0]):
        assert "point_score" in losses
        torch.testing.assert_close(
            losses["classification_branch"],
            (classification + losses["point_score"]) / 2,
        )
    else:
        assert "point_score" not in losses
        torch.testing.assert_close(losses["classification_branch"], classification)


def test_a0_does_not_update():
    model = Mustard(configuration("A0"), encoder=nn.Identity())
    result = model.update({"x": torch.ones(1, 2, 2)}, {"decision": "pnn_verified"})
    assert result["did_update"] is False and model.optimizer is None


def test_stage_b_freezes_encoder_and_preserves_stage_a_heads():
    model = Mustard(configuration("O0"))
    model._set_stage("stage_a")
    assert all(p.requires_grad for p in model.encoder.parameters())
    before = [p.detach().clone() for p in model.reconstruction_head.parameters()]
    model._set_stage("stage_b")
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.reconstruction_head.parameters())
    for current, old in zip(
        model.reconstruction_head.parameters(), before, strict=True
    ):
        torch.testing.assert_close(current, old)


def test_empty_verification_mask_and_nonoverlap():
    model = Mustard(configuration("A1"), encoder=nn.Identity())
    batch = {"x": torch.ones(1, 2, 2), "indices": torch.tensor([[2, 3]])}
    empty = model.update(
        batch, {"decision": "pnn_verified", "mask": torch.zeros(1, 2, dtype=torch.bool)}
    )
    assert empty["did_update"] is False
    model.update(batch, {"decision": "gray_zone"})
    model.update(batch | {"indices": torch.tensor([[3, 4]])}, {"decision": "gray_zone"})
    assert len(model.verification_buffer) == 1


def test_state_restores_weights_and_adaptation_buffer():
    cfg = configuration("A1")
    model = Mustard(cfg)
    x = torch.ones(1, 2, 2)
    model.update(
        {"x": x, "indices": torch.tensor([[10, 11]])}, {"decision": "gray_zone"}
    )
    saved = model.state_dict()
    other = Mustard(cfg)
    other.load_state_dict(saved)
    assert len(other.verification_buffer) == 1
    for actual, expected in zip(
        other.encoder.parameters(), model.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    assert torch.equal(other.rng.get_state(), model.rng.get_state())


def test_fit_keeps_each_checkpoint_and_restores_source(tmp_path):
    import numpy as np

    cfg = configuration("O1")
    cfg["data"] = {
        "window_size": 12,
        "batch_size": 12,
        "train_stride": 12,
        "validation_stride": 12,
        "tail_policy": {"train": "drop", "validation": "drop"},
        "shuffle_train": False,
        "gap_policy": "preserve_source_rows",
        "expected_interval_seconds": None,
    }
    cfg["limits"] = {"max_epochs_per_stage": 1}
    cfg["synthetic"] = load_yaml(
        ROOT / "configs/data_profiles/synthetic_profiles/redlamp_legacy.yaml"
    )
    rng = np.random.default_rng(71)
    data = rng.normal(size=(144, 2))
    splits = [
        {"x": data, "identity": {"dataset": "smd", "entity": "a", "split": name}}
        for name in ("train", "validation")
    ]
    model = Mustard(cfg)
    fit = model.fit(*splits, checkpoint_dir=tmp_path)
    assert [stage["name"] for stage in fit["stages"]] == ["stage_a", "stage_b"]
    assert all(
        Path(stage[key]).is_file()
        for stage in fit["stages"]
        for key in ("initial_checkpoint", "best_checkpoint")
    )
    online = configuration("A0") | {"data": cfg["data"]}
    online["method"]["required_offline_variant"] = "O1"
    child = Mustard(online)
    child.load_state_dict(model.state_dict())
    window = {"x": data[:12][None]}
    assert child.score(window)["raw_scores"].shape == (1, 12)


def test_a1_changes_only_projector():
    model = Mustard(configuration("A1"), encoder=nn.Identity())
    model.reconstruction_head = nn.Identity()
    model.memory.update(
        continuous=torch.eye(2),
        discrete=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
    )
    before = {
        name: value.detach().clone() for name, value in model.network.named_parameters()
    }
    batch = {"x": torch.tensor([[[0.5, 1.0], [1.0, 0.5]]])}
    result = model.update(
        batch,
        {
            "decision": "pnn_verified",
            "recorded_score": 0.1,
            "mask": torch.tensor([[True, False]]),
        },
    )
    assert result["did_update"]
    changed = {
        name
        for name, value in model.network.named_parameters()
        if not torch.equal(value, before[name])
    }
    assert changed and all(name.startswith("projector.") for name in changed)


def test_uncertainty_uses_sample_axis_and_true_window_means():
    model = Mustard(configuration("O0"))
    reconstruction = [
        torch.tensor([[[0.0, 2.0], [2.0, 0.0]]]),
        torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
    ]
    passes = [
        {
            "continuous": value,
            "discrete": value,
            "reconstruction": value,
            "logits": torch.zeros((1, 12)),
        }
        for value in reconstruction
    ]
    errors = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    fields = [
        "point_anomaly_score_variance",
        "window_anomaly_score_variance",
        "reconstruction_variance_window",
    ]
    result = model._uncertainty(passes, errors, fields)
    assert result[fields[0]].tolist() == [[2.0, 2.0]]
    assert result[fields[1]].tolist() == [0.0]
    assert result[fields[2]].tolist() == [2.0]
    assert model._uncertainty(passes[:1], errors[:, :1], fields) == {}

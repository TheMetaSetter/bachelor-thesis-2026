from pathlib import Path

import numpy as np
import pytest
import torch

from src.methods.redlamp import (
    RedLamp,
    _cell_mask,
    _corrected_targets,
    _masked_reconstruction,
)


def test_redlamp_correction_and_masked_reconstruction():
    targets = _corrected_targets(
        torch.tensor([0, 1]), 12, {"alpha": 0.1, "beta": 0.01}, dtype=torch.float32
    )
    torch.testing.assert_close(targets.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(
        targets[:, :2], torch.tensor([[0.89, 0.01], [0.11, 0.79]])
    )
    record = [{"family_parameters_by_channel": {"1": {"changed_positions": [0]}}}]
    x = torch.zeros(1, 2, 2)
    mask = _cell_mask(x, record)
    assert mask.tolist() == [[[False, True], [False, False]]]
    loss = _masked_reconstruction(x, torch.full_like(x, 2.0), mask)
    assert loss == 3


def test_redlamp_frozen_scaling_and_endpoint_alignment():
    config = {
        "device": "cpu",
        "seed": 6,
        "phase": "online",
        "method": {
            "parameters": {
                "classifier_hidden_size": 3,
                "classifier_dropout": 0.0,
                "class_selection_probability": 0.05,
                "zero_range_epsilon": 1e-8,
            }
        },
        "encoder": {"output_dim": 2},
        "prepared_identity": {"features": ["x", "y"]},
    }
    model = RedLamp(config, encoder=torch.nn.Identity())
    probabilities = np.hstack(
        (np.full((2, 1), 0.5), np.full((2, 1), 0.5), np.zeros((2, 10)))
    )
    model._fit_score_calibration(np.ones(2), probabilities)
    original = model.state_dict()["calibration"]
    first = model.score({"x": np.zeros((1, 2, 2)), "indices": np.array([[2, 3]])})
    second = model.score({"x": np.ones((1, 2, 2)), "indices": np.array([[3, 4]])})
    assert first["indices"].tolist() == [3]
    assert second["indices"].tolist() == [4]
    assert (
        np.isfinite(first["raw_scores"]).all()
        and np.isfinite(second["raw_scores"]).all()
    )
    assert model.state_dict()["calibration"] == original
    with pytest.raises(ValueError, match="labels"):
        model.score({"x": np.zeros((1, 2, 2)), "labels": np.zeros((1, 2))})


def test_stumpy_native_distance_and_state_round_trip():
    from src.methods.stumpy_channel_ab import StumpyChannelAB

    config = {
        "data": {"window_size": 3},
        "method": {
            "parameters": {
                "normalize": False,
                "p": 2.0,
                "channel_aggregation": "max",
                "calibration_epsilon": 1e-8,
            }
        },
    }
    model = StumpyChannelAB(config)
    reference = np.array([0.0, 2.0, 4.0, 7.0, 11.0])[:, None]
    model.reference = [reference]
    model.center, model.scale = np.array([0.0]), np.array([1.0])
    batch = {"x": np.array([[[0.0], [2.0], [5.0]]]), "indices": np.array([[4, 5, 6]])}
    result = model.score(batch)
    assert result["indices"].tolist() == [6]
    assert result["raw_scores"].tolist() == pytest.approx([1.0])
    restored = StumpyChannelAB(config)
    restored.load_state_dict(model.state_dict())
    assert restored.score(batch)["raw_scores"] == pytest.approx(result["raw_scores"])
    with pytest.raises(ValueError, match="length"):
        model.score({"x": batch["x"][:, :2], "indices": batch["indices"][:, :2]})


def test_kmeans_known_distance_and_seeded_fit():
    from src.methods.kmeans_ad import KMeansAD

    config = {
        "seed": 6,
        "data": {"window_size": 2},
        "method": {
            "parameters": {
                "n_clusters": 1,
                "n_init": 2,
                "normalize_windows": False,
                "normalization_epsilon": 1e-6,
            }
        },
    }
    windows = np.array([[[0.0], [0.0]], [[2.0], [2.0]]])
    a, b = KMeansAD(config), KMeansAD(config)
    a._fit_windows(windows)
    b._fit_windows(windows)
    np.testing.assert_allclose(a.centers, [[1.0, 1.0]])
    np.testing.assert_array_equal(a.centers, b.centers)
    a.center, a.scale = 0.0, 1.0
    other = KMeansAD(config)
    other.load_state_dict(a.state_dict())
    batch = {"x": windows, "indices": np.array([[0, 1], [2, 3]])}
    assert other.score(batch)["indices"].tolist() == [1, 3]
    assert other.score(batch)["raw_scores"] == pytest.approx([2**0.5] * 2)


def test_isolation_forest_score_direction_seed_and_checkpoint(tmp_path):
    from src.artifacts import load_checkpoint, save_checkpoint
    from src.methods.iforest import IsolationForestAD

    parameters = {
        "n_estimators": 9,
        "max_samples": "auto",
        "max_features": 1.0,
        "contamination": "auto",
        "normalize_windows": False,
        "normalization_epsilon": 1e-6,
    }
    config = {
        "seed": 6,
        "data": {"window_size": 2},
        "method": {"parameters": parameters},
    }
    values = np.random.default_rng(31).normal(size=(40, 2, 1))
    a, b = IsolationForestAD(config), IsolationForestAD(config)
    for model in (a, b):
        model._fit_windows(values)
        model.center, model.scale = 0.0, 1.0
    batch = {"x": values[:4], "indices": np.arange(8).reshape(4, 2)}
    result = a.score(batch)
    np.testing.assert_array_equal(result["raw_scores"], b.score(batch)["raw_scores"])
    np.testing.assert_array_equal(
        result["raw_scores"], -a.model.decision_function(values[:4].reshape(4, -1))
    )
    checkpoint = tmp_path / "forest.pt"
    save_checkpoint(
        checkpoint, a.state_dict(), identity={"method": "iforest"}, role="fit.best"
    )
    restored = IsolationForestAD(config)
    restored.load_state_dict(
        load_checkpoint(checkpoint, expected_identity={"method": "iforest"})
    )
    np.testing.assert_array_equal(
        result["raw_scores"], restored.score(batch)["raw_scores"]
    )
    assert result["indices"].tolist() == [1, 3, 5, 7]


def _candi_config():
    from src.config import load_yaml

    profile = load_yaml(
        Path(__file__).resolve().parents[1]
        / "configs/method_profiles/baselines/candi.yaml"
    )
    profile["parameters"]["minimum_samples"] = 2
    profile["parameters"]["sana"] |= {
        "hidden_size": 4,
        "attention_heads": 2,
        "feedforward_size": 8,
    }
    return {
        "method": profile,
        "phase": "online",
        "device": "cpu",
        "seed": 6,
        "data": {"window_size": 2},
        "prepared_identity": {"features": ["x", "y"]},
        "encoder": {"output_dim": 2},
    }


def _candi_reference(model):
    latents = np.array([[1.0, 0.0]] * 6)
    model._fit_references(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), latents, 0.95)


def test_candi_covariance_empty_hard_set_and_pending_state():
    from src.methods.candi import CANDI

    cfg = _candi_config()
    model = CANDI(cfg, encoder=torch.nn.Identity())
    _candi_reference(model)
    vectors = np.array([1.0, 0.0])
    assert np.isfinite(model.references["inverse"]).all()
    assert not len(model.references["hard"])
    threshold = model.references["threshold"]
    assert model._select(threshold, vectors) is None
    assert model._select(threshold + 1, vectors) is None
    batch = {"x": np.ones((1, 2, 2)), "indices": np.array([[0, 1]])}
    context = {"recorded_score": 2.0, "latent": vectors}
    assert not model.update(batch, context)["did_update"]
    restored = CANDI(cfg, encoder=torch.nn.Identity())
    restored.load_state_dict(model.state_dict())
    assert len(restored.pending["moderate"]) == 1
    assert all(not p.requires_grad for p in restored.decoder.parameters())
    before = [p.detach().clone() for p in restored.decoder.parameters()]
    assert restored.update(batch, context)["did_update"]
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, restored.decoder.parameters(), strict=True)
    )


def test_candi_test_labels_stay_outside_method():
    from src.methods.candi import CANDI

    cfg = _candi_config()
    first = CANDI(cfg, encoder=torch.nn.Identity())
    second = CANDI(cfg, encoder=torch.nn.Identity())
    _candi_reference(first)
    second.load_state_dict(first.state_dict())
    results = []
    for model, label in ((first, 0), (second, 1)):
        batch = {
            "x": np.ones((1, 2, 2)),
            "indices": np.array([[0, 1]]),
            "labels": np.full((1, 2), label),
        }
        with pytest.raises(ValueError, match="labels"):
            model.score(batch)
        method_batch = {key: value for key, value in batch.items() if key != "labels"}
        scores = model.score(method_batch)
        update = model.update(
            method_batch, {"recorded_score": 2.0, "latent": np.array([1.0, 0.0])}
        )
        results.append((scores["raw_scores"], update))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_m2n2_trend_threshold_and_resume():
    from src.config import load_yaml
    from src.methods.m2n2 import M2N2
    from pathlib import Path

    profile = load_yaml(
        Path(__file__).resolve().parents[1]
        / "configs/method_profiles/baselines/m2n2.yaml"
    )
    profile["parameters"]["trend_decay"] = 0.5
    config = {
        "method": profile,
        "phase": "online",
        "device": "cpu",
        "seed": 6,
        "data": {"window_size": 2},
        "prepared_identity": {"features": ["x", "y"]},
        "encoder": {"output_dim": 2},
    }
    model = M2N2(config, encoder=torch.nn.Identity())
    for parameter in model.decoder.parameters():
        torch.nn.init.zeros_(parameter)
    batch = {"x": np.full((1, 2, 2), 2.0), "indices": np.array([[0, 1]])}
    result = model.score(batch)
    np.testing.assert_array_equal(result["raw_scores"], [[1.0, 1.0]])
    assert torch.equal(model.trend, torch.ones(2))
    model.adaptation_threshold = 1.0
    assert not model.update(batch, {"recorded_scores": result["raw_scores"]})[
        "did_update"
    ]
    assert torch.equal(model.trend, torch.ones(2))
    other = M2N2(config, encoder=torch.nn.Identity())
    other.load_state_dict(model.state_dict())
    batch["indices"] += 2
    first, second = model.score(batch), other.score(batch)
    np.testing.assert_array_equal(first["raw_scores"], second["raw_scores"])
    assert model.update(batch, {"recorded_scores": first["raw_scores"]})["did_update"]
    assert other.update(batch, {"recorded_scores": second["raw_scores"]})["did_update"]
    for a, b in zip(
        model.decoder.parameters(), other.decoder.parameters(), strict=True
    ):
        torch.testing.assert_close(a, b)


def _moment_config(variant, phase):
    from pathlib import Path
    from src.config import load_yaml

    root = Path(__file__).resolve().parents[1]
    suffix = "" if variant == "zero_shot" else "_linear_probing"
    return {
        "project_root": root,
        "method": load_yaml(
            root / f"configs/method_profiles/baselines/moment{suffix}.yaml"
        ),
        "variant": variant,
        "phase": phase,
        "device": "cpu",
        "seed": 6,
        "data": {"window_size": 512},
        "prepared_identity": {"features": ["x", "y"]},
    }


def test_moment_checkpoint_zero_shot_and_linear_probing():
    from src.methods.moment import Moment

    model = Moment(_moment_config("zero_shot", "online"))
    assert not any(parameter.requires_grad for parameter in model.model.parameters())
    t = np.arange(512) / 10
    batch = {
        "x": np.stack([np.sin(t), np.cos(t)], axis=1)[None],
        "indices": np.arange(512)[None],
    }
    score = model.score(batch)
    assert score["raw_scores"].shape == (1, 512)
    assert np.isfinite(score["raw_scores"]).all()
    assert not hasattr(model, "update")
    del model

    probe = Moment(_moment_config("linear_probing", "offline"))
    assert all(
        name.startswith("head.")
        for name, parameter in probe.model.named_parameters()
        if parameter.requires_grad
    )
    x = torch.as_tensor(batch["x"], dtype=torch.float32)
    loss = (probe._reconstruct(x) - x).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None for parameter in probe.model.head.parameters()
    )
    assert all(parameter.grad is None for parameter in probe.model.encoder.parameters())
    assert not hasattr(probe, "update")


_METHOD_ROOT = Path(__file__).resolve().parents[1]
_METHOD_PROFILES = sorted((_METHOD_ROOT / "configs/method_profiles").rglob("*.yaml"))


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            _assert_state_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_state_equal(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("path", _METHOD_PROFILES, ids=lambda path: path.stem)
def test_shared_constructor_state_round_trip(path, tmp_path):
    from src.artifacts import load_checkpoint, save_checkpoint
    from src.config import create_method, load_yaml, resolve_run

    profile = load_yaml(path)
    name = profile["method"]
    cell = {
        "data_profile": "data_profiles/smd.yaml",
        "method_profile": path.relative_to(_METHOD_ROOT / "configs").as_posix(),
        "evaluation_profile": f"evaluation_profiles/{'raw_input' if name == 'mustard' else 'native'}_q99_ewma075.yaml",
        "entity": "machine-1-1",
        "phase": profile["phases"][0],
        "seed": 6,
        "device": "cpu",
    }
    if name in {"mustard", "redlamp", "candi", "m2n2"}:
        cell["encoder"] = dict(
            hidden_channels=4, output_dim=4, kernel_size=3, num_layers=2, dropout=0.1
        )
    config = resolve_run(cell, project_root=_METHOD_ROOT)
    first = create_method(config)
    first_state = first.state_dict()
    del first
    checkpoint = tmp_path / "state.pt"
    identity = {"method": name, "variant": profile["variant"]}
    save_checkpoint(checkpoint, first_state, identity=identity, role="fit.initial")
    second = create_method(config)
    second.load_state_dict(
        load_checkpoint(
            checkpoint, expected_identity=identity | {"role": "fit.initial"}
        )
    )
    _assert_state_equal(first_state, second.state_dict())

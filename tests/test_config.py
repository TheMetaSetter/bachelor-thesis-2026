from copy import deepcopy
from pathlib import Path

import pytest

from src.config import list_names, load_yaml, resolve_run


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text", ["", "1", "[a, b]", "x: 1\nx: 2", "x:\n  y: 1\n  y: 2"]
)
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / "input.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="input.yaml"):
        load_yaml(path)


def standalone():
    return load_yaml(ROOT / "configs/standalones/smd_mustard.yaml")


def test_resolution_is_read_only():
    cell = standalone()
    before = deepcopy(cell)
    resolved = resolve_run(cell, project_root=ROOT)
    assert cell == before
    assert resolved["dataset"] == "smd"
    assert resolved["project_root"] == ROOT
    assert "run_id" not in resolved
    resolved["limits"]["max_epochs_per_stage"] = 1
    assert cell == before


@pytest.mark.parametrize(
    "reference", ["../outside.yaml", "/outside.yaml", "data_profiles/missing.yaml"]
)
def test_invalid_profile_reference(reference):
    cell = standalone()
    cell["data_profile"] = reference
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_run(cell, project_root=ROOT)


def test_unknown_field():
    cell = standalone()
    cell["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        resolve_run(cell, project_root=ROOT)


def test_supplied_offline_epoch_totals():
    paths = sorted((ROOT / "configs/method_profiles").rglob("*.yaml"))
    totals = {}
    for path in paths:
        method = load_yaml(path)
        if method.get("training") is not None:
            totals[method["method"], method["variant"]] = sum(
                stage["epochs"] for stage in method["training"]["stages"]
            )
    assert {method for method, _ in totals} == {
        "mustard",
        "redlamp",
        "candi",
        "m2n2",
        "moment",
    }
    assert set(totals.values()) == {30}


def test_discovery():
    assert set(list_names("datasets", project_root=ROOT)) == {
        "iccad",
        "smd",
        "iops",
        "nasa",
        "swat",
    }


def test_mustard_online_cell_can_choose_another_offline_variant():
    from src.config import create_method

    cell = load_yaml(ROOT / "configs/standalones/smd_mustard.yaml")
    cell.update(
        phase="online",
        method_profile="method_profiles/mustard_a1_online.yaml",
        required_offline_variant="O1",
    )
    config = resolve_run(cell, project_root=ROOT)
    assert config["required_offline_variant"] == "O1"
    assert config["source_method"]["variant"] == "O1"
    assert create_method(config).source_variant == "O1"


def test_method_resolution_rejects_an_optimizer_the_method_does_not_use(monkeypatch):
    from src import config

    read_yaml = config.load_yaml

    def wrong_optimizer(path):
        settings = read_yaml(path)
        if path.name == "mustard_o0_offline.yaml":
            settings["training"]["stages"][0]["optimizer"]["name"] = "sgd"
        return settings

    monkeypatch.setattr(config, "load_yaml", wrong_optimizer)
    with pytest.raises(ValueError, match="optimizer.*adam"):
        resolve_run(standalone(), project_root=ROOT)

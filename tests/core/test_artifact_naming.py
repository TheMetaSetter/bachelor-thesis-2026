from __future__ import annotations

import pytest

from src.core.artifact_naming import (
    build_artifact_identity,
    build_wandb_artifact_name,
    build_wandb_smoke_run_name,
    is_valid_wandb_smoke_run_name,
    resolve_wandb_run_name,
)
from src.core.config import _validate_experiment_top_level_structure


def _config(
    *, variant: str = "O0", entity_id: str = "machine_1_6", seed: int = 36
) -> dict[str, object]:
    return {
        "seed": seed,
        "data": {"dataset_name": "smd"},
        "task": {"offline_variant": variant, "entity_id": entity_id},
    }


def test_builds_short_config_name_from_human_identity() -> None:
    identity = build_artifact_identity(_config(), stage="stageA")

    assert build_wandb_artifact_name(role="cfg", identity=identity) == (
        "cfg-stageA-O0-machine_1_6-s36"
    )


def test_builds_online_checkpoint_name_with_online_variant() -> None:
    identity = build_artifact_identity(
        _config(), stage="online", online_variant="A1"
    )

    assert build_wandb_artifact_name(role="ckpt", identity=identity) == (
        "ckpt-online-A1-O0-machine_1_6-s36"
    )


def test_variant_and_entity_are_part_of_the_name_identity() -> None:
    first = build_wandb_artifact_name(
        role="cfg",
        identity=build_artifact_identity(_config(variant="O0"), stage="stageA"),
    )
    second = build_wandb_artifact_name(
        role="cfg",
        identity=build_artifact_identity(
            _config(variant="O1", entity_id="machine_3_9"), stage="stageA"
        ),
    )

    assert first != second
    assert "O0" in first
    assert "machine_1_6" in first
    assert "O1" in second
    assert "machine_3_9" in second


def test_missing_seed_is_rejected() -> None:
    config = _config()
    config.pop("seed")

    with pytest.raises(ValueError, match="seed"):
        build_artifact_identity(config, stage="stageA")


def test_name_longer_than_wandb_limit_is_rejected() -> None:
    identity = {
        "dataset": "smd",
        "variant": "O0",
        "entity": "machine_1_6",
        "seed": 36,
        "stage": "x" * 120,
    }

    with pytest.raises(ValueError, match="128"):
        build_wandb_artifact_name(role="cfg", identity=identity)


@pytest.mark.parametrize(
    ("phase_token", "identity_tokens", "entity_token", "seed", "expected"),
    [
        ("off", ["O0"], "e1_6", 8, "smk-off-O0-e1_6-s8"),
        ("on", ["KA", "A2"], "e3_9", 36, "smk-on-KA-A2-e3_9-s36"),
        ("off", ["KA"], "e3_4", 6, "smk-off-KA-e3_4-s6"),
    ],
)
def test_builds_selected_smoke_run_names(
    phase_token: str,
    identity_tokens: list[str],
    entity_token: str,
    seed: int,
    expected: str,
) -> None:
    assert (
        build_wandb_smoke_run_name(
            phase_token=phase_token,
            identity_tokens=identity_tokens,
            entity_token=entity_token,
            seed=seed,
        )
        == expected
    )
    assert is_valid_wandb_smoke_run_name(expected)


def test_smoke_run_name_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        build_wandb_smoke_run_name(
            phase_token="off",
            identity_tokens=["O0"],
            entity_token="e1_6",
            seed=None,
        )


def test_smoke_run_name_rejects_unsupported_token_characters() -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        build_wandb_smoke_run_name(
            phase_token="off",
            identity_tokens=["O0/bad"],
            entity_token="e1_6",
            seed=8,
        )


def test_resolve_wandb_run_name_preserves_configured_smoke_name() -> None:
    configured_name = "smk-on-KA-A2-e3_9-s36"

    resolved_name = resolve_wandb_run_name(
        {"wandb_run_name": configured_name},
        {
            "experiment_name": "legacy-name",
            "seed": 8,
            "data": {"dataset_name": "smd"},
            "task": {
                "offline_variant": "O0",
                "entity_id": "machine_1_6",
            },
        },
    )

    assert resolved_name == configured_name


def _top_level_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "experiment_name": "identity-test",
        "seed": 8,
        "device": "cpu",
        "output_dir": "outputs/test",
        "checkpoint_dir": "outputs/test/checkpoints",
        "data": {},
        "model": {"model_name": "online_adaptation"},
        "task": {},
        "optimizer": {},
        "epochs": 1,
    }
    config.update(overrides)
    return config


def test_top_level_validation_accepts_canonical_variant_fields() -> None:
    config = _top_level_config(offline_variant="O0", online_variant="A2")

    _validate_experiment_top_level_structure(config)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("offline_variant", "O2"), ("online_variant", "A3")],
)
def test_top_level_validation_rejects_invalid_canonical_variant_fields(
    field_name: str, value: str
) -> None:
    config = _top_level_config(**{field_name: value})

    with pytest.raises(ValueError, match=field_name):
        _validate_experiment_top_level_structure(config)

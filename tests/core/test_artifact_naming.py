from __future__ import annotations

import pytest

from src.core.artifact_naming import (
    build_artifact_identity,
    build_wandb_artifact_name,
)


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

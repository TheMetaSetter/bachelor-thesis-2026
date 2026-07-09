from __future__ import annotations

import torch

from src.data.scalers import SequenceStandardScaler


def _build_sequence(values: list[list[float]], split: str) -> dict[str, object]:
    x = torch.tensor(values, dtype=torch.float32)
    return {
        "x": x,
        "point_labels": torch.zeros(x.shape[0], dtype=torch.int64),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-1-6",
            "split": split,
            "num_channels": int(x.shape[1]),
            "sequence_length": int(x.shape[0]),
        },
    }


def test_sequence_standard_scaler_keeps_constant_channels_unchanged_everywhere() -> (
    None
):
    train_sequence = _build_sequence(
        [
            [5.0, 0.0, 1.0000],
            [5.0, 1.0, 1.0004],
            [5.0, 2.0, 1.0008],
            [5.0, 3.0, 1.0004],
        ],
        split="train",
    )
    clean_validation_sequence = _build_sequence(
        [
            [5.0, 10.0, 1.0020],
            [5.0, 11.0, 1.0030],
            [5.0, 12.0, 1.0040],
            [5.0, 13.0, 1.0050],
        ],
        split="val",
    )

    scaler = SequenceStandardScaler()
    scaler.fit([train_sequence])

    transformed_train = scaler.transform_sequence(train_sequence)
    transformed_clean_validation = scaler.transform_sequence(clean_validation_sequence)

    assert scaler.feature_active_mask is not None
    assert scaler.feature_active_mask.tolist() == [False, True, True]
    assert torch.allclose(transformed_train["x"][:, 0], train_sequence["x"][:, 0])
    assert torch.allclose(
        transformed_clean_validation["x"][:, 0],
        clean_validation_sequence["x"][:, 0],
    )


def test_sequence_standard_scaler_floors_active_channel_std_at_one_e_minus_three() -> (
    None
):
    train_sequence = _build_sequence(
        [
            [5.0, 0.0, 1.0000],
            [5.0, 1.0, 1.0004],
            [5.0, 2.0, 1.0008],
            [5.0, 3.0, 1.0004],
        ],
        split="train",
    )

    scaler = SequenceStandardScaler()
    scaler.fit([train_sequence])
    transformed_train = scaler.transform_sequence(train_sequence)

    expected_channel_2 = (
        train_sequence["x"][:, 2] - scaler.feature_mean[2]
    ) / torch.tensor(1.0e-3, dtype=torch.float32)

    assert torch.allclose(transformed_train["x"][:, 2], expected_channel_2)
    assert scaler.feature_std is not None
    assert scaler.feature_std[2].item() < 1.0e-3

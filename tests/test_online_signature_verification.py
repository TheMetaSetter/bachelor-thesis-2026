import torch

from src.engine.online_tta.signature_verification import (
    SignatureWindow,
    build_pnn_token_mask,
    filter_known_anomaly_tokens,
    find_recurrent_signatures,
    ordered_continuous_signature,
)


def test_signature_helpers_keep_order_and_mask_known_anomalies() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    signatures = ordered_continuous_signature(hidden, prototypes, topk=2)
    assert signatures[0][0] == (0, 1)
    assert signatures[0][1] == (1, 0)
    mask = build_pnn_token_mask(signatures, {(0, 1)}, torch.tensor([[False, True]]))
    assert mask.tolist() == [[True, False]]


def test_recurrence_requires_non_overlapping_windows() -> None:
    signature = (0, 1)
    windows = [
        SignatureWindow("m1", 0, 20, [[signature]]),
        SignatureWindow("m1", 20, 40, [[signature]]),
    ]
    assert find_recurrent_signatures(windows) == {signature}


def test_anomaly_radius_filter_is_read_only() -> None:
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = filter_known_anomaly_tokens(
        hidden, codebook, torch.tensor([True, False]), torch.tensor([0.1, 0.1])
    )
    assert result.tolist() == [[True, False]]

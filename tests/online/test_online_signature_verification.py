import torch

from src.engine.online_tta.signature_verification import (
    SignatureWindow,
    build_pnn_token_mask,
    filter_known_anomaly_tokens,
    find_recurrent_signatures,
    ordered_continuous_signature,
)
from src.engine.online_tta.verification_adapter import verify_buffer_entries


class _ReferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discrete_codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.anomalous_codeword_mask = torch.tensor([True, False])
        self.anomaly_radii = torch.tensor([0.01, 0.01])
        self.verification_codeword_class_ids = torch.tensor([0, 1])
        self.verification_contributing_token_counts = torch.tensor([1.0, 1.0])
        self.verification_metadata_split = "synthetic_train"
        self.verification_metadata_schema_version = 1
        self.verification_metadata_initialization_seed = 7
        self.continuous_prototype_bank = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        self.verification_metadata_source = "test"


class _VerificationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reference_encoder = type("Adapter", (), {"model": _ReferenceModel()})()

    def forward(self, batch):
        return {"aux": {"projected_hidden": batch["x"]}}


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


def test_buffer_verification_filters_known_anomaly_before_recurrence() -> None:
    entries = [
        {
            "entry_id": f"w{index}",
            "entity_id": "machine-1",
            "window_start": index * 2,
            "window_end": index * 2 + 2,
            "stream_step": index,
            "window": [[1.0, 0.0], [0.0, 1.0]],
        }
        for index in range(2)
    ]
    results = verify_buffer_entries(_VerificationModel(), entries, "cpu")
    assert results["w0"].pnn_mask.tolist() == [[False, True]]
    assert results["w1"].pseudo_normal_points == 1

import pytest
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
        self.forward_source_calls = 0
        self.forward_calls = 0
        self.reference_encoder = type("Adapter", (), {"model": _ReferenceModel()})()

    def forward_source(self, batch):
        self.forward_source_calls += 1
        return {"hidden": batch["x"], "aux": {"reference_hidden": batch["x"]}}

    def forward(self, batch):
        self.forward_calls += 1
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


def test_verify_buffer_entries_uses_frozen_source_latents_only() -> None:
    model = _VerificationModel()
    entries = [
        {
            "entry_id": "w0",
            "entity_id": "machine-1",
            "window_start": 0,
            "window_end": 2,
            "stream_step": 0,
            "window": [[1.0, 0.0], [0.0, 1.0]],
        },
        {
            "entry_id": "w1",
            "entity_id": "machine-1",
            "window_start": 2,
            "window_end": 4,
            "stream_step": 1,
            "window": [[1.0, 0.0], [0.0, 1.0]],
        },
    ]
    verify_buffer_entries(model, entries, "cpu")
    assert model.forward_source_calls == 2
    assert model.forward_calls == 0


def test_verify_buffer_entries_reuses_current_event_source_hidden() -> None:
    model = _VerificationModel()
    entries = [
        {
            "entry_id": "w0",
            "entity_id": "machine-1",
            "window_start": 0,
            "window_end": 2,
            "stream_step": 0,
            "window": [[1.0, 0.0], [0.0, 1.0]],
        },
        {
            "entry_id": "w1",
            "entity_id": "machine-1",
            "window_start": 2,
            "window_end": 4,
            "stream_step": 1,
            "window": [[1.0, 0.0], [0.0, 1.0]],
        },
    ]

    verify_buffer_entries(
        model,
        entries,
        "cpu",
        source_hidden_by_entry_id={
            "w1": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        },
    )

    assert model.forward_source_calls == 1
    assert model.forward_calls == 0


def test_ordered_continuous_signature_breaks_ties_deterministically() -> None:
    hidden = torch.tensor([[[1.0, 0.0]]])
    prototypes = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    signatures = ordered_continuous_signature(hidden, prototypes, topk=2)
    assert signatures == [[(0, 1)]]


def test_signature_window_from_dict_rejects_non_integer_ids() -> None:
    from src.engine.online_tta.signature_verification import signature_window_from_dict

    with pytest.raises(TypeError, match="integers"):
        signature_window_from_dict(
            {
                "entity_id": "m1",
                "start": 0,
                "end": 2,
                "signatures": [[[0, "stochastic-id"]]],
            }
        )

import pytest
import torch

from src.engine.online_tta.signature_verification import PrototypeVerificationMetadata


def test_prototype_metadata_validates_shapes() -> None:
    metadata = PrototypeVerificationMetadata(
        torch.zeros(3, 4), torch.tensor([True, False, False]), torch.ones(3)
    )
    assert metadata.codebook.shape == (3, 4)


def test_prototype_metadata_rejects_missing_model_fields() -> None:
    with pytest.raises(AttributeError, match="anomalous_codeword_mask"):
        PrototypeVerificationMetadata.from_model(torch.nn.Linear(2, 2))

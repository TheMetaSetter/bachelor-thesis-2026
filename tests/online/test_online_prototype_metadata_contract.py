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


def test_prototype_metadata_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PrototypeVerificationMetadata(
            torch.zeros(2, 3), torch.tensor([True, False]), torch.tensor([-0.1, 0.0])
        )


def test_model_calibrates_and_roundtrips_anomaly_metadata() -> None:
    from src.models.thesis_multitask import ThesisMultitaskModel

    model = ThesisMultitaskModel(
        input_dim=2,
        window_size=4,
        encoder_dim=4,
        hidden_dim=2,
        num_classes=2,
        continuous_num_prototypes=2,
        discrete_codebook_size=4,
        dropout=0.0,
        use_synthetic_augmentation=False,
    )
    model._initialize_memory_buffers_from_token_pool(
        continuous_hidden_tokens=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        discrete_hidden_tokens_by_class={
            0: torch.tensor([[1.0, 0.0], [0.9, 0.1]]),
            1: torch.tensor([[0.0, 1.0], [0.1, 0.9]]),
        },
    )
    state = model.get_checkpoint_extra_state()
    restored = ThesisMultitaskModel(
        input_dim=2,
        window_size=4,
        encoder_dim=4,
        hidden_dim=2,
        num_classes=2,
        continuous_num_prototypes=2,
        discrete_codebook_size=4,
        dropout=0.0,
        use_synthetic_augmentation=False,
    )
    restored.load_checkpoint_extra_state(state)
    metadata = PrototypeVerificationMetadata.from_model(restored)
    assert metadata.anomalous_codeword_mask.tolist() == [False, False, True, True]
    assert torch.all(metadata.anomaly_radii >= 0)
    assert metadata.source_split == "synthetic_train"
    assert metadata.schema_version == 1
    assert metadata.initialization_seed >= 0
    assert metadata.codeword_class_ids is not None
    assert metadata.contributing_token_counts is not None
    assert metadata.codeword_class_ids.shape == (4,)
    assert metadata.contributing_token_counts.shape == (4,)


def test_checkpoint_extra_state_rejects_malformed_verification_provenance() -> None:
    from src.models.thesis_multitask import ThesisMultitaskModel

    model = ThesisMultitaskModel(
        input_dim=2,
        window_size=4,
        encoder_dim=4,
        hidden_dim=2,
        num_classes=2,
        continuous_num_prototypes=2,
        discrete_codebook_size=4,
        dropout=0.0,
        use_synthetic_augmentation=False,
    )
    state = model.get_checkpoint_extra_state()
    state["verification_metadata_schema_version"] = 2
    with pytest.raises(ValueError, match="verification_metadata_schema_version"):
        model.load_checkpoint_extra_state(state)

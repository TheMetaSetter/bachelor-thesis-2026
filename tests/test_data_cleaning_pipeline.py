from __future__ import annotations

from src.data import load_smd_data
from src.data.cleaning import SequenceCleaningPipeline
from src.data.datasets.smd import SMDDatasetParser


def test_sequence_cleaning_pipeline_preserves_shapes_and_can_annotate_metadata() -> (
    None
):
    parser = SMDDatasetParser(
        root_dir="data/ServerMachineDataset", validation_split_ratio=0.2
    )
    parsed_sequences = parser.parse()
    cleaning_pipeline = SequenceCleaningPipeline(annotate_metadata=True)

    cleaned_sequences = cleaning_pipeline.transform_splits(
        {"train": [parsed_sequences["train"][0]]}
    )
    first_cleaned_sequence = cleaned_sequences["train"][0]

    assert first_cleaned_sequence["x"].shape == parsed_sequences["train"][0]["x"].shape
    assert (
        first_cleaned_sequence["point_labels"].shape
        == parsed_sequences["train"][0]["point_labels"].shape
    )
    assert first_cleaned_sequence["meta"]["cleaning"] == "validated_sequence_contract"


def test_public_data_api_can_annotate_cleaning_metadata() -> None:
    public_bundle = load_smd_data(
        root="data/ServerMachineDataset",
        annotate_cleaning_metadata=True,
        max_train_windows=4,
        max_val_windows=2,
        max_test_windows=2,
    )

    assert (
        public_bundle.raw_sequences["train"][0]["meta"]["cleaning"]
        == "validated_sequence_contract"
    )

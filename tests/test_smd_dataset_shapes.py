from __future__ import annotations

from src.data.datasets.smd import SMDDatasetParser


def test_smd_parser_reads_all_machine_files_and_preserves_entity_metadata() -> None:
    parser = SMDDatasetParser(root_dir="data/ServerMachineDataset", validation_split_ratio=0.2)
    parsed_sequences = parser.parse()

    assert len(parsed_sequences["train"]) == 28
    assert len(parsed_sequences["val"]) == 28
    assert len(parsed_sequences["test"]) == 28

    first_train_sequence = parsed_sequences["train"][0]
    first_test_sequence = parsed_sequences["test"][0]
    assert first_train_sequence["meta"]["entity_id"] == "machine-1-1"
    assert first_train_sequence["x"].shape[1] == 38
    assert first_test_sequence["x"].shape[0] == first_test_sequence["point_labels"].shape[0]


def test_smd_parser_can_filter_to_single_entity() -> None:
    parser = SMDDatasetParser(
        root_dir="data/ServerMachineDataset",
        validation_split_ratio=0.2,
        entity_ids=["machine-2-1"],
    )
    parsed_sequences = parser.parse()

    assert len(parsed_sequences["train"]) == 1
    assert len(parsed_sequences["val"]) == 1
    assert len(parsed_sequences["test"]) == 1
    assert parsed_sequences["train"][0]["meta"]["entity_id"] == "machine-2-1"
    assert parsed_sequences["val"][0]["meta"]["entity_id"] == "machine-2-1"
    assert parsed_sequences["test"][0]["meta"]["entity_id"] == "machine-2-1"

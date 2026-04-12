from __future__ import annotations

"""Conservative cleaning and validation helpers for raw sequence bundles."""

from typing import Any

from src.core.contracts import validate_raw_sequence


class SequenceCleaningPipeline:
    """A conservative cleaning layer that makes existing validation explicit.

    The initial implementation is intentionally narrow. It validates sequence
    contracts, checks metadata consistency, and can annotate metadata to show
    that the cleaning layer ran. It does not invent dataset-specific heuristics.
    """

    def __init__(self, annotate_metadata: bool = False) -> None:
        self.annotate_metadata = annotate_metadata

    def transform_sequence(self, sequence: dict[str, Any]) -> dict[str, Any]:
        validate_raw_sequence(sequence)
        cleaned_sequence = dict(sequence)
        cleaned_sequence["meta"] = dict(sequence["meta"])
        if int(cleaned_sequence["meta"]["sequence_length"]) != int(cleaned_sequence["x"].shape[0]):
            raise ValueError("sequence_length metadata must match x length")
        if int(cleaned_sequence["meta"]["num_channels"]) != int(cleaned_sequence["x"].shape[1]):
            raise ValueError("num_channels metadata must match x width")
        if self.annotate_metadata:
            cleaned_sequence["meta"]["cleaning"] = "validated_sequence_contract"
        return cleaned_sequence

    def transform_sequences(self, sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.transform_sequence(sequence) for sequence in sequences]

    def transform_splits(
        self,
        sequences_by_split: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            split_name: self.transform_sequences(split_sequences)
            for split_name, split_sequences in sequences_by_split.items()
        }

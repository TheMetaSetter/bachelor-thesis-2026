# Evaluation Reconstruction Evidence

Date: 2026-06-26

This note records small forensic artifacts used to inspect how the current evaluator:

1. merges overlapping windows back into pointwise timelines
2. keeps uncovered suffix points in the reconstructed record
3. concatenates multiple entity records into one global pointwise vector before metric computation

Artifacts:

- `evaluation_reconstruction_uncovered_suffix.png`
- `evaluation_reconstruction_entity_concatenation.png`

Related tests:

- `tests/test_evaluation_protocol_audit.py::test_reconstructed_records_keep_uncovered_suffix_as_zero_scored_points`
- `tests/test_evaluator_thresholding.py::test_evaluator_concatenates_entity_records_into_one_global_pointwise_vector`

Interpretation is intentionally deferred. These artifacts are collected as evidence only.

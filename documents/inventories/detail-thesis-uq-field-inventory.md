# THESIS UQ Field Inventory

Date: 2026-07-14

Scope:
- Full-spec UQ contract in `documents/spec/full-spec-v3.md`
- Runtime output validation in `src/core/contracts.py`
- Monte Carlo / variance construction in `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`
- Checkpoint metadata in `src/engine/checkpoint.py`

## 1. Fields stored in checkpoint metadata

These are the UQ-related fields that can appear in `checkpoint_metadata`, and the same family is also mirrored in `extra_state` when the model saves them.

- `stochastic_inference`
- `monte_carlo_samples`
- `continuous_temperature`
- `discrete_temperature`
- `variance_correction`
- `return_mc_samples`
- `sample_retention_policy`

## 2. Fields stored in checkpoint `extra_state`

The checkpoint can also carry the same UQ control / provenance fields through `extra_state`:

- `monte_carlo_samples`
- `continuous_temperature`
- `discrete_temperature`
- `variance_correction`
- `return_mc_samples`
- `sample_retention_policy`

Important:
- `extra_state` is metadata/control, not the full Monte Carlo tensor payload.
- In the smoke checkpoint we inspected, `extra_state` did **not** store the actual uncertainty tensors.

## 3. Runtime fields in `outputs["aux"]["stochastic_query"]`

These are the Monte Carlo sample payloads and their control metadata:

- `schema_version`
- `enabled`
- `num_samples`
- `continuous_temperature`
- `discrete_temperature`
- `continuous_retrieved_samples`
- `discrete_retrieved_samples`
- `discrete_topk_ids`
- `reconstruction_samples`
- `classification_probability_samples`
- `point_score_samples`
- `window_score_samples`
- `return_mc_samples`
- `sample_retention_policy`
- `logits_samples`  (present in code; not part of the strict spec schema block)

## 4. Runtime fields in `outputs["aux"]["uncertainty"]`

These are the actual uncertainty summaries computed from the Monte Carlo samples:

- `point_anomaly_score_variance`
- `window_anomaly_score_variance`
- `continuous_retrieval_variance_point`
- `continuous_retrieval_variance_window`
- `discrete_retrieval_variance_point`
- `discrete_retrieval_variance_window`
- `reconstruction_variance_full`
- `reconstruction_variance_point`
- `reconstruction_variance_window`
- `classification_probability_variance`
- `classification_variance_mean`

## 5. Top-level mean outputs tied to UQ

These are the official prediction outputs produced from the Monte Carlo mean:

- `recon`
- `logits`
- `point_scores`
- `window_scores`

Spec-side sample names:

- `recon_samples`
- `classification_probs`
- `point_score_samples`
- `window_score_samples`

## 6. Not UQ, but nearby

These live in `outputs["aux"]["deterministic_geometry"]` and should not be confused with UQ fields:

- `nearest_codeword_ids`
- `nearest_codeword_distances`
- `known_anomaly_mask`
- `continuous_signature_ids`
- `latent_window_score`

## 7. Quick interpretation

If the question is “what must exist for full UQ support?”, the minimum set is:

- Monte Carlo sample tensors in `outputs["aux"]["stochastic_query"]`
- uncertainty summaries in `outputs["aux"]["uncertainty"]`
- UQ control metadata in `checkpoint_metadata` and `extra_state`
- top-level mean outputs `recon`, `logits`, `point_scores`, `window_scores`

If the question is “what did the smoke checkpoint actually store?”, it stored only the control metadata, not the full sample/variance tensors.

---
title: "Online TTA Terminology Ontology"
status: authoritative-for-naming
scope: "THESIS online test-time adaptation"
applies_to: "specifications, pseudocode, runtime code, tests, configuration, checkpoints, metrics, and experiment artifacts"
root_ontology: "documents/spec/offline_pretraining_terminology_ontology.md"
evidence_revision: e58602f45ee5439a1e001f060e8ea640aeddde9c
---

# Online TTA Terminology Ontology

## The story of names in the online phase

The online phase receives two artifacts from offline training:
`stage_b_best_checkpoint` and `threshold_artifact`. It does not rebuild them
from the test stream. From these artifacts, the runtime creates
`frozen_source_model`, adds `online_mlp_projector`, processes each
`causal_window`, and records scores, triage decisions, verification cycles, and
update events.

This ontology follows that story through owner, lifecycle, and evidence status.
Inherited names keep their old identity. Mutable names state exactly which
state may change.

> **Notation authority:** The point-level anomaly-score notation in this document follows [Point-level anomaly score designs and standard notation](anomaly-score-designs-and-notation.md). Runtime, config, and artifact names do not change because of notation normalization.


## 1. Purpose and offline-first principle

This document standardizes object names in `online_tta_phase`. It inherits object identity from [`offline_pretraining_terminology_ontology.md`](offline_pretraining_terminology_ontology.md) and does not create a new name for the same offline object.

The base principle is:

```text
offline_pretraining_phase
    produces stage_b_best_checkpoint
    produces threshold_artifact

online_tta_phase
    loads stage_b_best_checkpoint as its reference checkpoint role
    loads the same threshold_artifact
    freezes all inherited offline model state
    may update only online_mlp_projector
```

Therefore:

- `reference_checkpoint_path` is a field pointing to `stage_b_best_checkpoint`; it does not define a new checkpoint type.
- `task.threshold_artifact_path` is the canonical config field pointing to a `threshold_artifact` from the matching offline run. Online must read this file; it must not calibrate the artifact from the test stream.
- `frozen_source_model` is `offline_source_model` after loading the Stage B checkpoint and freezing it.
- `continuous_prototype_bank`, `discrete_codebook`, `anomaly_verification_metadata`, `reconstruction_head`, and `classification_head` keep their offline ontology names.
- `online_point_ewma_threshold` belongs to the `threshold_artifact` created offline. Online only reads it and does not recalibrate it from the test stream.

The online story also inherits the routing decision made at the offline
`stage_b_memory_initialization` boundary. The canonical default is
`fusion_mode: direct_branch_routing` for the Stage B path that produced
`stage_b_best_checkpoint`. Online does not choose a new routing mode. If the
checkpoint metadata records another mode, online preserves that explicit
override.

Online also preserves the reason for this separation. The discrete codebook
may carry anomaly patterns, so its output must not enter the reconstruction
head. The continuous prototype bank describes normal patterns, so its output
must not add noise to the classification head. Online adaptation therefore
keeps the inherited branch-specific inputs and does not introduce simple
fusion.

## 2. Evidence status and mapping rules

This document uses these statuses:

- `inherited`: the object is defined in the offline ontology;
- `implemented`: the current online source has the object or behavior;
- `desired-contract`: the object belongs to the desired user flow, but the current runtime does not have the same contract;
- `documented-intent`: the specification requests it and runtime comparison is required;
- `configured`: the current config selects the object or behavior;
- `tested`: a test checks the object or behavior;
- `historical`: the object is kept only for reading older code or documents;
- `unknown`: there is not enough evidence.

`inherited` describes ownership. It may appear together with `implemented`
when online code implements an object inherited from the offline ontology.

The mapping types keep the offline ontology meanings: `canonical`, `exact alias`, `contextual alias`, `historical name`, and `not an alias`.

## 3. High-level ontology

| Canonical name | Object type | Owner | Status |
| --- | --- | --- | --- |
| `online_tta_phase` | phase | online TTA engine | implemented |
| `offline_variant` | inherited experiment dimension | offline config/checkpoint identity | inherited |
| `online_variant` | online experiment dimension | online config | implemented |
| `causal_window` | online input object | online stream | implemented |
| `frozen_source_model` | inherited model state | `OnlineAdaptationModel` | implemented |
| `online_mlp_projector` | mutable online module | `OnlineAdaptationModel` | implemented |
| `threshold_artifact` | inherited calibration artifact | offline benchmark, read by online | inherited |
| `online_event` | one score/triage/adaptation lifecycle | online engine | implemented |
| `online_event_record` | immutable per-window result | online engine | implemented |
| `online_update_event` | one projector update | online engine | implemented |
| `verification_result` | result for one verified entry | verification cycle | documented-intent |
| `verification_buffer` | stateful container | `VerificationBuffer` | implemented |
| `online_runtime_state` | resumable stream state | runtime-state owner | implemented |

### 3.1 `online_tta_phase`

The phase processes the causal stream after offline pre-training. Whenever a
point appears in `causal_window`, the code updates its EWMA score, prediction,
and display data. When later windows no longer contain the point, the latest
prediction is kept.

| Name found in the repository | Mapping |
| --- | --- |
| `online TTA` | exact alias in prose |
| `online phase` | contextual alias |
| `online adaptation phase` | contextual alias |
| `Phase 4` | historical project-planning name; do not use it as a runtime identifier |

### 3.2 `online_variant`

| Value | Canonical meaning |
| --- | --- |
| `A0` | Inference only; it does not call the projector or use an optimizer |
| `A1` | Updates only through the verified non-empty PNN reconstruction path |
| `A2` | Updates through the guarded hard-old path or verified non-empty PNN path, with `online_contrastive_loss` |

`O0_A2` or `O1_A2` is a combined run label containing `offline_variant` and
`online_variant`. It is not a new value of `online_variant`.

`O2` is an offline value, not an online value. It inherits its fixed
three-loss, no-Balanced-Point-Score-Loss, direct-routing checkpoint from the
offline phase. It does not change the meanings of A0, A1, or A2.

## 4. Inherited offline objects

### 4.1 `stage_b_best_checkpoint`

The official input checkpoint for the online source model.

| Online name | Mapping | Note |
| --- | --- | --- |
| `reference_checkpoint_path` | field pointing to the canonical object | A path field, not a checkpoint object |
| `reference checkpoint` | contextual role alias | Correct only when the path resolves to Stage B `best.pt` |
| `offline checkpoint` | contextual alias | Too broad for a new contract |
| `stage_a_best_checkpoint` | not an alias | It does not contain the final initialized and fine-tuned Stage B state |

### 4.2 `frozen_source_model`

`ThesisMultitaskModel` is restored from `stage_b_best_checkpoint`, placed in
evaluation mode, and frozen. It owns:

```text
shared_encoder
continuous_prototype_bank
discrete_codebook
anomaly_verification_metadata
reconstruction_fusion_projection
classification_fusion_projection
reconstruction_head
classification_head
```

`reference_encoder.model` is the runtime path to this object. `online_encoder` is
a second adapter over the same inherited offline model; it is not an encoder
updated by the online optimizer.

### 4.3 `threshold_artifact`

Entity-scoped artifact created by the offline phase. Online reads at least:

```text
online_point_ewma_threshold
point_score_transform
point_score_c
point_score_tau
point_score_tau_estimator
point_score_mad_normalizer
input_window_threshold
latent_window_low_threshold
latent_window_high_threshold
checkpoint identity
entity identity
window_size
EWMA weights
```

The runtime artifact may have nested field names such as
`thresholds.online_ewma_point.value`. Field serialization may differ from the
canonical name, but object semantics do not change.

`task.threshold_artifact_path` is the canonical reference field. It must point
to an artifact schema version 4 whose identity matches
`reference_checkpoint_path`. The new runtime rejects older schemas. This is the
migration boundary for the online vector-EWMA contract; do not infer a new
vector artifact from an old endpoint artifact.

## 5. Online input, score, and prediction objects

### 5.1 `causal_window`

The latest online window contains only observations that have appeared by the
current cursor:

```text
x: FloatTensor[B, L, D]
point_labels: None
absolute_indices: LongTensor[B, L]
timestamps: Tensor[B, L] | None
meta.entity_id: str
meta.start_index: int
meta.end_index: int
meta.stream_step: int
```

`W_t`, `online_batch`, and `batch` are contextual aliases. `offline_window` is
not an alias because the online window has a different causal lifecycle and
stride.

### 5.2 `window_point_scores`

Inherited model output vector `[B,L]`. Each value is the selected point-level
MSE in the current window. The default is the Monte Carlo mean raw input MSE
with `point_score_transform: identity`. A latent-MSE run must declare
`score_space: latent` and use a named latent MSE. The sigmoid is historical and
opt-in only.

`raw_point_scores` is a legacy name in the desired-flow draft. The canonical
name shows that the container belongs to one window and does not use `raw` to
describe the field value.

### 5.3 Vector prediction objects

The following objects are the current runtime contract:

| Canonical name | Shape | Meaning |
| --- | --- | --- |
| `active_ewma_point_scores` | `map[absolute_index, float]` | EWMA state only for points in the latest active causal window |
| `current_window_ewma_point_scores` | `[L]` | EWMA vector of the current causal window |
| `window_point_predictions` | `[L]` | Binary prediction vector after thresholding |

`point_level_binary_predictions` is an exact alias of
`window_point_predictions`. Do not call this vector `prediction` because the
runtime record currently uses `prediction` for a scalar endpoint prediction.

New points in the map use the current raw-MSE `window_point_scores`. A point
that appears again in an overlapping window uses EWMA weights `0.9 current +
0.1 previous`. The runtime replaces the whole map with the points from the
current causal window, so it has no finalized-point table.

### 5.4 Endpoint compatibility fields

The runtime record still writes the last point of the vector so older scalar
readers can use it:

| Canonical name | Runtime name | Shape |
| --- | --- | --- |
| `endpoint_point_score` | `raw_point_score` (legacy runtime field name; value is the selected raw MSE by default) | scalar |
| `previous_endpoint_ewma_point_score` | `previous_ewma_score` | scalar or absent |
| `current_endpoint_ewma_point_score` | `ewma_point_score` | scalar |
| `endpoint_point_prediction` | `prediction` | scalar binary |

Scalar endpoint objects are not aliases of vector objects. They are
compatibility fields and must not be used for triage, verification, or updates.

### 5.5 `online_point_ewma_threshold`

The point threshold applies to the online EWMA raw-MSE score. Offline creates
this threshold with:

```text
Q_0.99(clean-validation raw MSE after sliding-window + EWMA)
```

One point is anomalous when `current_window_ewma_point_scores >
online_point_ewma_threshold`.

| Name found in the repository | Mapping |
| --- | --- |
| `online_ewma_point_threshold` | exact runtime/artifact alias |
| `B_point_high` | exact mathematical alias |
| `T_point_EWMA` | exact mathematical alias |
| `threshold_value` | contextual local-variable alias |
| `offline_point_threshold` | not an alias |
| `input_window_threshold` | not an alias |

### 5.6 `score_protocol` in the online story

Online scoring follows the protocol selected by offline calibration:

```text
score_protocol
    → score_space
    → point_score_transform
    → window_point_scores
    → current_window_ewma_point_scores
    → window_point_predictions
```

The default is `score_space: raw_input` with
`point_score_transform: identity`. Online must read the matching
`threshold_artifact`; it must not silently replace the protocol with latent MSE
or the historical sigmoid transform.

## 6. Representation and model objects

### 6.1 `source_hidden`

Frozen source encoder output for `causal_window`. The runtime field is
`reference_hidden`. In A0, source hidden directly becomes the query
representation.

### 6.2 `projected_hidden`

Output of `online_mlp_projector(source_hidden)`. A1/A2 use this object as the
mutable query representation. `online_hidden` is a contextual alias; do not use
it because it can be confused with the frozen `online_encoder` adapter.

### 6.3 `online_mlp_projector`

The only mutable module in an accepted A1/A2 event. `projector` is a runtime
convenience alias. Do not map `online_encoder` to the projector; the encoder
remains frozen.

### 6.4 `online_model_outputs`

Stable output contract inherited from the offline model:

```text
hidden
pooled
reconstruction
classification_logits
window_point_scores
window_anomaly_scores
aux.reference_hidden
aux.projected_hidden
aux.latent_window_score
```

Runtime top-level aliases are `recon`, `logits`, `point_scores`, and
`window_scores`.
`point_scores` is the selected point-level MSE; raw input MSE with the identity
transform is the default. `window_scores` is the raw window reconstruction MSE
used for `input_window_score` and triage.

## 7. Triage objects

### 7.1 `input_window_score`

Window-level reconstruction MSE in input space. This is the score used with
`input_window_threshold` in four-region triage.

`raw_point_score` and `endpoint_point_score` are not aliases: they have
different levels and reductions.

### 7.2 `latent_window_score`

Deterministic latent-memory score used with the latent threshold band. Runtime
may fall back to `window_scores` when the output has no separate field; the
fallback does not make the two score types aliases in the ontology.

### 7.3 Triage thresholds

| Canonical name | Mathematical alias | Runtime artifact meaning |
| --- | --- | --- |
| `input_window_threshold` | `B_window` | High quantile of the clean-validation input-window score |
| `latent_window_low_threshold` | `A_low` | Lower edge of the latent threshold band |
| `latent_window_high_threshold` | `A_high` | Upper edge of the latent threshold band |

`online_point_ewma_threshold` is not an alias of the three triage thresholds.

### 7.4 `triage_region`

Four-region classification result before admission and verification:

```text
normal
hard_old_normality
gray_zone
strong_anomaly
```

Truth table:

| Condition | `triage_region` |
| --- | --- |
| `input_window_score <= input_window_threshold` | `normal` |
| `input_window_score > input_window_threshold` and `latent_window_score <= latent_window_low_threshold` | `hard_old_normality` |
| `input_window_score > input_window_threshold` and `latent_window_low_threshold < latent_window_score <= latent_window_high_threshold` | `gray_zone` |
| `input_window_score > input_window_threshold` and `latent_window_score > latent_window_high_threshold` | `strong_anomaly` |

| Old name | Mapping |
| --- | --- |
| `triage_decision` | exact alias when the value belongs to the same four regions |
| `decision` | contextual alias; do not use it in a new contract |
| `event_decision` | not an alias when the object also combines verification and adaptation state |

### 7.5 `hard_old_normality`

One `triage_region`, not a buffer entry or verification result. Only A2 may
update on this region, and only when `hard_old_interval_guard` accepts the
interval.

`hard_old` and `hard-old` are prose aliases. Use `hard_old_normality` as the
identifier.

### 7.6 `hard_old_interval_guard`

State object preventing accepted hard-old updates from using overlapping
intervals. The runtime class is `NonOverlapGuard`. It is not
`verification_buffer`.

## 8. Verification and PNN objects

### 8.1 `verification_buffer`

Instance of `VerificationBuffer`. This container owns admitted gray-zone
entries, non-overlap admission, capacity, TTL, and the “new since cycle” state.

`VerificationBuffer` is the class name. `TTLBuffer` is not an alias; the
historical endpoint TTL buffer was removed from the active flow.

### 8.2 `verification_entry`

One admitted gray-zone causal window:

```text
entry_id: str
entity_id: str
start_index: int
end_index: int
x: FloatTensor[L,D]
status: "unresolved" | "adapted"
ttl_remaining: int
admitted_at_cursor: int
```

Runtime serialization mapping:

| Runtime field | Canonical field | Mapping |
| --- | --- | --- |
| `window_start` | `start_index` | exact alias |
| `window_end` | `end_index` | exact alias |
| `window` | `x` | exact alias |
| `stream_step` | `admitted_at_cursor` | contextual alias; counter equivalence must be preserved |
| `point_score` | `endpoint_point_score` | contextual snapshot field |

`verification_buffer` is not an alias of `verification_entry`; one object is a
container and the other is an item.

### 8.3 `verification_cycle`

One cycle starts when the buffer reaches capacity, has a new entry since the
previous cycle, and no other cycle is running. The cycle encodes stored entries
with the frozen source model, computes deterministic geometry, creates results,
commits adapted statuses, and then ticks TTL for unresolved entries.

### 8.4 Deterministic verification tensors

| Canonical name | Shape | Meaning |
| --- | --- | --- |
| `nearest_codeword_ids` | `[N,L]` | Nearest discrete codeword per token |
| `nearest_codeword_distances` | `[N,L]` | Cosine distance to the nearest codeword |
| `known_anomaly_mask` | `[N,L]` | Tokens inside anomalous codeword radii |
| `continuous_signature_ids` | `[N,L,3]` | Ordered top-3 continuous prototype ids per token |
| `recurrent_signature_set` | set of signature tuples | Signatures appearing in more than one non-overlapping window |
| `pnn_mask` | `[N,L]` | Pseudo-new-normal tokens remaining after known-anomaly filtering and recurrence checks |

`recurrent_signatures` is the historical runtime alias of
`recurrent_signature_set`. `recurrent_signature_ids` is not an alias; this
runtime field may contain a tensor aligned with selected tokens.

If the current gray-zone entry was just admitted and the cycle runs immediately,
the cycle reuses the event's `reference_hidden` for that entry. This tensor
exists only inside the event. It is not a `verification_entry` field and is not
serialized.

`pnn_verified` is only an internal control value in the compatibility path. The
code uses it to select the PNN update path when `pnn_mask` is non-empty. It is
not a canonical object and is not one of the four `triage_region` values.

### 8.5 `verification_result`

One `verification_cycle` creates one `verification_result` for each admitted
`verification_entry`. The result records the geometry used for the decision:

```text
entry_id
known_anomaly_mask
recurrent_signature_set
pnn_mask
accepted_for_adaptation
```

`VerificationResult` is the class-style spelling used by the existing relation
story. The canonical ontology name is `verification_result`. It is not an
alias of `verification_entry`, `pnn_mask`, or `online_update_event`.

The object remains `documented-intent` until the runtime schema and serializer
store this result explicitly.

## 9. Adaptation objects and losses

### 9.1 `hard_old_reconstruction_loss`

Hinge loss that pushes the online window reconstruction score below
`input_window_threshold`:

```text
hard_old_reconstruction_loss
    = RELU(online_window_anomaly_score - input_window_threshold)^2
```

Runtime names include `reconstruction_loss` in the hard-old branch and
`loss_hard_recon` in metrics.

### 9.2 `pnn_reconstruction_loss`

Masked reconstruction loss only on positions where `pnn_mask = TRUE`.

`PNN loss`, `masked PNN reconstruction loss`, and `loss_pnn_recon` are
contextual aliases.

### 9.3 `online_contrastive_loss`

Source-consistency contrastive regularization added to an accepted A2 hard-old
or PNN event.

| Name found in the repository | Mapping |
| --- | --- |
| `L_online_contrastive` | exact mathematical alias |
| `SRC-ON loss` | exact prose alias |
| `contrastive_loss` | contextual runtime local-variable alias |

### 9.4 `online_total_loss`

| Event | Formula |
| --- | --- |
| A1 `pnn_verified` | `pnn_reconstruction_loss` |
| A2 `hard_old_normality` | `hard_old_reconstruction_loss + lambda_online_contrastive * online_contrastive_loss` |
| A2 `pnn_verified` | `pnn_reconstruction_loss + lambda_online_contrastive * online_contrastive_loss` |

`lambda_online_contrastive` is the canonical config meaning. The current
runtime hard-codes multiplier `0.1`; do not call this value offline
`lambda_contrastive` when config ownership differs.

### 9.5 `online_update_event`

One atomic update includes a fresh optimizer, zero gradients, one finite loss,
backward, frozen-gradient assertions, projector gradient clipping, and exactly
one optimizer step. Only `online_mlp_projector` is mutated.

## 10. Record and state objects

### 10.1 `online_event_record`

Per-window immutable record after scoring and an optional update. Vectors use the
same length `L` as `causal_window.absolute_indices`:

```text
entity_id
causal_window.absolute_indices
window_point_scores
current_window_ewma_point_scores
window_point_predictions
online_point_ewma_threshold
online_variant
triage_region
did_update
online_total_loss
```

`raw_point_score`, `ewma_point_score`, and `prediction` may remain beside the
vector as compatibility fields. There is no root-level `absolute_indices` copy.

### 10.2 `online_runtime_state`

Resumable state containing entity identity, offline/online variant identity,
threshold artifact identity, the stream cursor counted by processed causal
windows, `active_ewma_point_scores`, verification buffer state, and hard-old
guard state. It does not contain optimizer moments or
`recurrent_signature_set`.

### 10.3 The online event story

The three event names now have one clear direction:

```text
online_event
    → online_event_record
    → optional online_update_event
```

`online_event` is the whole lifecycle for one causal window. It owns the
immutable `online_event_record`. If A1 or A2 updates the projector, the event
also owns one `online_update_event`. The update object is optional because A0,
normal regions, rejected intervals, and empty PNN results do not update.

These objects are related, but they are not aliases.

## 11. Standard relations between objects

| Subject | Relation | Object |
| --- | --- | --- |
| `online_tta_phase` | `loads` | `stage_b_best_checkpoint` |
| `stage_b_best_checkpoint` | `restores` | `frozen_source_model` |
| `online_tta_phase` | `reads` | `threshold_artifact` |
| `frozen_source_model.shared_encoder` | `produces` | `source_hidden` |
| `online_mlp_projector` | `maps` | `source_hidden` to `projected_hidden` |
| `frozen_source_model` | `produces` | `raw_point_mse` |
| score selection | `maps` | selected raw MSE to `window_point_scores`; identity is the default |
| `score_protocol` | `selects` | `score_space`, `point_score_transform`, and threshold interpretation |
| online EWMA step | `maps` | `window_point_scores` to `current_window_ewma_point_scores` |
| `triage_region` | `depends on` | `input_window_score`, `latent_window_score`, and triage thresholds |
| `gray_zone` | `may create` | `verification_entry` |
| `verification_buffer` | `contains` | `verification_entry` |
| `verification_cycle` | `produces` | `pnn_mask` and per-entry `verification_result` |
| `pnn_mask` | `selects positions for` | `pnn_reconstruction_loss` |
| `hard_old_interval_guard` | `gates` | A2 hard-old update |
| `online_event` | `contains` | `online_event_record` |
| `online_event` | `may contain` | `online_update_event` |
| `online_update_event` | `mutates only` | `online_mlp_projector` |

## 12. Runtime contract and compatibility fields

| Concern | Desired contract | Implemented runtime |
| --- | --- | --- |
| Point-score object | `window_point_scores [L]` | `raw_point_score` endpoint compatibility field |
| EWMA state | `active_ewma_point_scores` + `current_window_ewma_point_scores [L]` | `ewma_point_score` endpoint compatibility field |
| Prediction | `window_point_predictions [L]` | `prediction` endpoint compatibility field |
| PNN order | Triage, gray-zone admission, then verification | Same order |
| PNN update gate | non-empty `pnn_mask` | `pnn_verified` is only internal step control |
| Signature set | Local verification cycle | Not serialized into runtime state |

Pseudocode and source must use canonical vector names for runtime behavior. Only
compatibility readers may use scalar endpoint fields.

## 13. Terminology changes

### 13.1 Raw-input-space MSE v4

Raw input always means original sensor units after applying the fitted scaler's
inverse transform. Use `raw_input_point_mse` and `raw_input_window_mse` for
operational online scores. Use `normalized_input_point_mse` and
`normalized_input_window_mse` for diagnostics only. The raw protocol declares
`score_space: raw_input` and `point_score_transform: identity`; it does not load
the historical sigmoid calibration.

`point_labels` and `window_labels` are ground truth. `point_predictions` and
`window_predictions` are threshold results. A window label is anomalous when
any point label in that window is anomalous.

| Old name | New canonical name | Status | Runtime owner | Migration boundary |
| --- | --- | --- | --- | --- |
| `raw_point_scores` | `window_point_scores` | legacy name replaced; canonical value is selected raw MSE by default | model output | Desired pseudocode |
| `previous_ewma_point_scores` | `active_ewma_point_scores` | replaced because state is keyed by absolute index | online runtime state | vector runtime |
| `current_ewma_point_scores` | `current_window_ewma_point_scores` | renamed for scope | desired online state | Desired pseudocode |
| `point_level_binary_predictions` | `window_point_predictions` | renamed for container clarity | desired output | Desired pseudocode |
| `raw_point_score` | `endpoint_point_score` | renamed for scalar meaning | compatibility record | source keeps field |
| `previous_ewma_score` | `not an alias` | removed scalar state | historical runtime | runtime state schema v2 rejects it |
| `ewma_point_score` | `current_endpoint_ewma_point_score` | renamed for scalar meaning | compatibility record | source keeps field |
| `B_point_high`, `T_point_EWMA` | `online_point_ewma_threshold` | exact alias normalization | threshold artifact | Pseudocode/docs |
| `B_window` | `input_window_threshold` | exact mathematical alias normalization | threshold artifact/triage | Pseudocode/docs |
| `A_low` | `latent_window_low_threshold` | exact mathematical alias normalization | threshold artifact/triage | Pseudocode/docs |
| `A_high` | `latent_window_high_threshold` | exact mathematical alias normalization | threshold artifact/triage | Pseudocode/docs |
| `triage_decision` | `triage_region` | renamed for object type | triage | New contracts |
| `entries` | `verification_entries` | renamed for ownership | verification buffer | Pseudocode |
| `P_known_anomaly` | `known_anomaly_mask` | replaced set-like prose with tensor object | verification | Pseudocode |
| `P_pseudo_new_normality` | `pnn_mask` | replaced set-like prose with tensor object | verification | Pseudocode |
| `SRC-ON loss` | `online_contrastive_loss` | renamed for identifier clarity | online loss | Pseudocode |

### 13.2 Audit bridge names

| Old or nearby name | Canonical name | Status | Meaning |
| --- | --- | --- | --- |
| `VerificationResult` | `verification_result` | documented-intent | One result produced for one admitted entry |
| `online_event` | `online_event` | unchanged | Whole score, triage, and optional adaptation lifecycle |
| `online_event_record` | `online_event_record` | unchanged | Immutable per-window record inside the event |
| `online_update_event` | `online_update_event` | unchanged | Optional projector update inside the event |
| generic `variant` | `method_variant`, `offline_variant`, `online_variant` | split | Separate method choice from phase choices |

The split is a terminology change, not a change to the online loss or
verification order. `O2` is a fixed `offline_variant`: reconstruction loss,
classification loss, and the existing `two_view_contrastive_loss`. It has no
Balanced Point-Score Loss and uses `direct_branch_routing`.

## 14. Required comparison for every new specification

Every new specification version must compare both the offline and online
ontologies. For each rename or new object, record:

```text
old name
new canonical name
mapping type
semantic equivalence or difference
runtime owner
schema and stored data
lifecycle and callers
offline-to-online lineage
checkpoint and artifact impact
migration boundary
```

Never map these automatically:

- `offline_point_threshold` to `online_point_ewma_threshold`;
- `input_window_threshold` to point thresholds;
- `verification_buffer` to `verification_entry`;
- the `window_point_scores` vector to the scalar `endpoint_point_score`;
- `frozen_source_model` to `online_mlp_projector`;
- `stage_a_best_checkpoint` to `stage_b_best_checkpoint`.

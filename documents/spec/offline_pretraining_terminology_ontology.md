---
title: "Offline Pre-training Terminology Ontology"
status: authoritative-for-naming
scope: "THESIS offline pre-training, memory initialization, offline evaluation, and the offline-to-online handoff"
applies_to: "specifications, pseudocode, runtime code, tests, configuration, checkpoints, metrics, and experiment artifacts"
evidence_revision: e58602f45ee5439a1e001f060e8ea640aeddde9c
---

# Offline Pre-training Terminology Ontology

## The story of names in the offline phase

An offline experiment passes through many objects before it becomes an artifact
that the online phase can use. Similar names can hide different owners,
lifecycles, or checkpoint contracts. This ontology follows each name through
the journey: where it appears, what it receives, what it creates, and whether it
is truly an alias of an older name.

The tables below are the record of that journey. They keep evidence status
separate from design intent, so `implemented`, `configured`, `tested`,
`documented-intent`, `historical`, and `unknown` are not mixed together.

> **Notation authority:** The point-level anomaly-score notation in this document follows [Point-level anomaly score designs and standard notation](anomaly-score-designs-and-notation.md). Runtime, config, and artifact names do not change because of notation normalization.


## 1. Purpose

This document is the naming source of truth for **objects** and **object relations** in the THESIS offline phase. It answers four questions for each name:

1. What is the object?
2. Which phase, stage, or operation owns it?
3. What input does it receive, and what output does it create?
4. Which name in code or older documents refers to the same object?

The ontology standardizes names without hiding differences between code and specification. Each behavior claim must have one of these statuses:

- `implemented`: the current source performs the behavior;
- `configured`: the current config selects the behavior;
- `tested`: a test checks the behavior;
- `documented-intent`: the specification requests the behavior, but source alignment is not certain;
- `inherited`: the object comes from an earlier ontology or phase;
- `desired-contract`: the future contract has been selected, but the source does not implement it;
- `historical`: the object belongs only to an older design;
- `unknown`: there is not enough evidence.

`inherited` describes object origin. It may appear with `implemented` when the
current runtime implements an inherited object.

Official names use `snake_case` in pseudocode, schemas, and technical prose. Python class names use `PascalCase`. Artifact file names keep their runtime spelling.

## 2. Background: where the offline phase sits

THESIS has two large phases in sequence:

```text
offline_pretraining_phase
    -> stage_b_best_checkpoint
    -> threshold_artifact
    -> online_tta_phase
```

`offline_pretraining_phase` learns the model from training data and fixes the model state needed for deployment. `online_tta_phase` does not retrain the offline model. The online phase loads `stage_b_best_checkpoint`, freezes the source model, and updates only the online projector when the protocol allows it.

Inside the offline phase:

```text
offline_pretraining_phase
    contains stage_a_multitask_pretraining
    transitions through stage_b_memory_initialization
    contains stage_b_fusion_finetuning
    is followed by offline_evaluation
```

The two easiest words to confuse are `phase` and `stage`:

- `offline_pretraining_phase` is the complete offline lifecycle.
- `stage_a_multitask_pretraining` and `stage_b_fusion_finetuning` are the two training stages inside the phase.
- `stage_b_memory_initialization` is the transition operation between the stages.
- `offline_evaluation` is a post-training operation, not Stage C.

## 3. Mapping rules

| Mapping | Meaning | May it be used in a new contract? |
| --- | --- | --- |
| `canonical` | The one official name | Yes |
| `exact alias` | The same object with the same semantics | No; migrate to the canonical name |
| `contextual alias` | The same object only in the stated context | No |
| `historical name` | An object from an older workflow | No |
| `not an alias` | A different object, even when the names are close | Never map it |

Similar names do not prove identity. When mapping a new name, compare the schema, owner, lifecycle, callers, state mutation, checkpoint contract, and artifact contract.

## 4. High-level ontology

| Canonical name | Object type | Owner | Lifecycle |
| --- | --- | --- | --- |
| `offline_pretraining_phase` | phase | offline benchmark runner | Starts when the experiment config loads; ends after artifact export |
| `offline_variant` | experiment dimension | experiment config | Fixed for one offline run; values are `O0`, `O1`, `O2` |
| `stage_a_multitask_pretraining` | training stage | two-stage orchestrator + `ThesisMultitaskModel` | Runs before memory initialization |
| `stage_b_memory_initialization` | transition operation | two-stage orchestrator + model memory state | Runs once between Stage A and Stage B |
| `stage_b_fusion_finetuning` | training stage | two-stage orchestrator + `ThesisMultitaskModel` | Starts from the initialization checkpoint |
| `offline_evaluation` | post-training operation | benchmark wrapper + `Evaluator` | Runs with the Stage B best checkpoint |
| `online_tta_phase` | downstream phase | online TTA engine | Receives the checkpoint and threshold artifact from offline |

### 4.1 `offline_pretraining_phase`

**Definition.** The complete phase creates the offline source model, runs two training stages, performs one memory-initialization transition, and exports post-training evaluation artifacts.

| Name found in the repository | Mapping | Note |
| --- | --- | --- |
| `offline pre-training` | exact alias in prose | Hyphenated prose form |
| `offline training` | contextual alias | Only when it means the whole phase, not one batch update |
| `two-stage offline pre-training` | contextual alias | Emphasizes the current topology |
| `phase_multitask_pretraining` | not an alias | An older proposed name for one training stage, not the whole phase |
| `training_phase` | not an alias at phase level | The current runtime field keeps `stage_name` |

### 4.2 `offline_variant`

**Definition.** The ablation axis decides whether Stage A uses `point_score_loss`.

| Value | Canonical meaning |
| --- | --- |
| `O0` | `point_score_loss` is off; Stage A uses reconstruction, classification, and two-view contrastive losses |
| `O1` | `point_score_loss` is on in Stage A; it is not on by default in Stage B |

`experiment_variant` values such as `two_stage_base_v1` and `two_stage_point_score_supervised_v1` describe detailed protocols. They do not replace the `offline_variant` object identity used for cross-phase checkpoint and artifact matching.

### 4.2.1 `method_variant` in `tsad-lib`

When the story moves beyond THESIS, one `variant` name becomes ambiguous.
Therefore, `tsad-lib` uses separate names:

```text
method_variant       → the method choice
offline_variant      → the offline ablation
online_variant       → the online adaptation choice
combined_run_label   → the offline_variant + online_variant pair
```

In the `tsad-lib` proposal, `O2` is a `method_variant` with status
`desired-contract`. The current ontology does not assign `O2` to
`offline_variant` and does not silently map it to `O0` or `O1`. A later
specification must state the loss set, checkpoint identity, and artifact
identity of `O2` before a run can start.

`O0` and `O1` are `offline_variant` values only in the offline context.
The same spelling does not create the same object in every context.

### 4.3 `stage_a_multitask_pretraining`

**Definition.** The first training stage. The model learns the encoder and two task heads from scratch with a multitask objective. Final continuous/discrete memory retrieval does not enter this stage's forward path.

| Runtime name | Mapping | Note |
| --- | --- | --- |
| `Stage A` | exact alias in prose | Use the canonical identifier in pseudocode |
| `Stage A: Multitask Pretraining` | exact display label | Use only in UI/logs |
| `TWO_STAGE_A_PHASE_NAME` | contextual alias | The constant is typed as `PHASE`; its value is a stage identifier |
| `training_phase = stage_a_multitask_pretraining` | compatibility field | Its meaning is `stage_name` |

### 4.4 `stage_b_memory_initialization`

**Definition.** The transition operation loads `stage_a_best_checkpoint`, encodes training batches in `eval()` and `no_grad()`, initializes memory banks and verification metadata, and saves `stage_b_initialization_checkpoint`.

**Default routing.** When the story reaches this boundary, the offline
pre-training contract records `fusion_mode: direct_branch_routing` as the
default for the Stage B path that follows. The initialization operation itself
still only collects training features and constructs memory state. It does not
train a fusion block. After initialization, the continuous branch goes to the
reconstruction head and the discrete branch goes to the classification head.

The canonical contract does not use simple fusion, which would join both
branches before the task heads. The discrete codebook can carry anomaly
patterns, so sending it to the reconstruction head could leak those patterns
into reconstruction. The continuous prototype bank describes normal patterns,
so sending it to the classification head could add noise to classification.
Direct routing keeps each branch with the head that owns its signal.

This is the canonical ontology default. A configuration with another
`fusion_mode` is an explicit or historical override and must keep its value in
the run configuration and checkpoint metadata. The shared base YAML currently
uses `task_specific_concat_projection`, so this document does not claim that
every existing configuration already uses the canonical default.

| Name found in the repository | Mapping | Note |
| --- | --- | --- |
| `end-of-Stage-A memory initialization` | exact alias | Viewed from the Stage A output |
| `Stage B initialization` | exact alias | Viewed from the Stage B input |
| `memory initialization stage` | not an alias | This operation is not an independent training stage |
| `bootstrap` | contextual alias | Only memory bootstrap; do not map it to encoder bootstrap epochs |

### 4.5 `stage_b_fusion_finetuning`

**Definition.** The second training stage. It uses the frozen encoder and frozen memory banks to train task-specific fusion projections and task heads.

| Runtime name | Mapping | Note |
| --- | --- | --- |
| `Stage B` | exact alias in prose | Use the canonical identifier in pseudocode |
| `Stage B: Fusion Finetuning` | exact display label | Use only in UI/logs |
| `TWO_STAGE_B_PHASE_NAME` | contextual alias | The constant contains a stage identifier |
| `training_phase = stage_b_fusion_finetuning` | compatibility field | Its meaning is `stage_name` |
| `fusion warm-up` | not an alias | It may be a substep or historical term; it does not represent all of Stage B |

### 4.6 `offline_evaluation`

**Definition.** The operation runs the Stage B model on clean validation, synthetic validation, and test data; restores point-score timelines; calibrates thresholds from clean validation; computes metrics; and exports artifacts.

`evaluation` in `two_stage_execution_report` is a display step name. It is not a training stage and does not change model parameters.

## 5. Data objects

### 5.1 `raw_sequence`

One entity sequence before windowing:

```text
x: FloatTensor[T, D]
point_labels: LongTensor[T] | None
mask: Tensor | None
timestamps: Tensor[T] | None
meta.entity_id: str
meta.split: str
meta.sequence_length: int
```

`train_sequence`, `clean_validation_sequence`, and `test_sequence` have the same object type but belong to different splits.

### 5.2 `offline_window`

One segment of length `window_size` taken from `raw_sequence`:

```text
x: FloatTensor[L, D]
point_labels: LongTensor[L] | None
meta.start_index: int
meta.end_index: int
```

`window`, `input window`, and `segment` are aliases only when the schema above is unchanged. Online `causal_window` is not an alias because its lifecycle and stride differ.

### 5.3 `offline_batch`

The standard batch passed by the trainer to the model:

```text
x: FloatTensor[B, L, D]
point_labels: LongTensor[B, L] | None
mask: Tensor[B, L, D] | None
timestamps: Tensor[B, L] | None
meta: list[dict]
```

### 5.4 `synthetic_training_batch`

`offline_batch` after synthetic anomaly injection:

```text
x: FloatTensor[B, L, D]
classification_labels: LongTensor[B]
synthetic_anomaly_mask: BoolTensor[B, L]
augmentation_metadata: list[dict]
```

The canonical field `x` is the actual model input. Older design names `x_input` and `x_clean` are not current runtime fields. Use them only if a new schema really stores both tensors.

### 5.5 `classification_labels`

Window-level class labels. In `redlamp_multiclass`, class `0` is normal and classes `1..11` are the 11 synthetic anomaly families.

`class_labels` is an exact alias in the older specification; the official runtime field is `classification_labels`.

### 5.6 `synthetic_anomaly_mask`

Point-level binary mask marking the positions that were actually injected. It is not the same as `classification_labels` because an anomalous-class window still contains many clean positions.

### 5.7 `latent_tokens`

The encoder output has shape `[B, L, H]`. The runtime top-level field is `hidden`.

| Name | Mapping |
| --- | --- |
| `hidden` | exact runtime field |
| `latent tensor` | exact alias in the specification |
| `features` | contextual alias; too broad for a new contract |
| `token` | one row `latent_tokens[b, t, :]`, not a channel |

### 5.8 The `tsad-lib` data story

Before THESIS sees a sequence, `tsad-lib` must explain where that sequence
came from. The library keeps the source story explicit:

```text
dataset
    → entity
    → file
    → file_extension

file
    → source_provenance
    → entity
    → raw_sequence
    → offline_window
    → offline_batch
```

These are bridge objects with status `desired-contract`:

| Object | Minimal meaning |
| --- | --- |
| `dataset` | Logical dataset family and its root path |
| `entity` | One selectable time-series identity inside a dataset |
| `file` | One physical source file used by an entity |
| `file_extension` | The extension used by the file parser, such as `.npy`, `.csv`, `.mat`, or `.zip` |
| `source_provenance` | Source path, file hash, parser name, parser settings, and code revision |
| `dataset_adapter` | Reader that converts one source format into the common data object |
| `series_set` | Common adapter output before THESIS windowing |

`entity_id` remains the canonical runtime identity inside THESIS. `entity` is
the notebook or report display field when a separate display name is useful.
They are not aliases unless they contain the same identity contract.

`series_set` is not an exact alias of `raw_sequence`. The adapter output may
still contain timestamps, channel names, capabilities, and source provenance.
The bridge converts it into `raw_sequence` only after the adapter and pipeline
validate the schema.

## 6. Model and state objects

### 6.1 `offline_source_model`

The model is trained and checkpointed in the offline phase. Its runtime class is `ThesisMultitaskModel`. When the same model is loaded online and frozen, the online ontology calls the instance `frozen_source_model`.

### 6.2 `shared_encoder`

The module transforms `offline_batch.x` into `latent_tokens`. Its runtime attribute is `encoder`.

### 6.3 `continuous_memory_initialization_token_pool`

The latent-token set used to initialize `continuous_prototype_bank`.

Normative meaning: clean/normal training tokens only; it does not receive validation, test, or future online tokens.

Current implemented source: normal positions from normal-class synthetic batches. Injected positions do not enter this pool.

### 6.4 `discrete_memory_initialization_token_pools_by_class`

Mapping from class id to the latent-token pool used to initialize `discrete_codebook`.

Normative meaning in full-spec-v3:

- class `0`: normal training tokens;
- class `1..11`: injected anomaly tokens from the matching synthetic class.

Current implemented source gathers all tokens from each class window. This is a conflict, not another alias.

### 6.5 `continuous_prototype_bank`

Frozen memory bank representing normal latent structure. Its current shape is `[32, H]`. It is created by k-means on `continuous_memory_initialization_token_pool`.

`continuous memory`, `continuous bank`, and `continuous prototypes` are contextual aliases. `prototype_context` is not an alias; it is a retrieval output.

### 6.6 `discrete_codebook`

Frozen memory bank containing class-stratified codewords. Its current shape is `[60, H]`, representing 12 classes and 5 codewords per class.

`discrete memory` and `discrete bank` are contextual aliases. `quantized_hidden` is not an alias; it is a retrieval output.

### 6.7 `anomaly_verification_metadata`

Metadata created with `discrete_codebook` so online verification can use deterministic source geometry:

```text
anomalous_codeword_mask
anomaly_radii
verification_codeword_class_ids
verification_contributing_token_counts
verification_metadata_source
```

This state is owned offline but consumed online. Do not call the whole object
`anomaly_radii`, because radii are only one field.

### 6.8 Stage B retrieval and heads

| Canonical object | Runtime field/module | Meaning |
| --- | --- | --- |
| `continuous_prototype_context` | `prototype_context` | Retrieval output from the continuous bank |
| `discrete_codeword_context` | `quantized_hidden` | Retrieval output from the discrete codebook |
| `reconstruction_fusion_projection` | `reconstruction_concat_projection` | Fuses base, continuous, and discrete representations for reconstruction |
| `classification_fusion_projection` | `classification_concat_projection` | Fuses representations for classification |
| `reconstruction_fused_hidden` | `hidden_reconstruction` | Latent input to the reconstruction head |
| `classification_fused_hidden` | `hidden_classification` | Latent input to the classification head |
| `reconstruction_head` | `reconstruction_head` | Creates `reconstruction` |
| `classification_head` | `classification_head` | Creates `classification_logits` |

`fusion head` is a contextual group name, not one module in the active `task_specific_concat_projection` mode.

Under canonical `direct_branch_routing`, the model sends
`continuous_prototype_context` directly to `reconstruction_head` and
`discrete_codeword_context` directly to `classification_head`. The two
projection modules remain named model components, but they do not mix these
branches in this routing mode.

## 7. Prediction, loss, and score objects

### 7.1 Model outputs

| Canonical name | Runtime field | Shape | Meaning |
| --- | --- | --- | --- |
| `reconstruction` | `recon` | `[B,L,D]` | Reconstructed input window |
| `classification_logits` | `logits` | `[B,12]` | Window-class logits |
| `raw_point_mse` | `aux.point_score_samples` reduced over `M` | `[B,L]` | Monte Carlo mean channel-wise reconstruction MSE |
| `window_point_scores` | `point_scores` | `[B,L]` | Selected point-level MSE; raw-input MSE with identity transform by default |
| `window_anomaly_scores` | `window_scores` | `[B]` | Per-window raw reconstruction MSE used by window-level triage |

Raw point MSE is the default value used to compute `window_point_scores`.
Current artifacts must declare `score_space: raw_input` and
`point_score_transform: identity`, unless the run explicitly selects a named
latent-MSE score. Clean validation sets thresholds in the selected MSE space.
The shifted-and-scaled logistic sigmoid is historical and opt-in only.

### 7.2 Stage A losses

| Canonical name | Runtime name | Active in O0 | Active in O1 |
| --- | --- | --- | --- |
| `reconstruction_loss` | `reconstruction_loss` | Yes | Yes |
| `classification_loss` | `classification_loss` | Yes | Yes |
| `two_view_contrastive_loss` | `contrastive_loss` | Yes | Yes |
| `point_score_loss` | `score_loss` | No | Yes, when the batch has enough groups |
| `stage_a_total_loss` | `total_loss` in Stage A | Yes | Yes |

`L_recon`, `L_cls`, `L_contrastive`, and `L_score_point` are mathematical aliases. Use canonical snake-case names in pseudocode and `L_*` notation in formulas.

`point_score_loss` uses `raw_point_mse` in training. The default inference path
also uses raw MSE with the identity transform. A sigmoid is not an additional
training loss and may appear only in an explicitly marked legacy run.

### 7.2.1 Point-level contrastive loss: one loss, unchanged computation

The decision on 2026-09-12 is that `point-level contrastive loss` and the loss
computed by `_compute_two_view_contrastive_loss` are the same offline loss, not
two losses added together. Keep the canonical identifier
`two_view_contrastive_loss` and the complete current computation flow.

| Name | Mapping to `two_view_contrastive_loss` |
| --- | --- |
| `point-level contrastive loss`, `point_level_contrastive_loss` | exact alias in the offline scope |
| `_compute_two_view_contrastive_loss` | runtime function that computes the same loss |
| `contrastive_loss` | runtime local-variable alias in the offline loss step |

The implemented contract remains unchanged:

1. Flatten both latent tensors `[B,L,H]` to `[B*L,H]` and keep the same positions where `synthetic_anomaly_mask == 0` in both views.
2. Normalize each token with `F.normalize` and `epsilon`, then compute the `[K,K]` cosine-similarity matrix divided by `max(contrastive_temperature, epsilon)`; `K` is the number of kept points in the whole batch.
3. Each clean anchor has one positive augmented token from the same window and point position. Every other filtered augmented token is a negative, including points from other windows in the batch.
4. Use `F.cross_entropy(logits, arange(K))`, average over anchors, and use only the clean-to-augmented direction. Return zero loss when `K = 0`.

Runtime source: `src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py`, function `_compute_two_view_contrastive_loss`.
This is not `point_score_loss` and not `online_contrastive_loss`.
Do not replace the positive/negative selection with the interpretation of
Equation (3.28) in the PDF report.
This name normalization does not create another loss for O2 or change the
ablation configuration.

### 7.3 Stage B losses

| Canonical name | Active |
| --- | --- |
| `reconstruction_loss` | Yes |
| `classification_loss` | Yes |
| `two_view_contrastive_loss` | Not by default |
| `point_score_loss` | Not by default |
| `stage_b_total_loss` | Yes |

### 7.4 Timeline scores and thresholds

| Canonical name | Meaning |
| --- | --- |
| `clean_validation_point_score_timeline` | Selected point-level MSE values mapped to the absolute entity timeline on clean validation |
| `synthetic_validation_point_score_timeline` | The same MSE timeline for synthetic validation |
| `test_point_score_timeline` | The same MSE timeline for test data |
| `offline_point_threshold` | `Q_0.99` of the default raw-MSE `clean_validation_point_score_timeline` on non-overlapping windows |
| `online_point_ewma_threshold` | `Q_0.99` of the default raw-MSE clean-validation timeline after sliding-window and absolute-index EWMA |

`offline_point_threshold_nonoverlap` is an exact schema alias in full-spec-v3. `online_ewma_point_threshold` is the runtime artifact alias of `online_point_ewma_threshold`.

### 7.5 `score_protocol`

The score story is complete only when the run records which space and
transformation it uses:

```text
score_protocol
    → score_space
    → point_score_transform
    → window_point_scores
    → threshold_artifact
    → metric computation
```

The current default is `score_space: raw_input` and
`point_score_transform: identity`. A latent-MSE or legacy sigmoid path must
use a different named protocol. Similar score names do not make those
protocols equivalent.

## 8. Checkpoint and artifact objects

### 8.1 `two_stage_run_manifest`

The manifest describes stage order, generated config paths, checkpoint paths, global epoch ranges, and the evaluation checkpoint.

### 8.2 `stage_a_best_checkpoint`

The best checkpoint selected by the Stage A trainer with the configured monitor metric. It does not yet contain the initialized final memory banks.

### 8.3 `stage_b_initialization_checkpoint`

The checkpoint created from `stage_a_best_checkpoint` after `stage_b_memory_initialization`. The runtime filename is `initializations/stage_b_init.pt`.

`stage_a_checkpoint` is not an exact alias because Stage A may have best and final checkpoints. Pseudocode must name `stage_a_best_checkpoint` explicitly.

### 8.4 `stage_b_best_checkpoint`

The best checkpoint of `stage_b_fusion_finetuning`. This is the official checkpoint for offline evaluation and online source-model loading.

`evaluation_checkpoint` is a contextual alias when the manifest points to the Stage B `best.pt`. `reference_checkpoint` is an online-context alias.

### 8.5 `threshold_artifact`

Entity-scoped artifact contains `offline_point_threshold`,
`online_point_ewma_threshold`, triage thresholds, `score_space`,
`point_score_definition`, `point_score_transform`, calibration identity,
checkpoint identity, seed, and protocol fields. Legacy sigmoid parameters are
present only when a run explicitly selects the legacy transform.

### 8.6 `offline_evaluation_record`

Per-entity record after window scores are restored to the timeline:

```text
entity_id
point_scores
point_labels
covered_point_mask
raw_num_points
evaluated_num_points
```

### 8.7 `offline_metrics`

Metric mapping computed from covered `test_point_score_timeline`, ground-truth labels, and a fixed threshold. Metric objects are not score objects.

The current THESIS names are `VUS-PR`, `VUS-ROC`, and `Affiliation F1`.
`Affiliation F1-score` is a separate `tsad-lib` report name until its exact
serialization and formula are approved. It is not an automatic alias of
`Affiliation F1`.

### 8.8 `offline_artifact_bundle`

The output group of `offline_evaluation`, including score artifacts,
`offline_metrics`, `threshold_artifact`, the uncertainty summary, provenance,
the retention manifest, and the benchmark report. This is a logical bundle; it
does not have to be one file.

### 8.9 The `tsad-lib` metric story

The library groups report metrics without changing the THESIS calculations:

```text
metric_family
    → metric
    → metric_priority
```

The bridge records these desired report rows:

| Family | Metric | Priority | Status |
| --- | --- | --- | --- |
| `VUS` | `VUS-PR@FPR-budget` | `primary_required` | `desired-contract` |
| `VUS` | `VUS-PR` | `default` | `implemented` |
| `VUS` | `VUS-ROC` | `default` | `implemented` |
| `Affiliation` | `Affiliation F1-score` | `default` | `desired-contract` |
| `FPR` | `raw-FPR` | `default` | `desired-contract` |

`metric_priority` is a reporting order, not a new mathematical operation.
`false alarm rate` and `humility` are not canonical names in these ontologies.
They must not be mapped to `raw-FPR` without a separate decision.

### 8.10 The `tsad-lib` result story

The public library wraps the THESIS objects in a small, explicit flow:

```text
run_request
    → offline_pretraining_phase
    → result
    → report
    → report_metric_row
```

`run_request` selects one `dataset`, `entity`, `method`, `method_variant`,
optional `offline_variant`, optional `online_variant`, `window`, and `seed`.
`result` contains the matching `offline_evaluation_record`, artifact paths, and
source provenance. `report` contains metric rows and the run manifest.

`result` is not an alias of `offline_evaluation_record`.
`report` is not an alias of `offline_artifact_bundle`.
They are public wrappers that preserve the THESIS objects and add library-level
selection and provenance.

### 8.11 The `tsad-lib` method story

The method graph keeps model parts and loss parts visible:

```text
method
    → method_variant
    → method_model_component
    → method_loss_component
    → method_artifact_component

method_variant
    → method_model_component
    → method_loss_component
```

For the component-count rule, `num_components` counts the union of
`method_model_component` and `method_loss_component`. It does not count
artifact components. The intended rule is:

```text
num_components(method) >= num_components(corresponding method_variant)
```

This bridge is a desired contract. It does not claim that every current THESIS
module has already been registered under these generic names.

## 9. Standard relations between objects

| Subject | Relation | Object |
| --- | --- | --- |
| `offline_pretraining_phase` | `contains` | `stage_a_multitask_pretraining` |
| `offline_pretraining_phase` | `contains` | `stage_b_fusion_finetuning` |
| `stage_a_multitask_pretraining` | `produces` | `stage_a_best_checkpoint` |
| `stage_b_memory_initialization` | `loads` | `stage_a_best_checkpoint` |
| `stage_b_memory_initialization` | `reads only` | train split |
| `stage_b_memory_initialization` | `records default` | `fusion_mode: direct_branch_routing` for the following Stage B path |
| `stage_b_memory_initialization` | `constructs` | `continuous_prototype_bank` |
| `stage_b_memory_initialization` | `constructs` | `discrete_codebook` |
| `stage_b_memory_initialization` | `constructs` | `anomaly_verification_metadata` |
| `stage_b_memory_initialization` | `produces` | `stage_b_initialization_checkpoint` |
| `stage_b_fusion_finetuning` | `loads` | `stage_b_initialization_checkpoint` |
| `stage_b_fusion_finetuning` | `produces` | `stage_b_best_checkpoint` |
| `offline_evaluation` | `loads` | `stage_b_best_checkpoint` |
| `offline_evaluation` | `computes` | `raw_point_mse` |
| score selection | `maps` | selected raw MSE to `window_point_scores`; identity is the default |
| `offline_evaluation` | `calibrates from` | `clean_validation_point_score_timeline` |
| `offline_evaluation` | `produces` | `threshold_artifact` |
| `offline_evaluation` | `produces` | `offline_artifact_bundle` |
| `dataset` | `contains` | `entity` |
| `entity` | `uses` | `file` |
| `file` | `has` | `file_extension` |
| `file` | `records` | `source_provenance` |
| `dataset_adapter` | `converts` | `series_set` |
| `series_set` | `converts to` | `raw_sequence` |
| `method` | `has` | `method_variant` |
| `method` | `has` | `method_model_component` |
| `method` | `has` | `method_loss_component` |
| `method_variant` | `has` | `method_model_component` |
| `method_variant` | `has` | `method_loss_component` |
| `metric_family` | `has` | `metric` |
| `metric` | `has` | `metric_priority` |
| `run_request` | `selects` | `dataset`, `entity`, `method`, and variant fields |
| `run_request` | `produces` | `result` |
| `result` | `contains` | `offline_evaluation_record` |
| `report` | `contains` | `offline_metrics` and `report_metric_row` |
| `score_protocol` | `selects` | `threshold_artifact` and metric computation |
| `online_tta_phase` | `inherits` | `stage_b_best_checkpoint` |
| `online_tta_phase` | `inherits` | `threshold_artifact` |

## 10. Terminology changes from older specifications

The decision on 2026-09-12 is that `point-level contrastive loss` is an exact
alias of `two_view_contrastive_loss` (§7.2.1), with unchanged semantics and the
offline model as owner.
This adds documentation mapping only. It does not change function, config,
metric, checkpoint, or artifact names, callers, or computation flow.

### 10.1 Raw-input-space MSE v4

Version 4 makes simple MSE with the identity transform the current default and
splits the old ambiguous point-score names. The operational raw
fields are `raw_input_point_mse` and `raw_input_window_mse`. The diagnostic
fields are `normalized_input_point_mse` and `normalized_input_window_mse`.
Raw input means original sensor units restored with the fitted train-only
scaler. The v4 raw protocol uses `score_space: raw_input` and
`point_score_transform: identity`; sigmoid calibration is outside the default
path and requires explicit opt-in.

`point_labels` and `window_labels` are ground truth categories. Predictions are
separate threshold outputs. A window is anomalous when any point label is
anomalous.

| Old name | Canonical name | Status | Runtime owner | Migration boundary |
| --- | --- | --- | --- | --- |
| `offline pre-training` | `offline_pretraining_phase` | unchanged, normalized identifier | benchmark runner | Docs/pseudocode identifiers |
| `Stage 1: Separate Task-Specific Training` | — | deprecated historical stage | historical runner | Do not map to active Stage A |
| `Stage 2: Zipping and Short Recovery` | — | deprecated historical stage | historical runner | Do not map to active Stage B |
| `Stage 3: Memory Initialization and Fusion Warm-Up` | `stage_b_memory_initialization` + `stage_b_fusion_finetuning` | split | active orchestrator/model | Keep the operation and training stage separate |
| `TWO_STAGE_A_PHASE_NAME` | `stage_a_multitask_pretraining` | contextual alias | orchestrator/model source | Rename the source constant after an approved migration |
| `TWO_STAGE_B_PHASE_NAME` | `stage_b_fusion_finetuning` | contextual alias | orchestrator/model source | Rename the source constant after an approved migration |
| `training_phase` | `stage_name` | renamed semantically, compatibility retained | generated config/model config parser | Do not change source before a migration plan |
| `continuous memory` | `continuous_prototype_bank` | renamed for specificity | model state | Docs/pseudocode |
| `discrete memory` | `discrete_codebook` | renamed for specificity | model state | Docs/pseudocode |
| `point-wise reconstruction score` | `window_point_scores` | renamed/refined: canonical field stores selected MSE; raw-input MSE with identity transform is the default | model output | Field remains `point_scores` |

### 10.2 Audit bridge changes

The audit adds library-level names without renaming the THESIS runtime objects:

| New bridge name | Mapping | Status |
| --- | --- | --- |
| `method_variant` | separate from `offline_variant` and `online_variant` | desired-contract |
| `dataset_adapter` → `series_set` → `raw_sequence` | conversion chain, not exact aliases | desired-contract |
| `metric_family` → `metric` → `metric_priority` | report grouping and ordering | desired-contract |
| `result` → `offline_evaluation_record` | public wrapper contains runtime record | desired-contract |
| `report` → `offline_metrics` | public wrapper contains metric output | desired-contract |
| `score_protocol` | selects score space and transform | desired-contract |

The bridge does not change the existing loss computation, checkpoint names, or
offline-to-online handoff.

## 11. Known semantic conflicts

### 11.1 Discrete token pool

`full-spec-v3` specifies injected anomaly tokens only for classes 1–11. The
current source reshapes and keeps all tokens from class windows. The ontology
keeps the canonical object `discrete_memory_initialization_token_pools_by_class`,
but its composition has two states:

- `documented-intent`: anomaly positions only for classes 1–11;
- `implemented`: all positions from each class window.

Do not call these two semantics equivalent.

### 11.2 Missing-class handling

The specification requires failure when eligible class tokens are insufficient.
The current source uses a combined fallback pool when a class is empty. This is
a behavior conflict, not a naming conflict.

### 11.3 Offline variant artifact identity

The current offline O1 config has `experiment_variant` but lacks a root
`offline_variant`. The artifact collector reads root `offline_variant` and falls
back to `O0`. Before using a threshold artifact as cross-phase evidence, verify
that `variant_name` matches the real checkpoint.

## 12. Required comparison for every new specification

A new specification must include a `Terminology changes` section. For every new
object or rename, record:

```text
old name
new canonical name
mapping type
semantic equivalence or difference
runtime owner
schema and stored data
lifecycle and callers
checkpoint and artifact impact
migration boundary
```

If evidence is not enough to choose `exact alias`, write `not an alias` or
`unknown`. Never map `phase` to `stage`, `memory bank` to retrieval output,
`window score` to a point-score timeline, or `checkpoint` to a specific
checkpoint role by assumption.

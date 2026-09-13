---
date: 2026-09-13
topic: Audit of the THESIS ontologies against the tsad-lib graph stories
status: audit
scope: offline ontology, online ontology, and the proposed tsad-lib architecture
---

# Audit of the Two Ontologies

The story begins with two older documents.
The offline ontology tells how THESIS learns and exports its state.
The online ontology tells how the frozen state is used on a causal stream.

The new `tsad-lib` story is wider.
It adds methods, variants, components, datasets, entities, files, metrics, reports, and adapters.

This audit checks whether the two ontologies already support that wider story.

The audit uses three labels:

- `pass`: the relation is present and consistent;
- `conflict`: two documents give different meanings;
- `missing contract`: the relation is needed but not formally defined.

## Short verdict

The two ontologies are coherent for the THESIS offline-to-online story.
They are not yet a complete ontology for `tsad-lib`.

The most urgent conflict is `O2`.
The `tsad-lib` proposal selects `THESIS` with `variant="O2"`, but the offline ontology defines only `O0` and `O1`.

The strongest existing alignment is the contrastive-loss story.
`point-level contrastive loss` is already an exact offline alias of `two_view_contrastive_loss`, and the ontology keeps the current batch computation.

The missing parts are the generic library graphs.
The ontologies do not yet define a formal graph for dataset files, method components, metric priorities, adapters, reports, or generic run requests.

## 1. Story: offline state moves to online state

The first story is already clear.

```text
offline_pretraining_phase
    → stage_b_best_checkpoint
    → threshold_artifact
    → online_tta_phase
```

The offline ontology defines this flow.
The online ontology inherits the same checkpoint and threshold objects.

| Relation | Audit |
| --- | --- |
| offline phase contains Stage A and Stage B | `pass` |
| Stage B produces `stage_b_best_checkpoint` | `pass` |
| offline evaluation produces `threshold_artifact` | `pass` |
| online phase loads the checkpoint and reads the threshold artifact | `pass` |
| online phase updates only `online_mlp_projector` | `pass` |

The two ontologies should remain authoritative for this THESIS-specific story.
The new `tsad-lib` ontology should reference these objects instead of renaming them.

## 2. Story: a method owns variants and components

The new graph expects:

```text
method → variant
method → component
variant → component
```

The current ontologies do not define this graph.

They define `offline_variant` and `online_variant` as experiment dimensions.
Those objects are not automatically the same as a generic `method_variant`.

The offline ontology does provide useful THESIS nodes:

```text
THESIS
    → shared_encoder
    → continuous_prototype_bank
    → discrete_codebook
    → anomaly_verification_metadata
    → reconstruction_fusion_projection (skip this node by default)
    → classification_fusion_projection (skip this node by default)
    → reconstruction_head
    → classification_head
```

It also provides loss nodes:

```text
O0
    → reconstruction_loss
    → classification_loss
    → two_view_contrastive_loss

O1
    → reconstruction_loss
    → classification_loss
    → two_view_contrastive_loss
    → balanced_point_score_loss
```

These are useful edges, but the ontology does not state whether model modules and loss terms belong to one common `component` type.

### Finding M-01 — `O2` is not defined

Status: `conflict`.

The proposal selects `variant="O2"` for `THESIS`.
The offline ontology defines `offline_variant` only as `O0` or `O1`.
The ontology also says that the contrastive loss is already active in `O0` and `O1`.

Evidence:

- `proposal-tsad-lib.md:28-36`
- `offline_pretraining_terminology_ontology.md:78-107`
- `offline_pretraining_terminology_ontology.md:314-352`

Required decision:

1. Extend `offline_variant` with `O2` and define its exact loss set, or
2. Replace `O2` in the proposal with an existing canonical variant.

Do not silently map `O2` to `O0` or `O1`.

### Finding M-02 — generic `variant` is ambiguous

Status: `missing contract`.

`RunRequest` and `ExperimentConfig` contain one field named `variant`.
The two ontologies contain separate `offline_variant` and `online_variant` dimensions.

The safe design is:

```text
method_variant       → method choice
offline_variant      → offline ablation
online_variant       → online adaptation choice
combined_run_label   → explicit pair of offline and online variants
```

`O0_A2` and `O1_A2` already show why one generic field is not enough.

### Finding M-03 — component counting has no object type

Status: `missing contract`.

The constraint

```text
num_components(method) ≥ num_components(corresponding_variant)
```

cannot be checked until the ontology defines the counted set.

The minimal safe split is:

```text
method_model_component
method_loss_component
method_artifact_component
```

The count rule must name which component type it counts.

## 3. Story: a dataset contains entities, files, and extensions

The new dataset story is:

```text
dataset → entity → file → extension
```

The two ontologies contain `entity_id` inside `raw_sequence`, `offline_evaluation_record`, `causal_window`, and `threshold_artifact`.
They do not define:

```text
dataset
file
extension
DatasetAdapter
SeriesSet
DatasetInfo
```

### Finding D-01 — dataset hierarchy is outside both ontologies

Status: `missing contract`.

The `tsad-lib` proposal supports SMD, ServerMachineDataset, NASA, SWaT, IOPS, AnomalyArchive, TSB-AD, ICCAD, and extra industrial files.
The ontologies only describe the sequence after it has entered the THESIS runtime.

They do not say which source file created an entity.
They do not say which extension was used.
They do not define archive members as entities.

The new library needs a separate data ontology.
It should connect to the existing objects with one explicit edge:

```text
dataset entity
    → raw_sequence
    → offline_window
    → offline_batch
```

`SeriesSet` must not be called an exact alias of `raw_sequence` until their schemas and lifecycles are compared.

### Finding D-02 — `entity` and `entity_id` need one canonical rule

Status: `missing contract`.

The proposal uses `entity`.
The ontologies use `entity_id`.

The minimal rule should be:

```text
entity_id = canonical runtime identity
entity     = notebook and report display field, if needed
```

If both fields remain, their relationship must be written once.

## 4. Story: a metric belongs to a family and has a priority

The new metric story is:

```text
metric_family → metric → priority
```

The offline ontology has one object named `offline_metrics`.
It names `VUS-PR`, `VUS-ROC`, and `Affiliation F1`.
It does not define `metric_family` or `priority` objects.

### Finding E-01 — required metrics do not match

Status: `conflict` and `missing contract`.

The `tsad-lib` proposal requires:

```text
VUS-PR@FPR-budget
VUS-PR
VUS-ROC
Affiliation F1-score
raw-FPR
```

The offline ontology currently names:

```text
VUS-PR
VUS-ROC
Affiliation F1
```

The ontology does not define `VUS-PR@FPR-budget`, `raw-FPR`, or metric priorities.

Evidence:

- `proposal-tsad-lib.md:128-132`
- `offline_pretraining_terminology_ontology.md:419-425`

Required decision:

1. Add these metrics to a new `tsad` metric ontology, or
2. revise the proposal to use only the existing metric contract.

Do not map `Affiliation F1` to `Affiliation F1-score` until exact naming and serialization are approved.

### Finding E-02 — metric family grouping is only inferred

Status: `missing contract`.

The graph spec groups metrics into `VUS`, `Affiliation`, and `FPR` families.
This is a useful grouping, but neither ontology declares these families.

The priority words `primary required` and `default` also belong to the new report contract.
They are not yet canonical ontology objects.

The names `false alarm rate` and `humility` do not appear in the two ontologies.
`raw-FPR` must remain a separate name until an exact alias is approved.

## 5. Story: the run produces a result and a report

The new runtime story is:

```text
run
    → RunRequest
    → Experiment
    → Result
    → Report
    → metrics.csv / metrics.md / run_manifest.json
```

The ontologies tell a different but related story:

```text
offline_evaluation
    → offline_evaluation_record
    → offline_metrics
    → offline_artifact_bundle
```

### Finding R-01 — `Result` and `Report` have no mapping

Status: `missing contract`.

`Result` and `Report` are public `tsad-lib` objects.
`offline_evaluation_record` and `offline_artifact_bundle` are THESIS offline objects.

They must not be treated as aliases.

The new ontology needs one of these explicit choices:

```text
Result → contains offline_evaluation_record
Report → contains offline_metrics and artifact references
```

or a statement that the public objects are adapters around the offline objects.

### Finding R-02 — provenance has no complete source schema

Status: `missing contract`.

The proposal requires source path, file hash, parser, config, seed, and code revision in every result.
The ontologies mention provenance inside the artifact bundle but do not define the full source-to-entity edge.

The minimum missing edge is:

```text
file
    → source_provenance
    → entity
    → raw_sequence
```

## 6. Story: points inside a batch are compared

This story passes.

The offline ontology defines the current computation for `two_view_contrastive_loss`:

```text
[B,L,H]
    → flatten to [B*L,H]
    → keep clean positions in both views
    → positive = same window and same point position
    → negatives = other augmented tokens in the whole filtered batch
    → cross entropy over all clean anchors
```

This means points from different windows in the same batch can become negatives.
The loss is not calculated as two independent per-window losses.

The ontology also keeps the correct identity boundary:

```text
point-level contrastive loss
    = two_view_contrastive_loss
    = _compute_two_view_contrastive_loss
```

It remains distinct from:

```text
point_score_loss
online_contrastive_loss
```

Evidence:

- `offline_pretraining_terminology_ontology.md:331-351`

Status: `pass`.

The only remaining question is variant ownership.
The ontology says this loss is active in `O0` and `O1`, while the proposal selects `O2`.

## 7. Story: scores keep their level and protocol

Most score boundaries are clear.

```text
raw_point_mse
    → window_point_scores
    → current_window_ewma_point_scores
    → window_point_predictions
```

The online ontology also keeps vector objects separate from endpoint compatibility fields.
This is a good boundary.

### Finding S-01 — raw-input protocol needs an explicit edge

Status: `missing contract`.

The ontologies describe both transformed point scores and the raw-input-space MSE v4 protocol.
The documents state that the protocols differ, but the graph does not name the protocol object that selects one path.

The minimal addition is:

```text
score_protocol
    → score_space
    → point_score_transform
    → threshold_artifact
    → metric computation
```

This prevents `raw_point_mse`, `raw_input_point_mse`, and `window_point_scores` from being merged by name alone.

## 8. Story: online events are recorded and updated

The online ontology has three similar names:

```text
online_event
online_event_record
online_update_event
```

The top-level table defines `online_event` as one score, triage, and adaptation lifecycle.
Section 10 defines `online_event_record` as the immutable per-window record.
Section 9 defines `online_update_event` as one atomic projector update.

### Finding O-01 — event hierarchy is not explicit

Status: `missing contract`.

The story needs one directed structure:

```text
online_event
    → online_event_record
    → optional online_update_event
```

Without this edge, readers may treat the three names as aliases.

### Finding O-02 — `VerificationResult` is used but not defined

Status: `missing contract`.

The relation table says that `verification_cycle` produces per-entry `VerificationResult`.
The ontology does not define `VerificationResult` as a canonical object.

Either define it, or replace it with an existing canonical object such as an explicit verification result record.

Do not leave the capitalized name without an object definition.

## 9. Story: module and adapter boundaries keep the code simple

The new library proposes:

```text
api
    → pipeline
    → data
    → models
    → metrics
    → reporting
```

The two ontologies describe model and runtime objects.
They do not describe module ownership or dataset adapter ownership.

This is not a contradiction.
It is a scope boundary.

The new `tsad-lib` ontology should own these edges:

```text
DatasetAdapter → SeriesSet
SeriesSet → raw_sequence
DataPipeline → offline_window
ModelRunner → method
Report → offline_metrics or metric rows
```

Each edge must be marked as `exact alias`, `wrapper`, or `conversion`.

## 10. Story: evidence status travels with each claim

The offline ontology uses:

```text
implemented
configured
tested
documented-intent
historical
unknown
```

The online ontology adds:

```text
inherited
desired-contract
```

### Finding G-01 — evidence vocabulary is not shared

Status: `missing contract`.

The difference is understandable, but a shared `tsad` audit or ontology needs one status vocabulary.

The minimal rule is:

```text
inherited
implemented
configured
tested
documented-intent
desired-contract
historical
unknown
```

The new status list must say whether `inherited` and `implemented` can appear together.

## 11. Safe mappings already supported

The following mappings are safe because the two ontologies already define them clearly:

| Object or relation | Status |
| --- | --- |
| `offline_pretraining_phase` → `stage_b_best_checkpoint` | `pass` |
| `stage_b_best_checkpoint` → `frozen_source_model` | `pass` |
| `threshold_artifact` shared from offline to online | `pass` |
| `shared_encoder` → `source_hidden` | `pass` |
| `online_mlp_projector` → `projected_hidden` | `pass` |
| `verification_buffer` contains `verification_entry` | `pass` |
| `window_point_scores` vector differs from endpoint scalar fields | `pass` |
| `point-level contrastive loss` = `two_view_contrastive_loss` | `pass` |
| `two_view_contrastive_loss` differs from `point_score_loss` | `pass` |
| `two_view_contrastive_loss` differs from `online_contrastive_loss` | `pass` |

## 12. Required bridge before implementing `tsad-lib`

The next document should be a small bridge ontology.
It should not rewrite the two THESIS ontologies.

The bridge should define these canonical object groups:

```text
method
method_variant
method_model_component
method_loss_component
dataset
entity
file
file_extension
dataset_adapter
series_set
metric_family
metric
metric_priority
run_request
result
report
source_provenance
score_protocol
```

The bridge should then state the few cross-boundary mappings:

```text
THESIS method
    → offline_variant
    → offline source model
    → two-stage checkpoints
    → offline metrics

dataset entity
    → raw_sequence
    → offline_window
    → offline_batch

offline_metrics
    → report metric rows
```

## Final audit decision

The two ontologies pass the THESIS offline-to-online story.

They pass the current point-level contrastive-loss story, including points within one batch.

They do not yet pass the complete `tsad-lib` architecture story.

Implementation should wait until `O2`, generic variant ownership, metric names, and the bridge ontology are decided.

No ontology file was modified by this audit.

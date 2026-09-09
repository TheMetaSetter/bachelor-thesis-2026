# Terminology and Naming Conventions

This note records conventions derived from the authoritative offline and online terminology ontologies.

Sources:

- [Offline Pre-training Terminology Ontology](../spec/offline_pretraining_terminology_ontology.md)
- [Online TTA Terminology Ontology](../spec/online_tta_terminology_ontology.md)

## 1. Authority and scope

The two ontology files are authoritative for object names and object relationships.

They apply to specifications, pseudocode, runtime code, tests, configuration, checkpoints, metrics, and experiment artifacts.

The online ontology inherits offline object identity and must not create a new name for an existing offline object.

Mathematical notation may be normalized separately, but runtime, configuration, and artifact names do not change only because notation changes.

## 2. Evidence status

| Status | Meaning |
| --- | --- |
| `implemented` | Current source performs the behavior. |
| `configured` | Current configuration selects the behavior. |
| `tested` | A test checks the behavior. |
| `documented-intent` | A specification requires the behavior, but source agreement is not confirmed. |
| `historical` | The name or behavior belongs only to an older workflow. |
| `inherited` | The online object is defined by the offline ontology. |
| `desired-contract` | The object belongs to a desired flow that current runtime may not implement. |
| `unknown` | Available evidence is insufficient. |

Never present documented intent or inference as implemented behavior.

## 3. Mapping rules

| Mapping type | Rule for new contracts |
| --- | --- |
| `canonical` | Use this as the official name. |
| `exact alias` | Same object and same semantics; migrate new contracts to the canonical name. |
| `contextual alias` | Use only in the stated context; do not use as a general contract name. |
| `historical name` | Keep only for reading old code or documents. |
| `not an alias` | Treat as a different object even when the names look similar. |

Names that look similar are not evidence of semantic identity.

Before mapping two names, compare schema, owner, lifecycle, callers, state mutation, checkpoint contract, artifact contract, and serialized data.

One runtime object must have one canonical name across specification versions.

Every intentional rename must record the old name, new name, mapping type, semantic difference or equivalence, runtime owner, lifecycle, callers, schema, checkpoint impact, artifact impact, and migration boundary.

## 4. Identifier style

Use `snake_case` for canonical identifiers in pseudocode, schemas, configuration keys, and technical prose.

Keep Python class names in `PascalCase`.

Keep artifact filenames exactly as defined by the runtime.

Use explicit full words for variable and configuration names when a short name could have more than one meaning.

Use a short display label only when the canonical identifier remains available in configuration, metadata, tags, or provenance.

## 5. Offline-to-online lifecycle

The canonical cross-phase flow is:

```text
offline_pretraining_phase
    -> stage_b_best_checkpoint
    -> threshold_artifact
    -> online_tta_phase
```

`offline_pretraining_phase` is the complete offline lifecycle.

`stage_a_multitask_pretraining` and `stage_b_fusion_finetuning` are training stages inside that phase.

`stage_b_memory_initialization` is a transition operation between the two training stages.

`offline_evaluation` is a post-training operation, not a third training stage.

`online_tta_phase` does not retrain the offline model.

## 6. Canonical offline names

### 6.1 Phase and variant

| Canonical name | Meaning |
| --- | --- |
| `offline_pretraining_phase` | Complete offline source-model lifecycle. |
| `offline_variant` | Offline ablation dimension with values `O0` or `O1`. |
| `O0` | `point_score_loss` is disabled in Stage A. |
| `O1` | `point_score_loss` is enabled in Stage A when the batch has sufficient groups. |
| `offline_evaluation` | Evaluation and artifact export after Stage B. |

`experiment_variant` describes detailed protocol configuration and does not replace `offline_variant` for cross-phase identity.

### 6.2 Training stages and transition

| Canonical name | Meaning |
| --- | --- |
| `stage_a_multitask_pretraining` | First training stage that learns the encoder and task heads. |
| `stage_b_memory_initialization` | One-time operation that loads `stage_a_best_checkpoint`, builds memory state, and saves `stage_b_initialization_checkpoint`. |
| `stage_b_fusion_finetuning` | Second training stage that uses frozen encoder and memory banks while training fusion projections and task heads. |

Do not map `memory initialization stage` to a training stage.

Do not map historical Stage 1, Stage 2, or Stage 3 names into the active Stage A or Stage B contract without an explicit terminology mapping.

### 6.3 Data objects

| Canonical name | Meaning |
| --- | --- |
| `raw_sequence` | One entity sequence before windowing. |
| `offline_window` | Fixed-length window extracted from `raw_sequence`. |
| `offline_batch` | Batch passed by the offline trainer to the model. |
| `synthetic_training_batch` | `offline_batch` after synthetic anomaly injection. |
| `classification_labels` | Window-level class labels. |
| `synthetic_anomaly_mask` | Point-level mask of positions actually injected. |
| `latent_tokens` | Encoder output with shape `[B, L, H]`; runtime top-level field is `hidden`. |

`x` is the actual model input field.

`x_input` and `x_clean` are not current runtime fields unless a schema explicitly stores both tensors.

`class_labels` is an exact legacy alias of `classification_labels`.

`features` is too general for a new contract.

An individual row of `latent_tokens` is a token, not a channel.

### 6.4 Model and memory state

| Canonical name | Meaning |
| --- | --- |
| `offline_source_model` | Model trained and checkpointed by the offline phase. |
| `shared_encoder` | Module that maps `offline_batch.x` to latent tokens. |
| `continuous_memory_initialization_token_pool` | Clean or normal train tokens used to build the continuous bank. |
| `discrete_memory_initialization_token_pools_by_class` | Class-indexed token pools used to build the discrete codebook. |
| `continuous_prototype_bank` | Frozen bank representing normal latent structure. |
| `discrete_codebook` | Frozen class-stratified codeword bank. |
| `anomaly_verification_metadata` | Offline-owned geometry metadata consumed by online verification. |
| `continuous_prototype_context` | Retrieval output from the continuous bank; runtime field `prototype_context`. |
| `discrete_codeword_context` | Retrieval output from the discrete codebook; runtime field `quantized_hidden`. |
| `reconstruction_fusion_projection` | Projection that fuses representations for reconstruction. |
| `classification_fusion_projection` | Projection that fuses representations for classification. |
| `reconstruction_fused_hidden` | Reconstruction-head input representation. |
| `classification_fused_hidden` | Classification-head input representation. |
| `reconstruction_head` | Module that produces `reconstruction`. |
| `classification_head` | Module that produces `classification_logits`. |

`continuous memory`, `continuous bank`, and `continuous prototypes` are contextual aliases of `continuous_prototype_bank`.

`discrete memory` and `discrete bank` are contextual aliases of `discrete_codebook`.

`prototype_context` and `quantized_hidden` are retrieval outputs, not memory-bank aliases.

`fusion head` is a group name, not one module in the active task-specific projection mode.

### 6.5 Outputs, scores, losses, and thresholds

| Canonical name | Runtime or meaning |
| --- | --- |
| `reconstruction` | Runtime field `recon`; reconstructed input window. |
| `classification_logits` | Runtime field `logits`; window-class logits. |
| `raw_point_mse` | Intermediate channel-wise reconstruction MSE before score transformation. |
| `window_point_scores` | Runtime field `point_scores`; transformed point-level anomaly score. |
| `window_anomaly_scores` | Runtime field `window_scores`; per-window raw reconstruction MSE. |
| `reconstruction_loss` | Reconstruction loss. |
| `classification_loss` | Classification loss. |
| `two_view_contrastive_loss` | Two-view contrastive loss; runtime name `contrastive_loss`. |
| `point_score_loss` | Stage A point-score loss; runtime name `score_loss`. |
| `stage_a_total_loss` | Total Stage A loss. |
| `stage_b_total_loss` | Total Stage B loss. |
| `clean_validation_point_score_timeline` | Transformed point-score timeline on clean validation. |
| `synthetic_validation_point_score_timeline` | Transformed point-score timeline on synthetic validation. |
| `test_point_score_timeline` | Transformed point-score timeline on test. |
| `offline_point_threshold` | Offline point threshold from the specified clean-validation timeline. |
| `online_point_ewma_threshold` | Online EWMA point threshold created offline and consumed online. |

`raw_point_mse` is not the same object as `window_point_scores`.

The shifted-and-scaled logistic sigmoid is applied after raw MSE computation for inference, timeline construction, and threshold calibration.

It is not an additional training loss term.

### 6.6 Checkpoints and artifacts

| Canonical name | Meaning |
| --- | --- |
| `two_stage_run_manifest` | Manifest of stage order, config paths, checkpoint paths, epoch ranges, and evaluation checkpoint. |
| `stage_a_best_checkpoint` | Best checkpoint selected by the Stage A monitor; it does not contain final initialized memory. |
| `stage_b_initialization_checkpoint` | Checkpoint after memory initialization and before Stage B finetuning. |
| `stage_b_best_checkpoint` | Best Stage B checkpoint used for offline evaluation and online source-model loading. |
| `threshold_artifact` | Entity-scoped thresholds, score-transform parameters, identity, seed, and protocol fields. |
| `offline_evaluation_record` | Per-entity timeline record with scores, labels, coverage, and point counts. |
| `offline_metrics` | Metrics computed from covered test timelines and fixed thresholds. |
| `offline_artifact_bundle` | Logical group of score artifacts, metrics, threshold artifact, uncertainty, provenance, retention, and report outputs. |

Use `stage_a_best_checkpoint` and `stage_b_best_checkpoint` when the checkpoint role matters.

Do not use the generic name `checkpoint` when a specific checkpoint role is known.

## 7. Canonical online names

### 7.1 Phase and variants

| Canonical name | Meaning |
| --- | --- |
| `online_tta_phase` | Causal online test-time adaptation lifecycle. |
| `online_variant` | Online dimension with values `A0`, `A1`, or `A2`. |
| `A0` | Inference only; no projector or optimizer update. |
| `A1` | Update only through the verified non-empty PNN reconstruction path. |
| `A2` | Guarded hard-old update or verified non-empty PNN update with `online_contrastive_loss`. |
| `O0_A2` or `O1_A2` | Combined offline and online run label, not a new online variant. |

### 7.2 Inherited offline objects

`stage_b_best_checkpoint`, `continuous_prototype_bank`, `discrete_codebook`, `anomaly_verification_metadata`, `reconstruction_head`, and `classification_head` keep their offline names.

`reference_checkpoint_path` is a configuration field pointing to `stage_b_best_checkpoint`, not a new checkpoint type.

`frozen_source_model` is the frozen `offline_source_model` restored from `stage_b_best_checkpoint`.

`task.threshold_artifact_path` is the canonical configuration field pointing to the matching `threshold_artifact`.

Online reads `online_point_ewma_threshold` from the offline artifact and does not recalibrate it from the test stream.

### 7.3 Input, representation, and output objects

| Canonical name | Meaning |
| --- | --- |
| `causal_window` | Latest online window containing only observations available at the current cursor. |
| `source_hidden` | Frozen source encoder output; runtime field `reference_hidden`. |
| `projected_hidden` | Output of `online_mlp_projector(source_hidden)`. |
| `online_mlp_projector` | Only mutable online module in accepted A1/A2 updates. |
| `online_model_outputs` | Stable output contract inherited from the offline model. |
| `active_ewma_point_scores` | Absolute-index keyed EWMA state for points in the active causal window. |
| `current_window_ewma_point_scores` | EWMA vector for the current causal window. |
| `window_point_predictions` | Binary prediction vector after thresholding. |
| `input_window_score` | Window-level input-space reconstruction MSE for triage. |
| `latent_window_score` | Latent-memory score used with the latent threshold band. |
| `online_point_ewma_threshold` | Point-level threshold applied to online EWMA anomaly scores. |

`window_point_predictions` is the canonical vector name.

`point_level_binary_predictions` is an exact alias.

The scalar runtime fields `raw_point_score`, `ewma_point_score`, and `prediction` are endpoint compatibility fields, not vector aliases.

### 7.4 Triage and verification

`triage_region` has exactly four canonical values:

```text
normal
hard_old_normality
gray_zone
strong_anomaly
```

`triage_region` is determined by `input_window_score`, `latent_window_score`, `input_window_threshold`, `latent_window_low_threshold`, and `latent_window_high_threshold`.

`hard_old_normality` is a triage region, not a buffer entry or verification result.

`hard_old_interval_guard` prevents overlapping accepted hard-old updates.

`verification_buffer` is the container for admitted gray-zone entries.

`verification_entry` is one admitted gray-zone causal window.

`verification_cycle` verifies stored entries and commits adaptation status changes.

`nearest_codeword_ids`, `nearest_codeword_distances`, `known_anomaly_mask`, `continuous_signature_ids`, `recurrent_signature_set`, and `pnn_mask` are canonical verification tensors or sets.

`pnn_verified` is internal control state, not a canonical object and not a `triage_region` value.

### 7.5 Losses and state

| Canonical name | Meaning |
| --- | --- |
| `hard_old_reconstruction_loss` | Hinge loss that pushes online window score below `input_window_threshold`. |
| `pnn_reconstruction_loss` | Masked reconstruction loss on positions selected by `pnn_mask`. |
| `online_contrastive_loss` | Source-consistency regularization used by accepted A2 events. |
| `online_total_loss` | Event loss selected by the online variant and event path. |
| `online_update_event` | Atomic update with one finite loss, one backward pass, and one optimizer step. |
| `online_event_record` | Immutable per-window record after scoring and optional update. |
| `online_runtime_state` | Resumable stream state without optimizer moments or `recurrent_signature_set`. |

Only `online_mlp_projector` may be mutated by `online_update_event`.

## 8. Forbidden automatic mappings

Never automatically map these pairs:

- `offline_point_threshold` to `online_point_ewma_threshold`.
- `input_window_threshold` to a point threshold.
- `verification_buffer` to `verification_entry`.
- `window_point_scores` to scalar `endpoint_point_score`.
- `frozen_source_model` to `online_mlp_projector`.
- `stage_a_best_checkpoint` to `stage_b_best_checkpoint`.
- `offline_window` to `causal_window`.
- `continuous_prototype_bank` to `prototype_context`.
- `discrete_codebook` to `quantized_hidden`.
- `pnn_mask` to `pnn_verified`.
- `triage_region` to `event_decision` when the latter also contains adaptation or verification state.
- `TTLBuffer` to `VerificationBuffer`.

## 9. Known specification and runtime conflicts

The ontology records conflicts instead of silently resolving them.

The intended discrete memory token pool uses injected anomaly positions for classes `1..11`, while the current source collects all tokens from each class window.

The intended missing-class behavior fails when eligible tokens are unavailable, while current source uses a combined fallback pool.

Some O1 configurations may omit root `offline_variant`, causing artifact collection to fall back to `O0`; artifact identity must be checked before cross-phase use.

The online scalar endpoint fields remain for compatibility, but canonical runtime behavior must use vector fields.

## 10. Rule for prompts containing “name”

Whenever a user prompt contains the word `name` in the context of a variable, configuration key, object, checkpoint, metric, artifact, run, or file, first read both ontology files in full.

Use the canonical names and mappings from both ontologies as the grounding source.

Check whether the term belongs to the offline phase, online phase, inherited cross-phase lineage, runtime compatibility layer, or historical terminology.

If the evidence is ambiguous, report `unknown` or `not an alias` and ask for clarification before changing a name.

Do not silently map similar names.

If the user asks for a new name, preserve the ontology's canonical object identity and document the mapping from old name to new name.

If the user asks for a W&B run name, use canonical phase, variant, entity, seed, and stage tokens as available.

The ontologies do not define one canonical W&B display-name format, so any compact W&B naming scheme must be labelled as a proposal rather than an implemented ontology rule.

Keep the full experiment identity in configuration, tags, metadata, or provenance even when the W&B display name is shortened.

The selected project convention for W&B smoke-run display names is:

```text
smk-<phase>-<method_or_variant>-<entity>-s<seed>
```

Selected examples are:

```text
smk-off-O0-e1_6-s8
smk-on-KA-A2-e3_9-s36
smk-off-KA-e3_4-s6
```

In these display names, `off` and `on` are short display tokens for the offline and online phases.

The canonical phase identifiers remain `offline_pretraining_phase` and `online_tta_phase`.

`O0` is the canonical `offline_variant` value.

`A2` is the canonical `online_variant` value.

`KA` is a display token for the `kmeans_ad` method and is not a new ontology object.

The `KA-A2` sequence combines a method display token with the online variant; it is not a new value of `online_variant`.

The full stage, checkpoint, window, protocol, and artifact identity remains in configuration, tags, metadata, or provenance.

## 11. Required terminology checklist

Before writing or reviewing a new specification, variable, configuration key, checkpoint, metric, artifact, or experiment name:

1. Read both ontology files.
2. Identify the object type and lifecycle.
3. Select the canonical `snake_case` name.
4. Check all exact, contextual, historical, and forbidden mappings.
5. Compare schema, owner, callers, state mutation, checkpoint contract, and artifact contract.
6. Record terminology changes and migration boundaries.
7. Label behavior as implemented, configured, tested, documented-intent, inherited, desired-contract, historical, or unknown.
8. Preserve the offline-to-online lineage.

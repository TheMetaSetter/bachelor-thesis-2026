---
date: 2026-04-18 17:55:33 +07 +0700
author: TheMetaSetter
git_commit: c70a3dbe70bf60b74cda147917fde24a418ee79d
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for label refurbishment, normal-only reconstruction loss, and AUC-PR scheduler monitoring in the offline multitask thesis model"
tags: [detail, multitask, anomaly-detection, label-refurbishment, anomaly-mask, scheduler]
status: complete
last_updated: 2026-04-18
last_updated_by: TheMetaSetter
---

# Detailed Plan: Label Refurbishment, Normal-Only Reconstruction Loss, and AUC-PR Scheduler Monitoring

## Scope

This document specifies the implementation detail plan for three connected changes in the current offline multitask thesis pipeline:

1. adding label refurbishment to the binary clean-versus-anomalous classification objective;
2. computing reconstruction loss only over normal time steps so that synthetic anomalous time steps do not dominate reconstruction supervision; and
3. switching the learning-rate scheduler monitor from synthetic validation ROC-AUC to synthetic validation PR-AUC.

This plan is written against the repository as it exists at commit `c70a3dbe70bf60b74cda147917fde24a418ee79d`. It is grounded directly in the current research note [research-modifying-offline-multitask-path-for-label-refurbishment-normal-only-reconstruction-and-aupr-scheduler.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/04-18-2026/research/research-modifying-offline-multitask-path-for-label-refurbishment-normal-only-reconstruction-and-aupr-scheduler.md). No task-specific structure document currently exists for this topic, so this detailed plan serves as the first implementation-grade breakdown for the requested change set.

The plan intentionally stays within the current binary multitask classification contract. It does not expand the classifier to one class per anomaly family, because that would materially widen the task definition and would no longer be the smallest coherent modification to the present codebase.

## Phase 1 - Extend the explicit configuration and runtime contracts

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make the requested behaviors configurable, reproducible, and ablation-friendly without creating hidden branches in the runtime. The repository preference is explicit YAML-driven control with minimal codepaths. Therefore, the first phase defines the new switches and their validation rules before the model behavior is changed.

This phase does not yet modify losses. It establishes the contract by which the model and trainer will later interpret label refurbishment, normal-only reconstruction masking, and AUC-PR scheduler monitoring.

### File-level edits

[configs/model/thesis_multitask.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/model/thesis_multitask.yaml)
- Add explicit model-level fields for the new classification and reconstruction behaviors.
- Keep all new fields adjacent to the current classification and objective configuration fields so that readers can inspect the entire multitask objective surface in one place.

[configs/task/multitask_tsad.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/task/multitask_tsad.yaml)
- Add task-level switches only if the team prefers these behaviors to be interpreted as training-policy choices instead of model-objective choices.
- Do not duplicate the same knob in both model and task configs. Choose one ownership location and keep it canonical.

[src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py)
- Validate the new refurbishment and masking fields.
- Extend scheduler monitor validation so `val_synth_pr_auc` is an accepted value.
- Keep validation scalar and explicit, consistent with the rest of the repository.

[configs/experiment/smd_multitask_rtx3090_seed11_machine_2_1_val_synth_roc_auc.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/smd_multitask_rtx3090_seed11_machine_2_1_val_synth_roc_auc.yaml)
- Rename or replace this experiment with an AUC-PR-monitored variant whose name, tags, and scheduler monitor are internally consistent.
- Apply the same update to any other experiment YAML that currently names and configures `val_synth_roc_auc`.

### Explicit edit content

The recommended first explicit fields are:

- `use_label_refurbishment: false`
- `refurbishment_alpha: 0.0`
- `refurbishment_beta: 0.0`
- `reconstruction_normal_only: false`

These values should default to the current behavior so that old experiments remain readable and reproducible. The scheduler monitor extension should accept:

- `val_loss`
- `val_synth_roc_auc`
- `val_synth_pr_auc`

The configuration validator should enforce:

- `use_label_refurbishment` is boolean;
- `reconstruction_normal_only` is boolean;
- `refurbishment_alpha` is numeric and in `[0, 1]`;
- `refurbishment_beta` is numeric and in `[0, 1]`;
- if `use_label_refurbishment` is false, `refurbishment_alpha` and `refurbishment_beta` may still be present but should not alter runtime behavior;
- scheduler monitor validation includes `val_synth_pr_auc`.

### Interface and contract definitions

Dataset contract:
- unchanged
- batches remain dictionaries with `x`, `point_labels`, `mask`, `timestamps`, and `meta`
- synthetic augmentation may continue to add `classification_labels`, `synthetic_anomaly_mask`, and `augmentation_metadata`

Encoder contract:
- unchanged
- the encoder still returns the thesis-facing hidden representation through the current `hidden` key

Model contract:
- extended but not replaced
- `ThesisMultitaskModel` must accept the new configuration fields in its constructor
- hard labels must remain present as `classification_labels`
- loss-only soft targets must remain internal to the model unless there is a debugging need to expose them in `aux`

Task contract:
- unchanged at the semantic level
- the task remains binary clean-versus-anomalous classification with sequence reconstruction
- no anomaly-family multiclass objective is introduced in this change set

Training engine contract:
- unchanged in interface
- the trainer continues to consume epoch-level metrics by name
- the scheduler monitor remains a string-valued config field resolved against aggregated epoch metrics

### Design pattern application

Composition over inheritance:
- the new behavior is added to the existing model and config validation surfaces
- no subclass such as `RefurbishedThesisMultitaskModel` should be introduced

Adapter pattern for encoders:
- unchanged
- the encoder remains an internal representation adapter that exposes the same hidden-state contract to the multitask heads

Strategy pattern for tasks:
- task behavior remains controlled by explicit flags rather than new trainer branches
- refurbishment and masked reconstruction behave like objective strategies selected by config

Registry or factory:
- runtime construction remains registry-driven through `scripts/train.py`
- the registry does not need new model names for these changes

### Risk mitigation steps

Prototype redundancy:
- do not modify prototype-bank sizing or retrieval rules in the same patch
- keep classification and reconstruction changes isolated to label and loss computation

Fusion collapse:
- do not modify `alpha`, `beta`, warmup, or gate regularization while introducing the new objectives

Adaptation contamination:
- do not touch [src/models/online_adaptation.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/online_adaptation.py)

Projector drift:
- not applicable to the offline objective change itself
- keep the online projector path unchanged so no adaptation-state behavior is confounded with offline loss behavior

Evaluation metric inflation:
- treat `val_synth_pr_auc` as a scheduler-monitoring surface, not as proof of better anomaly detection quality
- keep clean evaluation reporting unchanged in this phase

### Test plan and validation steps

Unit tests:
- extend [tests/test_config_loading.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_config_loading.py) for the new boolean and numeric fields
- add validation cases for `val_synth_pr_auc`

Integration tests:
- ensure experiment config loading succeeds when the new fields are present and defaults remain backward compatible

Validation steps:
- load the updated model config
- load at least one updated experiment config
- verify old scheduler-free and old-loss configs still parse

### Acceptance criteria

- the new objective switches are explicit in YAML and validated in `src/core/config.py`
- `val_synth_pr_auc` is accepted as a scheduler monitor
- old experiments still load with default behavior unchanged
- no new model name or trainer codepath is introduced

## Phase 2 - Implement binary label refurbishment inside the offline thesis model

### Phase summary tied to thesis objectives

The thesis objective in this phase is to reduce over-confident binary classification on synthetic anomalies without widening the current task into a new multiclass anomaly-family problem. The implementation must preserve the present clean-versus-anomalous supervision contract while allowing the classification objective to consume a softened target distribution inspired by the RedLamp formulation.

This phase should change only the classification-loss calculation and any immediately adjacent helper logic in the thesis multitask model.

### File-level edits

[src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask.py)
- Add constructor fields for refurbishment switches and parameters.
- Add one internal helper that converts hard binary labels into probability targets.
- Modify `_compute_classification_loss` so that it uses either the original hard-label cross entropy or a probability-target formulation, depending on configuration.
- Keep `classification_labels` hard and unchanged for logging and metric computation.

### Explicit edit content

The implementation should follow this exact ownership rule:

- `batch["classification_labels"]` remains a rank-1 integer tensor of shape `[B]`;
- the model internally derives `classification_target_probabilities` of shape `[B, 2]` only when refurbishment is enabled;
- the trainer, logger, and metric code must continue to see hard labels only.

For the binary task, the refurbished target can be defined as:

- normal window label `0` becomes `[1 - beta, beta]`
- anomalous window label `1` becomes `[alpha, 1 - alpha]`

This binary form is the correct minimal analogue of the RedLamp idea in the present codebase. It preserves the meaning of:

- `alpha`: probability mass shared from the anomalous class toward the normal class
- `beta`: probability mass shared from the normal class toward the anomalous class

The classification-loss implementation should be:

- hard-label `F.cross_entropy` when `use_label_refurbishment` is false;
- probability-target cross entropy when `use_label_refurbishment` is true, implemented explicitly with `F.log_softmax(logits, dim=-1)` and a negative target-weighted sum.

The code should not:

- change `num_classes`;
- reinterpret augmentation metadata as family labels;
- route classification metrics through soft labels.

### Interface and contract definitions

Dataset contract:
- unchanged

Encoder contract:
- unchanged

Model contract:
- `training_step`, `validation_step`, and `synthetic_validation_step` still return the same top-level keys
- `loss_terms["classification_loss"]` remains scalar
- if a debugging surface is needed, soft targets may be exposed only through `outputs["aux"]` or `step_output["batch"]` under a clearly named optional field

Task contract:
- still binary
- refurbishment changes confidence, not class cardinality

Training engine contract:
- unchanged
- classification metrics still compare predicted hard classes with hard labels

### Design pattern application

Composition over inheritance:
- add one helper inside the existing model instead of creating a second classification-loss class

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- `_compute_classification_loss` becomes configuration-switchable between two target strategies while preserving one public stage-step interface

Registry or factory:
- unchanged

### Risk mitigation steps

Prototype redundancy:
- do not use refurbishment as a reason to retune prototype usage or diversity weights in the same patch

Fusion collapse:
- monitor existing `train_alpha`, `train_beta`, and gate-related metrics to ensure classification softening does not indirectly destabilize fusion

Adaptation contamination:
- the refurbished target must remain an offline-only behavior in the thesis multitask model

Projector drift:
- unchanged because the online projector path is not modified

Evaluation metric inflation:
- keep PR-AUC and ROC-AUC computed from hard labels only
- do not report softened-label metrics as evaluation metrics

### Test plan and validation steps

Unit tests:
- add a targeted test that verifies the soft target generated from `[0, 1]` labels matches the configured `alpha` and `beta`
- add a test that disables refurbishment and confirms equality with the old hard-label loss path

Integration tests:
- extend [tests/test_one_multitask_train_step.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_one_multitask_train_step.py) so one forward-backward pass succeeds with refurbishment enabled
- verify gradients still reach the encoder, prototype branches, and fusion logits

Validation steps:
- run one training-step test with `use_label_refurbishment: true`
- run one training-step test with `use_label_refurbishment: false`

### Acceptance criteria

- the thesis multitask model supports optional binary label refurbishment without changing `num_classes`
- `classification_labels` remain hard labels for logging and metric computation
- one training-step integration test passes with refurbishment enabled
- disabling refurbishment reproduces the old classification-loss behavior

## Phase 3 - Implement normal-only reconstruction masking in the offline thesis model

### Phase summary tied to thesis objectives

The thesis objective in this phase is to train the reconstruction branch to focus on normal temporal behavior rather than on synthetically corrupted time steps. The implementation must use the existing anomaly-mask surface and must preserve the present point-score computation contract so that evaluation remains stable.

This phase changes the reconstruction-loss definition only. It does not change the forward reconstruction output or the point-score surface used by the evaluator.

### File-level edits

[src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask.py)
- Modify `_compute_reconstruction_loss`.
- Add a small helper to build a normal-time-step mask from `synthetic_anomaly_mask`, or from `point_labels` if the broader masking policy is selected.
- Keep point-score computation in `forward()` unchanged unless a later research decision explicitly changes evaluation semantics.

### Explicit edit content

The reconstruction masking rule should be:

- read `synthetic_anomaly_mask` of shape `[B, L]`;
- convert it to `normal_time_mask = 1 - synthetic_anomaly_mask`;
- unsqueeze to `[B, L, 1]`;
- cast to the dtype of `outputs["recon"]`;
- compute squared reconstruction error on `[B, L, D]`;
- multiply by the expanded normal mask;
- divide by the number of active normal cells, not by the full tensor size.

This normalization rule is important because it preserves the loss scale when the number of anomalous time steps varies from batch to batch.

The implementation should fall back to the current full mean squared error when:

- `reconstruction_normal_only` is false;
- the batch does not contain `synthetic_anomaly_mask`;
- or the mask would yield zero active normal cells.

The point-score computation in `forward()` should remain:

- `torch.mean((recon - batch["x"]) ** 2, dim=-1)`

because the evaluator and downstream thresholding already depend on that contract. The training loss and the inference-time anomaly score are allowed to differ here, provided that the difference is explicit and documented.

### Interface and contract definitions

Dataset contract:
- unchanged
- the existing augmentation path already supplies `synthetic_anomaly_mask`

Encoder contract:
- unchanged

Model contract:
- reconstruction output shape remains `[B, L, D]`
- `loss_terms["reconstruction_loss"]` remains scalar
- `outputs["point_scores"]` remains unmasked reconstruction error per time step

Task contract:
- unchanged
- the task still combines reconstruction and binary classification

Training engine contract:
- unchanged

### Design pattern application

Composition over inheritance:
- masked reconstruction is one branch inside `_compute_reconstruction_loss`, not a separate model class

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- the reconstruction target remains the same input sequence, but the loss-reduction strategy becomes config-selectable

Registry or factory:
- unchanged

### Risk mitigation steps

Prototype redundancy:
- do not reinterpret reduced reconstruction loss as evidence that prototype usage has improved

Fusion collapse:
- inspect whether masking normal-only reconstruction shifts `beta` unexpectedly toward the classification-preferred geometry

Adaptation contamination:
- keep this change offline-only

Projector drift:
- unchanged

Evaluation metric inflation:
- do not mask evaluation-time point scores in the same patch
- preserve evaluator comparability across old and new checkpoints

### Test plan and validation steps

Unit tests:
- add a targeted reconstruction-loss test with a hand-constructed batch where anomalous time steps are known and the expected masked mean squared error can be computed exactly
- add a second test confirming that the loss reverts to the old full MSE when masking is disabled

Integration tests:
- extend [tests/test_one_multitask_train_step.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_one_multitask_train_step.py) for the new loss mode
- confirm that synthetic validation and clean validation still execute

Validation steps:
- run shape and train-step tests
- run one evaluation smoke test to confirm that the evaluator remains unaffected

### Acceptance criteria

- enabling `reconstruction_normal_only` masks anomalous time steps out of the reconstruction loss
- the masked loss uses denominator normalization by active normal cells
- disabling the flag reproduces the old full MSE behavior
- evaluator-facing `point_scores` remain unchanged in shape and meaning

## Phase 4 - Switch scheduler monitoring to synthetic validation PR-AUC and keep trainer semantics coherent

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make learning-rate adaptation follow a more imbalance-appropriate classification surface while preserving the current clean-validation checkpointing semantics unless explicitly changed later. The codebase already computes synthetic validation PR-AUC, so the implementation should reuse the existing trainer aggregation path rather than add new scheduler-specific metrics.

### File-level edits

[src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py)
- extend accepted scheduler monitors to include `val_synth_pr_auc`

[scripts/train.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/train.py)
- no logic change should be required beyond ensuring documentation and tests cover the new monitor
- keep the generic rule that non-`val_loss` monitors use scheduler mode `"max"`

[src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/trainer.py)
- keep scheduler stepping generic
- do not change best-checkpoint selection in the first implementation

[configs/experiment/](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/)
- update experiment names, tags, and monitor fields so they consistently indicate AUC-PR monitoring

### Explicit edit content

The intended scheduler block becomes:

```yaml
optimizer:
  learning_rate: 0.001
  weight_decay: 0.0
  scheduler:
    scheduler_name: reduce_on_plateau
    monitor_metric: val_synth_pr_auc
    factor: 0.5
    patience: 20
    threshold: 0.0001
    threshold_mode: rel
    cooldown: 3
    min_lr: 1.0e-5
```

The trainer should continue to:

- aggregate `val_synth_pr_auc` through the existing classification-metric pipeline;
- step the scheduler from whatever monitor string the config specifies;
- save best checkpoints from clean `val_loss` unless a separate policy change is intentionally introduced.

### Interface and contract definitions

Dataset contract:
- unchanged

Encoder contract:
- unchanged

Model contract:
- unchanged

Task contract:
- unchanged

Training engine contract:
- epoch metrics remain the scheduler-monitoring interface
- scheduler monitor names are resolved against already aggregated metrics
- best-checkpoint selection remains a separate policy from scheduler monitoring

### Design pattern application

Composition over inheritance:
- scheduler policy remains composed through config, builder, and trainer

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- not directly modified, but synthetic validation metrics remain a stage strategy selected through the existing model interface

Registry or factory:
- unchanged

### Risk mitigation steps

Prototype redundancy:
- not directly implicated

Fusion collapse:
- do not retune objective weights while switching scheduler monitors

Adaptation contamination:
- no online adaptation code changes

Projector drift:
- unchanged

Evaluation metric inflation:
- make explicit in documentation that PR-AUC is used for scheduler control, not necessarily for best-checkpoint selection
- keep synthetic validation deterministic through the existing fixed-seed injector so scheduler signals remain comparable across epochs

### Test plan and validation steps

Unit tests:
- extend [tests/test_learning_rate_scheduler.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_learning_rate_scheduler.py) so the builder accepts `val_synth_pr_auc`
- add a trainer test that verifies `scheduler_monitor_val_synth_pr_auc` is emitted into epoch metrics

Integration tests:
- ensure a short training run with synthetic validation can step the scheduler from PR-AUC
- confirm checkpoint saving still tracks clean `val_loss`

Validation steps:
- run scheduler unit tests
- run one trainer integration test
- run config-loading tests for the new scheduler monitor

### Acceptance criteria

- `val_synth_pr_auc` is accepted by config validation
- the trainer can step `ReduceLROnPlateau` from `val_synth_pr_auc`
- epoch metrics record the selected PR-AUC monitor value
- best-checkpoint selection remains unchanged unless intentionally revised in a later task

## Phase 5 - Validate the complete change set and keep repository documentation aligned

### Phase summary tied to thesis objectives

The thesis objective in this phase is to finish the change set with explicit, minimal tests and documentation so future experiments can be reproduced and understood without reading the implementation diff line by line. This phase closes the loop between objective design, runtime behavior, and experiment reproducibility.

### File-level edits

[tests/test_config_loading.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_config_loading.py)
- add coverage for all new objective flags and scheduler monitor acceptance

[tests/test_one_multitask_train_step.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_one_multitask_train_step.py)
- add one-step training coverage for:
  - refurbishment only
  - normal-only reconstruction masking only
  - combined refurbishment and masking

[tests/test_learning_rate_scheduler.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_learning_rate_scheduler.py)
- extend scheduler coverage for the PR-AUC monitor

[documents/logs/04-18-2026/research/research-modifying-offline-multitask-path-for-label-refurbishment-normal-only-reconstruction-and-aupr-scheduler.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/04-18-2026/research/research-modifying-offline-multitask-path-for-label-refurbishment-normal-only-reconstruction-and-aupr-scheduler.md)
- append a short follow-up note after implementation if implementation decisions settle the open questions

### Explicit edit content

The final validation matrix should demonstrate:

- config parsing correctness;
- numerical correctness for soft targets and masked reconstruction reduction;
- one forward-backward optimizer step with the new behaviors enabled;
- scheduler stepping from PR-AUC; and
- unchanged evaluation and checkpoint-selection contracts where they were intentionally preserved.

### Interface and contract definitions

Dataset contract:
- unchanged and regression-tested

Encoder contract:
- unchanged and implicitly regression-tested through train-step execution

Model contract:
- unchanged at the stage-step interface
- extended only in constructor configuration and internal loss behavior

Task contract:
- remains binary multitask offline TSAD

Training engine contract:
- unchanged except for the newly accepted scheduler monitor name

### Design pattern application

Composition over inheritance:
- tests should confirm new behavior without introducing extra model classes

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- tests should exercise the new objective strategies through configuration flags

Registry or factory:
- experiment configs should continue to build through the same registry path

### Risk mitigation steps

Prototype redundancy:
- inspect metrics only for regressions, not for performance claims

Fusion collapse:
- keep current fusion observability metrics in the training logs and review them during smoke runs

Adaptation contamination:
- verify no online-model files changed

Projector drift:
- verify no online projector code changed

Evaluation metric inflation:
- verify evaluator tests and point-score contracts still pass

### Test plan and validation steps

Recommended test sequence:

1. run config-loading tests;
2. run targeted multitask unit tests;
3. run one training-step integration tests;
4. run scheduler tests;
5. run checkpoint and evaluator smoke tests if any trainer contract was touched beyond monitor acceptance.

Recommended minimum command set:

```bash
pytest -q tests/test_config_loading.py tests/test_one_multitask_train_step.py tests/test_learning_rate_scheduler.py
```

If trainer or checkpoint behavior is adjusted further, add:

```bash
pytest -q tests/test_multitask_validation_alignment.py tests/test_checkpoint_roundtrip.py
```

### Acceptance criteria

- all new objective switches are covered by tests
- the binary multitask train-step passes with the requested behaviors enabled
- scheduler tests pass with `val_synth_pr_auc`
- no online adaptation modules are changed
- the repository retains one-model-one-file readability for the offline thesis model

## Final Acceptance Criteria

- The current binary thesis multitask model supports optional label refurbishment without expanding to anomaly-family multiclass classification.
- The current offline reconstruction loss can be restricted to normal time steps using the existing anomaly-mask surface.
- The scheduler can monitor `val_synth_pr_auc` through the existing epoch-metric aggregation path.
- Best-checkpoint selection remains an explicit, separate policy and is not silently changed by the scheduler-monitor update.
- All implementation changes remain readable, explicit, YAML-driven, and colocated with the owning model or runtime assembly layer.

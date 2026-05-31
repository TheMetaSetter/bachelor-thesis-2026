---
date: 2026-04-17 21:59:04 +07+0700
author: Artificial Intelligence Agent
git_commit: 70e8186778862f559bff6c73af31dd2ad5a25327
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for a symmetric three-layer offline thesis model and multi-seed experiment execution"
tags: [detail, time-series, anomaly-detection, multitask, architecture, experiments]
status: complete
last_updated: 2026-04-17
last_updated_by: Artificial Intelligence Agent
---

# Detailed Plan: Symmetric Three-Layer Offline Thesis Model and Multi-Seed Experiment Execution

## Scope

This document specifies the implementation detail plan for two tightly related changes in the current offline thesis pipeline:

1. changing the active offline multitask thesis model from a fixed two-linear-layer encoder and two-linear-layer heads into a symmetric three-linear-layer design, where the encoder, reconstruction head, and classification head all use the same depth;
2. adding a controlled experiment-execution surface for running multiple seed-specific experiments with isolated outputs and logging, while keeping the repository readable, reproducible, and aligned with the existing single-GPU runtime.

The plan is written against the repository as it exists at commit `70e8186778862f559bff6c73af31dd2ad5a25327`. It assumes the current active offline model remains [src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask.py), the active offline entrypoint remains [scripts/train.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/train.py), and the active long-run configuration remains [configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-full__w100__seed7__rtx3090.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-full__w100__seed7__rtx3090.yaml).

## Phase 1 - Generalize the offline thesis architecture to an explicit symmetric depth contract

### Phase summary tied to thesis objectives

The thesis objective in this phase is to preserve the existing dual-prototypical multitask learning structure while making the encoder, reconstruction head, and classification head obey a common depth contract. The implementation should continue to satisfy the thesis-facing hidden-state contract and the one-model-one-file rule, but it should no longer hard-code the multilayer perceptron depth as two linear layers.

This phase does not change the prototype branches, fusion equations, losses, scheduler behavior, or validation semantics. It changes only the depth surface of the encoder and heads so that the intended symmetric autoencoder structure is explicit, configurable, and testable.

### File-level edits

[src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/thesis_multitask.py)
- Add an explicit depth parameter shared by:
  - `MultitaskWindowEncoder`
  - `reconstruction_head`
  - `classification_head`
- Keep the implementation self-contained in this file.
- Add a small internal helper for multilayer perceptron construction so the file remains readable and so depth-dependent logic is not duplicated three times in inconsistent ways.

[configs/model/thesis_multitask.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/model/thesis_multitask.yaml)
- Add an explicit configuration field such as `mlp_num_linear_layers`.
- Set the target value to `3` for the intended symmetric model configuration.
- Keep the rest of the model configuration unchanged in this phase.

[src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py)
- Validate the new depth field.
- Ensure the field is an integer greater than or equal to `2`.
- Keep validation explicit and scalar-by-scalar.

### Explicit edit content

The architecture contract should become:

- `mlp_num_linear_layers` is the shared number of linear layers for:
  - the encoder
  - the reconstruction head
  - the classification head

The intended first implemented value is:

- `mlp_num_linear_layers: 3`

The encoder construction rule should be:

- first projection starts from `input_dim`
- final encoder projection ends at `hidden_dim`
- intermediate encoder layers pass through `encoder_dim`
- each internal linear layer except the final output projection is followed by `ReLU`
- dropout remains present only on intermediate hidden transformations, not on the final logits-like output of the head

The reconstruction head construction rule should be:

- first projection starts from `hidden_dim`
- final projection ends at `input_dim`
- intermediate reconstruction layers pass through `encoder_dim`
- internal layers use `ReLU` and dropout

The classification head construction rule should be:

- first projection starts from `hidden_dim`
- final projection ends at `num_classes`
- intermediate classification layers pass through `hidden_dim`
- internal layers use `ReLU` and dropout

The implementation should not introduce three unrelated depth parameters in the first version, because the user requirement is symmetry. The single shared depth field should remain the canonical contract until a future ablation need justifies more granularity.

### Interface and contract definitions

Dataset contract:
- unchanged
- batches remain dictionaries with `x`, optional labels, and metadata

Encoder contract:
- unchanged at the interface level
- `MultitaskWindowEncoder.forward()` must still return:
  - `hidden`
  - `pooled`
  - `aux`

Model contract:
- unchanged at the top level
- `ThesisMultitaskModel.forward()` and stage methods must keep their current outputs
- prototype branches, fusion, optional losses, and validation mechanics remain unchanged

Task contract:
- unchanged
- reconstruction remains sequence reconstruction
- classification remains binary clean-versus-synthetic anomaly classification

Training engine contract:
- unchanged
- the trainer remains agnostic to encoder-head depth and only consumes standardized model outputs

### Design pattern application

Composition over inheritance:
- the encoder and heads remain composed inside the model file rather than split into new subclass hierarchies

Adapter pattern for encoders:
- the encoder still behaves as the thesis-facing representation adapter inside the model file
- no external encoder adapter layer is introduced in this phase

Strategy pattern for tasks:
- unchanged
- task behavior remains configuration-driven through the task YAML and model constructor inputs

Registry or factory:
- unchanged at the runtime level
- the registry still constructs `thesis_multitask` through the existing model factory path

### Risk mitigation steps

Prototype redundancy:
- do not modify continuous or discrete prototype dimensions in this phase
- keep all prototype branch logic unchanged so any observed effect can be attributed to encoder-head depth

Fusion collapse:
- do not change `alpha`, `beta`, or fusion warmup behavior in the same patch

Adaptation contamination:
- do not change [src/models/online_adaptation.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/models/online_adaptation.py) until the offline model change is stable

Projector drift:
- not applicable in this phase because the online projector path is unchanged

Evaluation metric inflation:
- do not claim performance gains from depth change without rerunning the full evaluation path

### Test plan and validation steps

Unit tests:
- extend [tests/test_multitask_shapes.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_multitask_shapes.py) to assert output shapes under `mlp_num_linear_layers=3`
- add config-loading tests for valid and invalid `mlp_num_linear_layers`

Integration tests:
- extend [tests/test_one_multitask_train_step.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_one_multitask_train_step.py) so one forward-backward optimizer step passes with `mlp_num_linear_layers=3`
- ensure checkpoint save-load still works after the architecture change when config matches

Validation steps:
- run the model shape tests
- run one training-step integration test
- run checkpoint round-trip tests

### Acceptance criteria

- The offline thesis model accepts one explicit symmetric depth parameter through YAML.
- Setting `mlp_num_linear_layers: 3` produces a three-linear-layer encoder, a three-linear-layer reconstruction head, and a three-linear-layer classification head.
- The model output contract remains unchanged.
- Existing multitask shape tests, one-step training tests, and checkpoint tests pass.

## Phase 2 - Isolate seed-specific experiment identity, outputs, and logging

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make seed-based experimental comparison reproducible without hidden collisions in outputs, checkpoints, or Weights and Biases logging. The goal is not yet full parallelism. The goal is to make seed-specific runs first-class and isolated so that either sequential or parallel execution becomes technically safe.

### File-level edits

[configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-full__w100__seed7__rtx3090.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-full__w100__seed7__rtx3090.yaml)
- keep this as the canonical base template, not the direct multi-seed execution target
- update stale scheduler comments and tags so they reflect the implemented scheduler state

[configs/experiment/](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/)
- add three seed-specific experiment files, for example:
  - `smd_multitask_rtx3090_seed11.yaml`
  - `smd_multitask_rtx3090_seed23.yaml`
  - `smd_multitask_rtx3090_seed47.yaml`
- each file must define unique:
  - `experiment_name`
  - `seed`
  - `output_dir`
  - `checkpoint_dir`
  - `wandb_run_name`

### Explicit edit content

Each seed-specific config should inherit the same modeling and data settings, but vary only:

- `seed`
- local artifact paths
- W&B run identity

Each seed-specific output path should be unique, for example:

- `outputs/smd_multitask_rtx3090_seed11`
- `outputs/smd_multitask_rtx3090_seed23`
- `outputs/smd_multitask_rtx3090_seed47`

Each seed-specific checkpoint path should stay inside the matching output directory.

The seed-specific configs should all point to the same:

- data config
- model config
- task config

unless later profiling shows that concurrent execution needs a smaller batch size variant.

### Interface and contract definitions

Dataset contract:
- unchanged

Encoder contract:
- unchanged

Model contract:
- unchanged except for Phase 1 depth configurability

Task contract:
- unchanged

Training engine contract:
- unchanged
- one trainer still handles one process and one seed

### Design pattern application

Composition over inheritance:
- seed isolation is handled by configuration composition, not by subclassing experiment types

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- unchanged

Registry or factory:
- unchanged
- experiment-level variation remains configuration-driven rather than hard-coded in scripts

### Risk mitigation steps

Prototype redundancy:
- keep model architecture identical across seeds

Fusion collapse:
- keep all non-seed training settings identical across seeds so fusion behavior is comparable

Adaptation contamination:
- do not mix offline seed runs with online adaptation in this phase

Projector drift:
- not applicable

Evaluation metric inflation:
- ensure each run has unique artifact paths and run names so logs are not merged accidentally

### Test plan and validation steps

Unit tests:
- extend config-loading tests to ensure all three seed-specific configs load successfully

Integration tests:
- verify that each config resolves to a distinct `output_dir`, `checkpoint_dir`, and `wandb_run_name`

Validation steps:
- load each config through the resolved config path
- assert no two seed configs share the same output or checkpoint directory

### Acceptance criteria

- Three seed-specific configs exist and load successfully.
- Each seed-specific config has isolated local artifact paths.
- Each seed-specific config has an isolated W&B run name.
- No path collisions remain between the three runs.

## Phase 3 - Add a controlled local launcher for multi-seed execution

### Phase summary tied to thesis objectives

The thesis objective in this phase is to provide a small, explicit, readable execution surface for running repeated seed experiments without manual command duplication. This phase should support parallel launch as an execution option, but must keep orchestration explicit and lightweight. It should not introduce a hidden job manager or a distributed-training stack.

### File-level edits

[scripts/](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/)
- add a new explicit launcher script, for example:
  - `run_multiseed_experiments.py`

[tests/](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/)
- add tests for launcher argument parsing and config-path enumeration

### Explicit edit content

The launcher should:

1. accept an explicit list of experiment config paths;
2. accept a mode flag such as:
   - `--execution-mode sequential`
   - `--execution-mode parallel`
3. support a bounded maximum concurrent process count such as:
   - `--max-concurrent-processes`
4. print the exact commands it launches;
5. fail fast if two configs share the same output or checkpoint directory.

The launcher should not:

- rewrite model internals
- change seeds itself after config resolution
- infer output directories automatically from partial names
- use distributed PyTorch

The first target behavior should be:

- sequential mode is always supported
- parallel mode is opt-in and explicit

The launcher should call the existing training script rather than duplicating its logic. This keeps the existing runtime graph intact.

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
- unchanged
- the launcher is outside the trainer and only orchestrates multiple independent invocations of the existing entrypoint

### Design pattern application

Composition over inheritance:
- the launcher composes existing experiment entrypoints instead of introducing a new training engine class

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- unchanged

Registry or factory:
- the launcher acts as an execution-level coordinator, not as a replacement for dataset or model registries

### Risk mitigation steps

Prototype redundancy:
- no effect

Fusion collapse:
- keep one process per run so there is no hidden parameter sharing

Adaptation contamination:
- do not include online adaptation configs in the first launcher version

Projector drift:
- not applicable

Evaluation metric inflation:
- require explicit config paths so users know exactly which runs were launched

### Test plan and validation steps

Unit tests:
- launcher rejects duplicate output directories
- launcher rejects duplicate checkpoint directories
- launcher accepts three distinct configs

Integration tests:
- dry-run mode prints three exact commands without executing training
- sequential mode can be tested with a fast smoke config set

Validation steps:
- run dry-run on the three seed-specific configs
- inspect emitted commands and resolved paths

### Acceptance criteria

- A single small script exists to launch multiple seed-specific experiments.
- The launcher supports both sequential and explicit parallel mode.
- Duplicate-path collisions are detected before training starts.
- The launcher reuses the existing training entrypoint rather than reimplementing training logic.

## Phase 4 - Add a preflight memory-safety and runtime-validation surface before enabling three-way parallel full runs

### Phase summary tied to thesis objectives

The thesis objective in this phase is to prevent unsupported concurrency claims from entering the experimental workflow. The current codebase has no evidence that three full batch-size-512 runs can coexist on one RTX3090. Therefore, any parallel execution feature must be gated by explicit preflight checks and a smoke-level validation path before full 300-epoch concurrent execution is allowed.

### File-level edits

[scripts/](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/scripts/)
- add a small preflight script or extend the launcher with:
  - `--preflight-only`
  - `--smoke-first`

[configs/experiment/](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/)
- add seed-specific smoke configs if needed for rapid concurrent validation

### Explicit edit content

Preflight should validate:

1. CUDA availability;
2. distinct output and checkpoint directories;
3. existence of dataset root paths;
4. ability to start one short smoke training run per config;
5. optional collection of per-process peak graphics memory if the environment exposes it cleanly.

The first concurrency gate should be:

- do not mark three-way full parallel execution as the default recommended path until:
  - one single-run smoke test passes for each seed config;
  - one two-process smoke parallel test passes;
  - one three-process smoke parallel test passes.

If those smoke gates fail, the plan should fall back to:

- sequential execution of the three seeds using the same launcher

This phase is about building the repository surface that makes that decision explicit and reproducible. It is not about assuming the hardware can already support the final full parallel workload.

### Interface and contract definitions

Dataset contract:
- unchanged

Encoder contract:
- unchanged

Model contract:
- unchanged except for the Phase 1 symmetric depth field

Task contract:
- unchanged

Training engine contract:
- unchanged
- preflight checks run outside the trainer

### Design pattern application

Composition over inheritance:
- preflight logic is a small orchestration layer on top of the existing launcher and training entrypoint

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- unchanged

Registry or factory:
- unchanged for datasets and models
- launcher and preflight remain execution utilities

### Risk mitigation steps

Prototype redundancy:
- none

Fusion collapse:
- keep model and training behavior identical between smoke and full configs except for runtime scale

Adaptation contamination:
- exclude online adaptation from first multi-seed launcher coverage

Projector drift:
- not applicable

Evaluation metric inflation:
- require users to distinguish:
  - smoke concurrency validation
  - full scientific experiment runs

### Test plan and validation steps

Unit tests:
- preflight rejects missing config files
- preflight rejects missing dataset root
- preflight rejects duplicate artifact paths

Integration tests:
- dry-run plus preflight mode with smoke configs succeeds
- a simulated failure path returns a nonzero exit code and a clear reason

Validation steps:
- run one smoke config sequentially
- run two smoke configs in parallel
- run three smoke configs in parallel
- only then promote the three seed-specific full configs to actual scientific runs

### Acceptance criteria

- The repository contains an explicit preflight surface before full parallel execution.
- Sequential execution remains available as the guaranteed-safe path.
- Parallel execution is gated behind successful smoke validation.
- The codebase does not implicitly claim three-way full parallel support without preflight evidence.

## Phase 5 - Consolidate documentation and experiment usability

### Phase summary tied to thesis objectives

The thesis objective in this phase is to keep the experiment surface obvious to future readers and to preserve reproducibility. The repository should document not only the model change, but also the execution semantics of multi-seed runs.

### File-level edits

[documents/design/idea.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/design/idea.md)
- add a short note that the active offline thesis model now exposes symmetric MLP depth through configuration if Phase 1 is implemented

[documents/design/long_term_codebase_roadmap.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/design/long_term_codebase_roadmap.md)
- add a short note that multi-seed execution is launcher-based and preflight-gated, not assumed

[documents/logs/04-17-2026/research/research-symmetric-three-layer-thesis-model-and-parallel-seed-feasibility.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TỰ%20NHIÊN/Khoá%20luận%20tốt%20nghiệp/bachelor-thesis-2026/documents/logs/04-17-2026/research/research-symmetric-three-layer-thesis-model-and-parallel-seed-feasibility.md)
- if implementation introduces resolved decisions that close its open questions, append a follow-up section there

### Explicit edit content

Documentation should record:

- the new symmetric depth field name and meaning
- the canonical seed-specific config naming pattern
- the existence of a launcher script
- the rule that full parallel execution must follow smoke preflight validation

### Interface and contract definitions

Dataset contract:
- unchanged

Encoder contract:
- unchanged externally

Model contract:
- unchanged externally except for depth configurability

Task contract:
- unchanged

Training engine contract:
- unchanged

### Design pattern application

Composition over inheritance:
- documentation should explain that orchestration remains external to the model and trainer

Adapter pattern for encoders:
- unchanged

Strategy pattern for tasks:
- unchanged

Registry or factory:
- unchanged

### Risk mitigation steps

Prototype redundancy:
- document that the depth change is architectural and not a branch redesign

Fusion collapse:
- document that fusion behavior should be compared at fixed settings across seeds

Adaptation contamination:
- document that the multi-seed launcher is initially for offline multitask experiments only

Projector drift:
- document that online adaptation remains outside this change set

Evaluation metric inflation:
- document that smoke concurrency success is not the same as improved scientific performance

### Test plan and validation steps

Validation steps:
- ensure the detail note, plan note, and any updated design docs do not contradict the implemented runtime

### Acceptance criteria

- The repository documentation matches the implemented architecture and orchestration surface.
- Users can identify the correct config files and launcher path without guessing.
- The multi-seed experiment workflow remains explicit, reproducible, and readable.

## Final implementation order

The implementation should proceed in this exact order:

1. add symmetric depth configurability to the offline thesis model;
2. add config validation and update the base model YAML;
3. update architecture tests and one-step integration tests;
4. add three seed-specific experiment configs with isolated outputs and W&B names;
5. add the multi-seed launcher in sequential mode first;
6. add duplicate-path guards and dry-run mode;
7. add preflight and smoke-gated parallel execution;
8. update design and log documentation after code and tests are stable.

This order preserves the principle that the codebase should first make one thing work clearly, then make repeated experimentation safe, and only then enable higher-throughput execution modes.

---
date: 2026-06-29 23:41:24 +07 +0700
researcher: Codex
git_commit: ad75d65538ac169b6253b757bdeef7a80f3bdfeb
branch: dev
repository: bachelor-thesis-2026
topic: "Structure outline for benchmark runtime simplification and legacy surface removal"
tags: [structure, time-series, anomaly-detection, benchmark, simplification]
status: draft
last_updated: 2026-06-29
last_updated_by: Codex
---

# Structure: Benchmark runtime simplification and legacy surface removal

**Date**: 2026-06-29 23:41:24 +07 +0700  
**Researcher**: Codex  
**Git Commit**: `ad75d65538ac169b6253b757bdeef7a80f3bdfeb`  
**Branch**: `dev`

## Overview

This structure outline implements the simplification plan documented in [plan-benchmark-runtime-simplification.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/documents/logs/06-29-2026/plan/plan-benchmark-runtime-simplification.md). The goal is to reduce the active benchmark runtime to one clear path, namely `train -> val -> val_synth -> test`, while removing legacy validation and orchestration surfaces that no longer serve the main benchmark.

No new framework layer will be introduced. The work will preserve the current batch contract, encoder contract, model output contract, and dataset builder architecture. The simplification will proceed by deleting obsolete branches, deleting obsolete config keys, renaming active orchestration surfaces, and strengthening tests around the remaining active path.

## Locked Decisions

The following low-level decisions are now fixed for the implementation:

1. The old `comparative` active surface may be removed aggressively because the historical state is already preserved on GitHub.
2. Active launcher files may be renamed if this improves clarity, provided the full pipeline is re-verified through tests and smoke runs.
3. The legacy balancing aliases `balance_binary_classes_within_batch` and `balance_classes_within_batch` should be removed immediately if no active configs depend on them.
4. The active RedLamp baseline naming surface should move toward an encoder-agnostic name because the implementation already supports multiple `encoder_family` values.

Current repository evidence supports decisions 3 and 4. A repository scan shows that the balancing aliases remain only in code, tests, and historical documents, not in active benchmark configs. The same scan also shows that `RedLampMLPBaseline` is still the public class and registry surface even though the active runtime already supports more than one encoder family.

## Implementation Phases

1. **Phase 1: Active Contract Lock and Test Gate**
   This phase establishes the minimal vertical slice for the simplification work. The codebase will first lock the active benchmark contract in tests before deleting runtime branches. The preserved contracts are:
   - batch contract: `x`, `point_labels`, `mask`, `timestamps`, `meta`
   - encoder contract: thesis-facing hidden representation with shape `[B, L, H]`
   - output contract: reconstruction, logits, point scores, window scores, and auxiliary fields
   The main engineering principle in this phase is stable interfaces before deletion. The main design principle is composition over inheritance: the trainer, evaluator, and dataset builders remain separate, while only stage semantics are simplified.

2. **Phase 2: Remove `val_realistic` from the Active Runtime**
   This phase removes `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override` from the active config schema, trainer logic, scheduler monitor logic, checkpoint monitor logic, and offline model stage APIs. After this phase, the only auxiliary validation namespace in the active runtime will be `val_synth`. The main engineering principle is single responsibility: synthetic validation is handled through one path only. The main design principle is a thinner active runtime interface with fewer state branches.

3. **Phase 3: Remove Legacy Balancing Aliases and Tighten Synthetic Validation Semantics**
   This phase removes `balance_binary_classes_within_batch` and `balance_classes_within_batch` from active model constructors, injector wiring, and active tests. The canonical field will remain `train_balance_classes` for now, but its semantics will be documented clearly as governing both synthetic training and synthetic validation balancing. The main engineering principle is explicit configuration semantics. The main design principle is one canonical parameter per behavior.

4. **Phase 4: Migrate Surviving Config Families to the Simplified Runtime**
   This phase makes the remaining config tree consistent with the cleaned runtime. The main objective is to avoid a half-clean repository in which the active runtime is simplified but many surviving YAMLs still point at removed namespaces. The main engineering principle is mechanical, auditable migration. The main design principle is that active config surfaces should describe only active runtime behavior.

5. **Phase 5: Rename the Active Baseline to an Encoder-Agnostic Public Name**
   This phase removes the misleading `MLP` qualifier from the active baseline public surface. The implementation should move the active naming surface toward `RedLampBaseline` and `redlamp_baseline` while keeping `encoder_family` as the mechanism that selects MLP, 1D-CNN, or later encoder variants. It includes model file names, class names, registry names, model config names, active experiment identifiers, and active baseline test names. The main engineering principle is truthful public naming. The main design principle is that one baseline family should be chosen by one model name plus encoder configuration, not by separate misleading public names.

6. **Phase 6: Rename Active Orchestration from `comparative` to `benchmark`**
   This phase converts the active orchestration layer from `comparative` vocabulary to `benchmark` vocabulary. It includes the launcher, preflight script, runner script, report names, manifest names, status strings, and active test names. If file renaming is performed, it will be accompanied by targeted test updates and smoke verification. The main engineering principle is naming consistency across runtime and artifacts. The main design principle is that the public active entrypoints should describe the actual experiment family they launch.

7. **Phase 7: Remove Residual Noise and Historical Ambiguity**
   This phase cleans the remaining low-value noise that can obscure benchmark execution. It includes reducing hot-path batch logging in `src/data/collate.py`, updating CLI help text, and rewriting active design or guideline notes that still present `val_realistic` or `comparative` as the primary path. Historical documents may remain, but they should no longer define active behavior. The main engineering principle is readability-first maintenance. The main design principle is a clear boundary between active runtime and historical reference material.

8. **Phase 8: Full Verification and Smoke Rehearsal**
   This phase verifies that simplification did not damage the benchmark pipeline. It includes:
   - focused pytest suites for config loading, validation alignment, scheduler behavior, and launcher behavior
   - at least one baseline train-step smoke
   - at least one thesis multitask train-step smoke
   - active benchmark launcher dry-run
   - active benchmark preflight validation
   The main engineering principle is evidence before claiming success. The main design principle is that a smaller runtime must still be proven operational end-to-end.

## Phase Boundaries and Dependencies

- Phase 1 must complete before any destructive deletion of stage branches.
- Phase 2 must complete before launcher and report naming cleanup, because the active runtime vocabulary must be stable first.
- Phase 3 may proceed in parallel with the later half of Phase 2 if the tests already protect the active constructor contract.
- Phase 4 should begin only after the active runtime semantics are already clean.
- Phase 5 should begin only after the active config tree is already reduced to surviving families.
- Phase 6 should begin only after the canonical baseline name is stable inside the active benchmark configs.
- Phase 7 should not change behavior, only clarity and noise level.
- Phase 8 is mandatory before any server launch commands are prepared.

## Minimal Vertical Slice

The first milestone that counts as a valid implementation slice is:

1. Benchmark configs load and validate without any `val_realistic` fields.
2. Trainer executes `train -> val -> val_synth`.
3. `RedLampMLPBaseline` supports the cleaned validation surface.
4. `ThesisMultitaskModel` supports the cleaned validation surface.
5. Active benchmark tests pass.

Only after this slice is green should the implementation continue into config migration, public naming cleanup, and broader document cleanup.

## Files Most Likely to Change in Each Phase

- **Phase 1**
  - `tests/test_config_loading.py`
  - `tests/test_multitask_validation_alignment.py`
  - `tests/test_learning_rate_scheduler.py`
  - `tests/test_redlamp_mlp_baseline.py`
  - `tests/test_thesis_multitask_config_refactor.py`

- **Phase 2**
  - `src/core/config.py`
  - `src/engine/trainer.py`
  - `scripts/train.py`
  - `src/models/redlamp_mlp_baseline.py`
  - `src/models/thesis_multitask.py`

- **Phase 3**
  - `src/data/augment.py`
  - `src/models/redlamp_mlp_baseline.py`
  - `src/models/thesis_multitask.py`
  - active tests that still instantiate models through legacy aliases

- **Phase 4**
  - `configs/task/multitask_tsad.yaml`
  - `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`
  - `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml`
  - `configs/task/multitask_tsad_redlamp_multiclass_window20_redlamp_aligned.yaml`
  - `configs/task/multitask_tsad_window10_binary.yaml`
  - `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`
  - surviving active experiment YAMLs

- **Phase 5**
  - `src/models/redlamp_mlp_baseline.py`
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/run_online_adaptation.py`
  - `configs/model/redlamp_mlp_baseline.yaml`
  - `configs/model/redlamp_mlp_baseline_comparative_smd.yaml`
  - `configs/model/redlamp_mlp_baseline_redlamp_aligned.yaml`
  - active benchmark experiment YAMLs that still reference `redlamp_mlp_baseline`
  - active baseline model tests

- **Phase 6**
  - `scripts/launch_tmux_comparative_smd_experiment.sh`
  - `scripts/run_comparative_smd_experiments.py`
  - `scripts/preflight_comparative_smd_server.py`
  - related orchestration tests

- **Phase 7**
  - `src/data/collate.py`
  - `src/core/config_help.py`
  - active guideline notes under `documents/design/`

- **Phase 8**
  - no core code redesign is expected
  - this phase mainly executes tests and smoke commands and only fixes findings

## Risks That Must Be Watched During Structure Execution

1. A hidden dependency may still expect the `val_realistic` metric namespace.
2. A historical test may fail because it was unintentionally acting as an active-path regression test.
3. Surviving active configs may still point at removed `val_realistic_*` fields unless the config migration is fully mechanical.
4. Renaming the baseline public name may break imports, registry lookups, or config loading if class names, file names, and model names are not updated atomically.
5. Renaming launcher files may break test discovery or shell references if not updated atomically.
6. Removing alias parameters may expose old constructor call sites outside benchmark tests.

These risks are acceptable if the implementation proceeds in the phase order above and runs verification after each phase boundary.

## Feedback Check

This phasing assumes the repository should prioritize maximum simplification of the active benchmark path, even if that means removing broad historical surfaces from the working tree and rewriting several tests immediately.

The key structural question is:

- Should the implementation keep baseline renaming and orchestration renaming in one phase, or split them after the config migration?

My recommendation is to split them. The baseline rename touches imports, registry names, model YAMLs, and many active experiment identifiers. The orchestration rename touches launch tooling and shell-facing artifact names. Keeping them as two separate later phases makes failures easier to localize while still keeping both tasks inside the public-naming cleanup part of the plan.

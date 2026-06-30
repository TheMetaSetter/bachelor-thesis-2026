---
date: 2026-06-30 14:05:00 +0700
planner: Codex
git_commit: ddd20afb2f45c83a17fa93d54624789b783ca29d
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation structure for aggressive runtime cleanup before benchmark execution"
tags: [structure, cleanup, runtime, configs, benchmark, readability]
status: draft
last_updated: 2026-06-30
last_updated_by: Codex
source_plan: documents/logs/06-30-2026/plan/plan-simplify-active-runtime-surface-and-remove-misleading-legacy-paths.md
source_research: documents/logs/06-30-2026/research/research-code-paths-not-yet-simplified-and-easy-to-misunderstand.md
---

# Structure: Simplify the active runtime surface and remove misleading legacy paths

## Overview

The cleanup should now be treated as a controlled hardening pass rather than a cosmetic rename pass. The repository already has the correct benchmark-critical core in many places, but the public runtime surface still exposes too many transitional names, partially legacy validation semantics, and test/config paths that teach the wrong mental model.

The safest aggressive strategy is not a single large rewrite. The safest aggressive strategy is a staged cleanup with explicit freeze points. Each phase must end with config-load checks, targeted pytest coverage, and at least one smoke command when the touched area can affect runtime behavior.

## Implementation Phases

1. **Freeze the canonical public surface**
   This phase establishes one official naming and semantics surface for active benchmark work. `redlamp_baseline` becomes the only canonical baseline model identity, while compatibility aliases are pushed into a clearly temporary boundary. Public help text, launchers, experiment config internals, and active runtime examples must all teach the same name. This phase preserves the minimal vertical slice principle because it does not redesign models, loaders, or metrics. It only removes public ambiguity.

2. **Collapse legacy baseline naming and file-path drift**
   This phase aligns physical file paths, preset names, experiment YAML filenames, and test imports with the canonical baseline identity. The purpose is to stop the current situation where the runtime says one thing while filenames and tests still say another. Software engineering discipline is preserved here by changing one identity family at a time: model config files, experiment config filenames, launcher references, then compatibility tests. If a temporary shim remains, it must be visibly compatibility-only and covered by a very small dedicated test set.

3. **Isolate or remove misleading validation semantics from the benchmark surface**
   This phase addresses the deeper confusion source: `val_realistic`, `val_realistic_source`, and `test_smd_all` currently leak SMD-specific ideas into generic trainer and config layers. The structure should either remove these semantics from the active benchmark path or isolate them so they no longer appear as generic first-class concepts. The design principle here is separation of concerns: dataset-specific prior estimation should not shape the generic trainer contract if the benchmark contract no longer depends on it.

4. **Clean the active config and task surface to match the benchmark contract**
   This phase rewrites the active task/model/experiment config surface so that a new reader can infer the true benchmark protocol directly from config files. Active benchmark configs should expose only the settings that are truly used for the final benchmark campaign. Legacy or exploratory configs may remain, but they must either move behind compatibility naming or be clearly marked non-comparable, deprecated, or exploratory. This keeps the configuration system simple without inventing a new abstraction layer.

5. **Reduce test-suite mental-model duplication**
   This phase updates tests so they distinguish clearly between canonical behavior and legacy compatibility behavior. Canonical runtime tests should import canonical files, use canonical model names, and assert the benchmark-safe semantics now intended for real runs. Only a small number of explicit compatibility tests should preserve old aliases. This preserves practical maintainability because future regressions will point to the real public contract instead of to migration leftovers.

6. **Run full verification on the cleaned benchmark path**
   This phase is the operational gate before benchmark execution. It should include config loading for all active benchmark configs, focused pytest bundles for renamed surfaces and validation semantics, and smoke runs for at least one baseline and one thesis benchmark-like path. This preserves reproducibility and practical safety because the cleanup is only considered complete once the exact runtime path intended for benchmark execution has been revalidated end to end.

## Execution Order Rationale

The order above is intentionally not “rename everything first.” If the repository renames files before the canonical runtime contract is frozen, the cleanup can spread confusion faster. The correct order is:

1. freeze the public meaning,
2. align names and paths to that meaning,
3. remove or isolate stale semantics,
4. simplify configs,
5. simplify tests,
6. verify the whole benchmark path.

This ordering keeps the repository readable at every intermediate checkpoint and lowers the risk of benchmark-blocking breakage.

## Main Code Areas

The most sensitive files for this cleanup are:

- `src/core/config.py`
- `src/core/config_help.py`
- `src/engine/trainer.py`
- `src/models/thesis_multitask.py`
- `src/models/redlamp_baseline.py`
- `src/models/redlamp_mlp_baseline.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_online_adaptation.py`
- `scripts/run_comparative_smd_experiments.py`
- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `configs/model/*.yaml`
- active `configs/experiment/**/*.yaml`
- tests still importing `RedLampMLPBaseline` or asserting legacy names

The highest semantic-risk zone is not the baseline model implementation itself. The highest semantic-risk zone is the combined config plus trainer plus task surface around validation and model identity.

## Risk-Control Checkpoints

After Phase 1:
- config-help examples and runtime registration surface must agree on one canonical baseline name.

After Phase 2:
- renamed config paths, renamed model config references, and launcher paths must all load successfully.

After Phase 3:
- generic trainer code must no longer look like it secretly assumes SMD as the default benchmark family.

After Phase 4:
- active benchmark configs must read like the benchmark protocol that will actually be reported.

After Phase 5:
- canonical tests and compatibility tests must be clearly separated.

After Phase 6:
- the benchmark launch commands intended for SMD `1-6`, `3-4`, `3-9`, and `SWaT` preparation must be rechecked from the cleaned surface.

## Recommendation

The structure should proceed as an aggressive cleanup with staged freeze points, not as a conservative partial cleanup. Given the current repository state, the main time loss risk is no longer code-writing speed. The main time loss risk is benchmark confusion caused by half-cleaned semantics. A well-bounded hard cleanup is therefore justified, as long as each phase ends with explicit verification before the next phase begins.

---
date: 2026-09-09 21:35:18 +07:00
researcher: OpenAI Codex
topic: "Detect code modifications required by the terminology ontology naming conventions"
status: complete
revision: ed10fbb4db70f9ed17f98330c18f6e456ba5f2f5
branch: dev
---

# Research: Detect code modifications required by the terminology ontology naming conventions

## Summary

The active smoke configuration surface does not implement the selected W&B display-name convention.

The repository currently has 508 smoke configuration files and none of their `wandb_run_name` values starts with `smk-`.

The code also overwrites configured names at runtime, so changing generated YAML alone would not be sufficient.

The ontology requires canonical phase and variant identity to remain available in configuration, metadata, tags, or provenance even when the W&B display name is short.

## Research question

Which source-code and configuration-generation lines must change so smoke W&B run names comply with `smk-<phase>-<method_or_variant>-<entity>-s<seed>` while preserving canonical ontology identity?

## System context

`documents/spec/offline_pretraining_terminology_ontology.md` defines `offline_pretraining_phase`, `offline_variant`, `stage_a_multitask_pretraining`, `stage_b_memory_initialization`, `stage_b_fusion_finetuning`, `stage_b_best_checkpoint`, and `threshold_artifact`.

`documents/spec/online_tta_terminology_ontology.md` defines `online_tta_phase`, `online_variant`, `causal_window`, `verification_buffer`, `verification_entry`, and the inherited `stage_b_best_checkpoint` and `threshold_artifact` lineage.

`documents/notes/terminology_ontology_naming_conventions.md:343-369` records the selected smoke display convention and the examples `smk-off-O0-e1_6-s8`, `smk-on-KA-A2-e3_9-s36`, and `smk-off-KA-e3_4-s6`.

The note records `off` and `on` as display tokens, `O0` as an `offline_variant` value, `A2` as an `online_variant` value, and `KA` as a display token for `kmeans_ad`.

The note explicitly says the full stage, checkpoint, window, protocol, and artifact identity must remain in configuration, tags, metadata, or provenance.

## Execution path

1. Configuration generators build experiment dictionaries and write YAML files.
2. `load_experiment_config` resolves the three referenced sections and merges override sections.
3. Benchmark and experiment runners copy or replace the logging configuration before constructing `ExperimentLogger`.
4. `ExperimentLogger` passes `logging_config["wandb_run_name"]` to `wandb.init`.
5. If no explicit name exists, `src/engine/logger.py:111-116` falls back to the long `experiment_name`.

## Detailed findings

### 1. The generated smoke names violate the selected format

`scripts/benchmarks/generate_smd_benchmark_configs.py:81-103` creates THESIS offline logging and appends `-smoke` or `-main` to an `off-...` name.

The smoke branch therefore produces names such as `off-O0-machine_1_6-s8-smoke` instead of `smk-off-O0-machine_1_6-s8`.

`scripts/benchmarks/generate_benchmark_smoke_configs.py:68-78` repeats the same legacy `off-...-smoke` construction for the benchmark-smoke directory.

`scripts/benchmarks/generate_online_benchmark_configs.py:219-237` produces `on-<offline_variant>-<online_variant>-<entity>-s<seed>-smoke` and therefore lacks the required `smk-` prefix and has an extra mode suffix.

`scripts/benchmarks/generate_online_streaming_benchmark_configs.py:221-235` produces the same legacy `on-...-smoke` shape for online baseline configs.

`scripts/benchmarks/generate_offline_benchmark_configs.py:111-125` uses the long `benchmark_name` as the W&B name for smoke baseline configs.

### 2. A matrix generator hides the smoke naming decision inside `run_id`

`scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-202` builds `run_id` as `<phase>-<method><variant>-<entity>-s<seed>` and passes it directly to `_common_logging` at lines 325-336.

This path makes W&B naming depend on a generic run identifier and does not distinguish the smoke display label from canonical experiment identity.

The same generator creates THESIS, RedLamp, offline-baseline, THESIS-online, and online-baseline configurations, so one shared naming helper must preserve their different identity fields.

### 3. Runtime callers overwrite configured W&B names

`scripts/cli/train.py:264-269` always replaces the configured name with `build_wandb_run_name` when W&B is enabled.

`scripts/cli/evaluate.py:306-312` does the same for evaluation.

`scripts/experiments/run_online_adaptation.py:140-146` does the same for online adaptation.

`scripts/benchmarks/run_thesis_online_benchmark.py:310-319` does the same for THESIS online benchmarks and passes the CLI `online_variant` separately.

`scripts/experiments/run_two_stage_offline_pretraining.py:157-162` creates per-stage names through the generic helper, which can remove the selected smoke display format from generated stage configurations.

`scripts/benchmarks/run_online_streaming_benchmark.py:328-336` creates a fallback name from `baseline_name`, `online_variant`, and seed without the smoke convention.

`scripts/run_direct_branch_routing_smoke.py:62-71` explicitly sets `off-O0-machine_1_6-s6-direct-smoke`, which is not the selected convention.

### 4. The generic W&B run-name helper uses a different contract

`src/core/artifact_naming.py:204-216` implements `build_wandb_run_name` by calling `build_wandb_artifact_name` with role `run`.

`build_wandb_artifact_name` at lines 189-201 produces names such as `run-online-A2-O0-...`, which are artifact-style names and not the selected smoke display format.

The function is used by several runtime callers, so it is a runtime naming surface rather than dead code.

### 5. Canonical identity is missing or inferred from the wrong field

`scripts/benchmarks/generate_smd_benchmark_configs.py:127-140` writes `experiment_variant` but does not write root `offline_variant`.

`documents/spec/offline_pretraining_terminology_ontology.md:471-473` records this as a known conflict because artifact collection can fall back to `O0` when the root identity is absent.

`scripts/benchmarks/generate_online_benchmark_configs.py:168-207` places `offline_variant` in `task_overrides` but does not write root `offline_variant` or root `online_variant`.

`src/core/config.py:245-263` does not allow root `offline_variant` or `online_variant`, so adding those canonical fields requires a deliberate schema change and tests.

`src/core/artifact_naming.py:113-122` falls back from missing offline identity to `experiment_variant`, although the ontology says `experiment_variant` describes protocol detail and does not replace `offline_variant`.

`src/core/artifact_naming.py:105-122` also parses `main` as a possible online variant, while the online ontology defines the canonical `online_variant` domain as `A0`, `A1`, and `A2`.

The resolved-config probe confirmed that the representative THESIS offline config has no root or task `offline_variant`, and the representative THESIS online config has `offline_variant` only in `task` and no `online_variant` field.

### 6. Existing runtime compatibility names are not all violations

The online ontology intentionally keeps scalar `raw_point_score`, `ewma_point_score`, and `prediction` fields as compatibility fields.

The ontology therefore does not authorize replacing every occurrence of those names.

`src/engine/online_tta/online_engine_window_metrics.py:92-138` does contain the legacy state parameter `previous_ewma_point_scores`, which the ontology maps to `active_ewma_point_scores` because state is keyed by absolute index.

That state rename is a separate runtime-contract change and should not be mixed into the W&B smoke-name change without an explicit scope decision.

`verification_buffer` and `verification_entry` are distinct objects, so their separate names must remain.

## Evidence

- `documents/spec/offline_pretraining_terminology_ontology.md:80-85` — defines the canonical offline phase, variant, stages, and evaluation operation.
- `documents/spec/offline_pretraining_terminology_ontology.md:109` — states that `experiment_variant` does not replace `offline_variant` for cross-phase identity.
- `documents/spec/offline_pretraining_terminology_ontology.md:371-381` — defines `stage_b_best_checkpoint` and `threshold_artifact` identity requirements.
- `documents/spec/offline_pretraining_terminology_ontology.md:471-473` — records the missing-root-`offline_variant` conflict.
- `documents/spec/online_tta_terminology_ontology.md:58-64` — defines the online phase, variants, causal window, and inherited artifact.
- `documents/spec/online_tta_terminology_ontology.md:88-90` — states that combined labels are not new `online_variant` values.
- `documents/spec/online_tta_terminology_ontology.md:207-226` — distinguishes `online_point_ewma_threshold` from offline and window thresholds.
- `documents/spec/online_tta_terminology_ontology.md:193-205` — preserves scalar endpoint fields as compatibility fields.
- `documents/notes/terminology_ontology_naming_conventions.md:53-63` — requires canonical snake-case identifiers and allows short display labels only when canonical identity remains available.
- `documents/notes/terminology_ontology_naming_conventions.md:323-341` — defines the procedure for name-related prompts and W&B display names.
- `src/core/artifact_naming.py:73-163` — extracts the current artifact identity and silently falls back to generic or inferred variant values.
- `src/core/artifact_naming.py:180-216` — constructs artifact names and the generic W&B run name.
- `src/engine/logger.py:94-118` — initializes W&B and falls back to `experiment_name` when no explicit run name exists.
- `scripts/benchmarks/generate_smd_benchmark_configs.py:81-103` — generates THESIS offline W&B names.
- `scripts/benchmarks/generate_benchmark_smoke_configs.py:47-80` — generates benchmark-smoke THESIS offline W&B names.
- `scripts/benchmarks/generate_online_benchmark_configs.py:160-240` — generates THESIS online W&B names and task overrides.
- `scripts/benchmarks/generate_offline_benchmark_configs.py:91-126` — generates traditional offline baseline W&B names.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:189-237` — generates online baseline W&B names.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-202` — creates the shared matrix run identity.
- `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:325-336` — copies a matrix run identifier into W&B logging.
- `scripts/cli/train.py:264-269` — overwrites configured names for training.
- `scripts/cli/evaluate.py:306-312` — overwrites configured names for evaluation.
- `scripts/experiments/run_online_adaptation.py:140-152` — overwrites configured names for online adaptation.
- `scripts/benchmarks/run_thesis_online_benchmark.py:310-324` — overwrites configured THESIS online names.
- `scripts/benchmarks/run_online_streaming_benchmark.py:328-341` — creates a fallback online baseline name.
- `scripts/run_direct_branch_routing_smoke.py:62-71` — hard-codes a non-compliant smoke name.
- `tests/benchmarks/test_smoke_wandb_logging.py:8-27` — verifies W&B is enabled for smoke configs but does not verify the selected name format.
- `tests/core/test_artifact_naming.py:21-76` — verifies the older artifact-style naming helper contract.

## Configuration observed

| Setting or surface | Current behavior | Evidence | Scope |
| --- | --- | --- | --- |
| `use_wandb` | Enabled for the inspected smoke configs | `tests/benchmarks/test_smoke_wandb_logging.py:20-27` | Smoke configuration inventory |
| `wandb_mode` | `online` for the inspected smoke configs | `tests/benchmarks/test_smoke_wandb_logging.py:25-27` | Smoke configuration inventory |
| `wandb_run_name` | 0 of 508 smoke configs starts with `smk-` | Read-only YAML inventory on 2026-09-09 | `configs/experiment` and `scripts/configs/experiment` |
| `offline_variant` | Missing from representative resolved offline config | `.venv/bin/python` config probe | THESIS offline identity |
| `online_variant` | Missing from representative resolved online config | `.venv/bin/python` config probe | THESIS online identity |

## Tests and validation performed

The focused existing test command passed 8 tests.

The command was `.venv/bin/python -m pytest -q tests/core/test_artifact_naming.py tests/benchmarks/test_smoke_wandb_logging.py tests/benchmarks/test_offline_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py`.

The passing tests establish the current artifact helper, W&B smoke enablement, and selected generator behavior.

They do not establish compliance with the selected `smk-` format because no assertion checks that format.

## Conflicts and uncertainties

The user explicitly selected `KA` for `kmeans_ad`, but the available convention does not define display tokens for every baseline method.

The safest implementation assumption is to preserve existing method tokens for methods without an explicitly selected short token and use `KA` only for `kmeans_ad`.

The requested smoke format does not include stage, window, protocol, or artifact tokens, so those values must remain in metadata, tags, job type, resolved config, or provenance.

The legacy `previous_ewma_point_scores` state name is an ontology mismatch, but changing it would affect online runtime state and tests beyond the directly requested W&B smoke naming surface.

## Open questions

The remaining method-token mapping should be confirmed before implementation if anh wants abbreviations for `candi`, `m2n2`, `stumpy`, or `iforest` instead of their existing method tokens.


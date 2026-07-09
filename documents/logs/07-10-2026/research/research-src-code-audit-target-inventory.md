---
date: 2026-07-10T03:18:01+0700
researcher: Codex
git_commit: 6dc99c0dd296e96bb28f563d38e00d13a0da94f8
branch: dev
repository: bachelor-thesis-2026
topic: "Inventory of src/ code requiring a codebase-preferences audit"
tags: [research, source-audit, readability, contracts]
status: complete
last_updated: 2026-07-10
last_updated_by: Codex
---

# Research: `src/` code-audit target inventory

## Research Question

Which current `src/` code paths require an audit before a preliminary
programming plan can enforce `codebase_preferences.md` without changing thesis
runtime contracts?

## Summary

The audit must target 13 source files above the 500-line limit and 69
functions or methods above the 50-line limit. The highest-risk targets are not
simply the largest files: they own config resolution, training, synthetic
labels, model phase state, online adaptation, and pointwise evaluation.

The current worktree also contains an uncommitted shared neural-block extraction
and pytest discovery configuration. The audit must review those changes as part
of the model dependency boundary, rather than assuming the base commit alone is
the current state.

## Detailed Findings

### Tier 1: runtime contract owners

| Target | Evidence | Why it must be audited first |
| --- | --- | --- |
| `src/core/config.py` | 942 lines; `_validate_optimizer_config` 199 lines; `_validate_logging_config` 179 lines; `load_experiment_config` 85 lines | Resolves YAML into the runtime contract and compatibility aliases. |
| `src/core/config_model_validation.py` | 691 lines; `_validate_model_and_task_config` 435 lines; `_validate_model_and_task_semantics` 192 lines | Owns model/task validation and must preserve exact error behavior. |
| `src/engine/trainer.py` | 921 lines; `Trainer.train` 388 lines | Owns optimizer stepping, validation scheduling, checkpoint selection, and experiment logging. |
| `src/data/augment.py` | 972 lines; injector constructor 92 lines; `augment_batch` 89 lines | Owns seeded synthetic anomaly generation, labels, masks, and class balance. |
| `src/engine/online_tta/online_engine.py` | 856 lines; online execution entrypoint at line 826 | Owns online calibration, buffering, updates, and report finalization. |

### Tier 2: thesis and baseline model boundaries

| Target | Evidence | Audit concern |
| --- | --- | --- |
| `src/models/thesis_multitask.py` and `thesis_multitask_*_mixin.py` | Five mixins remain 624-950 lines; `forward` is 185 lines and `_shared_step` is 148 lines | Public `ThesisMultitaskModel` is split across lifecycle mixins, conflicting with the updated public-entrypoint rule. |
| `src/models/thesis_multitask_components.py` | 528 lines; `ThesisMultitaskModelConfig.from_flat_kwargs` is 181 lines | Config assembly and encoder components must be separated without changing flat-YAML constructor compatibility. |
| `src/models/redlamp_baseline.py` | 640 lines; constructor 192 lines | Must preserve baseline output and synthetic-label contracts while removing model-to-model coupling. |
| `src/models/online_adaptation.py` | 528 lines; constructor 87 lines; `forward` 65 lines | Must preserve frozen-reference, projector, and online checkpoint semantics. |

### Tier 3: data, evaluation, and metric boundaries

| Target | Evidence | Audit concern |
| --- | --- | --- |
| `src/engine/evaluator.py` | `Evaluator.evaluate` 127 lines; payload accumulation 61 lines | Preserve overlap reconstruction and pointwise record fields. |
| `src/metrics/pointwise.py` | 617 lines; `compute_pointwise_metrics` 66 lines | Preserve metric names, threshold semantics, and VUS/range behavior. |
| `src/data/datasets/smd.py` | `SMDDatasetParser.parse` 121 lines | Preserve normalize-before-windowing and label extraction. |
| `src/data/window.py` | `slice_sequence_into_windows` 67 lines | Preserve half-open slice and overlap semantics. |
| `src/engine/online_loop.py` | `OnlineLoop.run` 163 lines | Keep stream/update sequencing distinct from online-TTA experiment orchestration. |

### Tier 4: remaining size-limit violations

Audit after the runtime owners above: `src/models/thesis_multitask_state_mixin.py`,
`src/models/thesis_multitask_routing_mixin.py`,
`src/models/thesis_multitask_loss_mixin.py`,
`src/models/thesis_multitask_setup_mixin.py`,
`src/analysis/evaluation_protocol_audit.py`, `src/engine/logger.py`,
`src/data/stream.py`, `src/data/datasets/anomaly_archive.py`,
`src/metrics/affiliation.py`, `src/engine/checkpoint.py`, and the online/traditional
baseline methods listed by the AST scan.

## Existing Contracts to Preserve

- Batch data remains `batch["x"]` with shape `[B, L, D]`; the active thesis
  design documents specify `L = 20`.
- The thesis-facing model output uses `hidden`, optional `pooled`, `recon`,
  `logits`, point/window scores, and `aux`.
- `ThesisMultitaskModel` remains the registry-facing public class; its phase,
  memory, and checkpoint behavior is covered by dedicated tests.
- The selected test root is now `tests/` through `pytest.ini`; reference
  codebase tests are intentionally excluded.

## Code References

- `src/core/config.py:858` — experiment-config resolution.
- `src/core/config_model_validation.py:6` — model/task validation surface.
- `src/data/augment.py:38` and `:884` — synthetic augmentation lifecycle.
- `src/engine/trainer.py:534` — offline training orchestration.
- `src/engine/evaluator.py:247` — evaluation owner.
- `src/engine/online_tta/online_engine.py:826` — online-TTA entrypoint.
- `src/models/thesis_multitask.py:37` — public thesis model entrypoint.
- `src/models/online_adaptation.py:141` — online model entrypoint.
- `src/metrics/pointwise.py:542` — pointwise metric assembly.

## Open Questions

None block a preliminary audit plan. The implementation plan must, however,
lock public import, state-dict, YAML, and metric-output compatibility before
moving any code.

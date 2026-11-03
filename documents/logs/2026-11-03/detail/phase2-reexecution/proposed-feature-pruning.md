# Phase 2 rerun: feature-pruning proposal

Date: 2026-11-03  
Status: proposal for review; no runtime code or configs changed.

This is the current feature-pruning proposal. The earlier tree drafts were removed after consolidation into `proposed-next-version-tree.md`. This proposal retires an unsupported configuration feature; it does not move source files between folders.

## User-selected runtime priorities

The user selected these responsibilities for the target design:

- Keep THESIS/MUSTARD and selected baseline methods runnable. The baseline method list is not yet named.
- Keep offline training and evaluation, online test-time adaptation with evaluation, and all THESIS O0/O1/O2 and A0/A1/A2 ablations.
- Keep configuration generation, experiment launchers, checkpoint/calibration/resume handling, and report collection/summary.

Analysis and visualization tools were not selected as required runtime responsibilities. This does not by itself authorize removing a particular tool or its outputs. The next scope decision is to name the baseline methods that must remain runnable; until then, baseline-specific retirement remains unresolved.

## Preserve

- THESIS implements the MUSTARD method. Keep its `O0/O1/O2` and `A0/A1/A2` ablations.
- Keep every required comparison method and its own runnable profile.
- Keep method math, data and scoring contracts, calibration, checkpoint selection, online update order, resume behavior, tests, and research assets.

## Retire: non-THESIS A-label baseline profiles

Remove the legacy `A0/A1/A2` online config profiles for STUMPY, KMeansAD, Isolation Forest, CANDI, and M2N2. Phase 1 counted 270 such YAML files: 54 per method. These profiles are outside the official experiment matrix. The current profiles are `main` for the traditional baselines and `reference_adapter_redlamp_encoder` for CANDI/M2N2.

Also reject those legacy A-label values when a baseline run is configured. This prevents recreating the misleading profiles. Keep THESIS A-label handling unchanged. For baselines, use the method's current profile. Do not rewrite old result identities; historical artifacts keep their original labels.

The source, script, and test scan found no references to those specific config filenames. Older research logs do reference them, so retain those logs as historical records. Existing canonical `main` and `reference_adapter_redlamp_encoder` configs remain.

## Why this is the first cut

It removes repeated baseline configurations that present THESIS-only ablation labels as baseline variants. It preserves all baseline algorithms and their official method profiles. This is a config and accepted-input reduction; it does not by itself achieve the former core-code reduction target. Count source changes separately with the established code-only counter.

## Core file-count correction

The current worktree has **244 counted source files, 42,297 code lines, and 1,706,623 code bytes**. Retiring the 270 YAML configs does not lower that source-file count. The file-level tree below is therefore a current inventory, not a reduced target; it must not be presented as satisfying the file-reduction goal.

The target is now **at most 40 counted source files, with no file over 100 lines and no more than 4,000 total code lines**. This is at least 83% fewer files and 90.5% fewer lines than the current worktree. The target requires removing at least 38,297 of the current 42,297 code lines, not only consolidating files.

The raw THESIS implementation count is 4,554 lines across 20 internal modules, before the public model entrypoint. This is not a lower bound on the mathematical implementation: the first source review found duplicate calculations and forwarding code. The earlier conclusion that the target must retire scientific behavior was premature. A behavior-preserving reduction must be measured before changing the Phase 1 boundary.

The overall 4,000-line target still requires reducing the 42,297-line source baseline by 38,297 lines. THESIS cleanup alone cannot establish feasibility. Audit duplicate code and unused features across the owned source before deciding whether any scientific behavior must be retired. The present file-level tree remains a baseline inventory.

## Initial THESIS redundancy audit

This is static evidence, not a completed refactor or a parity claim. The mathematical reference is the MUSTARD flow in the method-mathematics notes; preserve the O0/O1/O2 and A0/A1/A2 computations and contracts.

### Mathematical-flow ownership map

| Flow to preserve | Current implementation owner |
|---|---|
| Encoder and query representations | `thesis_multitask_encoder.py`; `thesis_multitask_routing_helpers.py` |
| Continuous-memory selection and discrete class-stratified K-means initialization | `thesis_multitask_state_memory_mixin.py` |
| Continuous/discrete retrieval, stochastic samples, and fusion | `thesis_multitask_routing_helpers.py`; `thesis_multitask_routing_geometry_helpers.py`; `thesis_multitask_routing_forward_helpers.py` |
| Reconstruction, classification, O1 score loss, and O2 contrastive loss | `thesis_multitask_loss_core_mixin.py`; `thesis_multitask_loss_step_mixin.py`; `thesis_multitask_routing_mixin.py` |
| Reconstruction scoring and threshold calibration | `thesis_multitask.py`; `src/engine/online_tta/online_calibration.py` |
| A0/A1/A2 online routing and updates | `src/models/online_impl/online_adaptation.py`; `src/engine/online_tta/online_engine_step.py`; `src/engine/online_tta/online_engine_window_core.py` |
| Stage and online state serialization | `thesis_multitask_state_serialization_mixin.py`; `src/engine/online_tta/` checkpoint modules |

This map sets the comparison boundary: reducing the model folder alone cannot simplify the online mathematics or prove full MUSTARD flow preservation.

Confirmed redundant code:

- `_variance_from_samples` has the same validation and variance calculation in `thesis_multitask_routing_geometry_helpers.py` and `thesis_multitask_routing_mixin.py`. Keep one implementation and route both callers through it.
- `_normalize_memory_vectors` and `_normalize_hidden_for_memory` both call `F.normalize` with the same dimension and epsilon. Use one helper at all call sites.
- `_memory_initialization_substep_active` and `_fusion_warmup_substep_active` return the same Stage B predicate. Compute it once when building lifecycle state while retaining the existing two output fields.

Do not count `_build_refurbished_binary_targets` as dead: one test calls this forwarding wrapper. Review that test's intended contract before removing or redirecting it.

Proposed feature removals outside the documented O0/O1/O2 and A0/A1/A2 flows:

- Retire optional diversity, variance-floor, covariance, prototype-usage, and gate regularizers. The model YAMLs do not enable them, and they are outside the documented O0/O1/O2 objectives. One focused training test exercises them; update its scope to cover the retained objectives.
- Retire CKA-gated fusion. The model YAMLs explicitly disable it, and the mathematical reference does not include it in the canonical routing flow. Keep the configured direct and task-specific flows.
- Remove the constructor's model-summary diagnostic (78 counted lines). It does not participate in computation; retain essential errors and experiment provenance elsewhere.
- Retire gradient-conflict profiling as an experiment diagnostic, together with the one model config that enables it. It does not change the loss or update equations.

Keep Monte Carlo sample retention, the MLP encoder, and score loss because the config scan found them enabled or selected. The full removal map and recount remain pending. These THESIS cuts preserve the documented ablation flows, but do not yet demonstrate the 4,000-line total target.

## Whole-source baseline audit

The agreed counter reports 244 owned `.py`/`.sh` files and 42,297 counted lines: `src/` has 126 files and 23,452 lines; `scripts/` has 118 files and 18,845 lines. In this scan, 127 files exceed 100 lines, 83 exceed 200, and 48 contain at most 10 counted lines. The small files are mostly package markers and CLI/re-export wrappers; removing them requires caller and import-path review.

An AST scan found 52 cross-file groups of identical function bodies, with about 712 counted lines beyond one copy if every group could share one implementation. Treat this as an upper-bound lead, not a savings claim: copies may have different contracts, some are test-facing, and nested or partial overlap can inflate the sum.

High-confidence review targets:

- `scripts/analysis/summarize_anomaly_span_lengths.py` repeats nine data helpers already defined in `scripts/analysis/summarize_anomaly_span_lengths_helpers.py` (about 195 counted lines). The CLI already imports collection and summary functions from the helper module, while tests import repeated helper names from the CLI module. Re-exporting the helper names from the CLI can preserve those import paths while removing duplicate bodies; verify all callers before choosing this.
- `scripts/benchmarks/run_thesis_offline_benchmark.py` and `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py` repeat several forwarding wrappers. Keep public names needed by tests, but replace duplicate wrapper bodies with shared aliases where monkeypatch behavior remains the same.
- Root-level `scripts/*.py` files re-export commands from `scripts/cli/`, `scripts/experiments/`, and `scripts/benchmarks/`. Retire obsolete aliases and update documented commands/callers; keep one canonical entrypoint for each retained operation.
- Batch cloning, label-span normalization, tensor-to-device movement, JSON writing, and positive-span extraction have identical bodies in multiple owned modules. Centralize only after checking their input types and edge cases.
- RedLamp and THESIS gradient-profiling helpers substantially overlap. Since profiling is proposed for retirement from the retained MUSTARD flow, remove its MUSTARD copy; keep RedLamp's diagnostic only if the required RedLamp experiment still uses it.

These targets show source-level redundancy beyond THESIS. The remaining work is to validate each call path, account for retired diagnostics and wrappers, produce a file-action map, and recount. No source files were changed in this audit.

The follow-up static caller and script review is recorded in `source-baseline-audit.md`. It confirms safe helper-sharing candidates and identifies contract-sensitive tools through their tests and documented outputs. A conservative 18-file set for the current comparison matrix still contains 6,672 counted lines, with 15 files above 100 lines. The whole repository has 244 counted source files and 42,297 lines. Thus the 40-file, 100-lines-per-file, and 4,000-line targets remain unachieved; the confirmed duplicate candidates and script retirements do not close the gap. Reaching the target requires a complete source file-action map and a deliberate decision about retained runtime responsibilities before any implementation.

## Terminology mapping

| Old label | Proposed status | Supported baseline profile |
|---|---|---|
| `A0`, `A1`, `A2` on a non-THESIS baseline | Retired; no automatic migration | `main` for STUMPY, KMeansAD, and Isolation Forest |
| `A0`, `A1`, `A2` on CANDI/M2N2 | Retired; no automatic migration | `reference_adapter_redlamp_encoder` |
| `A0`, `A1`, `A2` on THESIS | Unchanged | THESIS/MUSTARD ablations |

The old baseline labels are not renamed or claimed equivalent to each other. They cease to be accepted as baseline profiles. Their historical run artifacts remain identifiable by their stored names.

## Not proposed for removal

Keep the MOMENT adapter. Although it is outside the current matrix, the simplification lessons explicitly identify it as preserved method support. Keep timing diagnostics and resource monitoring until a caller and experiment-use audit shows they are unused; current source and tests exercise them.

## Next evidence before implementation

The exact 270 config paths are listed in `legacy-baseline-config-removal.csv`. Before implementation, check baseline configuration validation and runner callers, then update only tests that explicitly define the retired baseline inputs. Do not remove tests for retained methods or THESIS ablations. Recount code and config files separately. No method behavior or experiment output is changed by this proposal itself.

## Current file-level baseline inventory

This inventory lists every current `.py` and `.sh` file under `src/` and `scripts/`. It has 244 counted source files and is not the future 40-file target tree. The 270 retired YAML paths are listed separately in `legacy-baseline-config-removal.csv`.

The profile restriction is implemented within the existing runner and baseline files; it removes no source module:

- `scripts/benchmarks/run_online_streaming_benchmark.py`: reject A-label profiles for non-THESIS baselines and pass each baseline its canonical profile.
- `src/baselines/online/frozen.py`: give frozen traditional baselines only the `main` profile.
- `src/baselines/online/adaptive.py`: require `reference_adapter_redlamp_encoder` for CANDI/M2N2.
- `tests/online/test_online_streaming_baseline_contracts.py`: update the retained-profile contract and cover rejection of retired labels.
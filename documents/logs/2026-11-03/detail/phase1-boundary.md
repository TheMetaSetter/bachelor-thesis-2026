# Simplification Phase 1: establish the boundary

Date: 2026-11-03  
Status: complete; no runtime source, configuration, or test files changed.

## Ordered checklist

- [x] 1. Read repository guidance and relevant specifications.
- [x] 2. Record the starting revision and existing edits.
- [x] 3. Inventory tracked, ignored, untracked, and generated files.
- [x] 4. Save and check a recoverable backup of local files.
- [x] 5. List required scientific methods and variants.
- [x] 6. Map methods to runtime dependencies and callers.
- [x] 7. Define and apply one size-counting rule.

## 1. Boundary sources read

Repository guidance: `AGENTS.md`, `codebase_preferences.md`, and
`python_semantics.md`.

Scientific sources: relevant sections of
`documents/abstract-design-notes/design_starter.md`,
`documents/abstract-design-notes/offline_pretraining_phase_two_view_contrastive_design.md`,
`documents/spec/full-spec-v3.md` (lifecycle and online behavior),
`documents/spec/full-spec-v4.md` (raw-input score contract), the offline and
online terminology ontologies, `documents/spec/online_benchmark_contract.md`,
and `documents/notes/smd-all-machines-experiment-matrix.md` (required method
and variant matrix), `documents/inventories/online-benchmark-combinations-and-smoke-checklist.md`,
and `documents/notes/experiment-flows-and-equations.md` (variant ownership).

The development specification `documents/spec/tsad-lib-development-spec.md`
explicitly describes the sibling `tsad-lib` library. Its proposed 40-model
catalog is not treated as a runtime method list for this repository. The
repository's local SMD matrix and actual source, config, and test paths set the
boundary here.

Preservation constraints: retain scientific methods and runnable paths; protect
reference sources, specifications, research notes, data, and artifacts; treat
checkpoint selection, calibration, online update order, and resume as behavior;
do not count removed features, tests, vendor code, or minified code as
simplification.

## 2. Starting state

- Repository: this `bachelor-thesis-2026` checkout.
- Branch: `simplifying`.
- Starting revision: `753ad03ed60b2d41b078e53bf421429bb0bd6c0f`.
- Modified tracked files: `.gitignore`, `AGENTS.md`, `pytest.ini`.
- Staged changes: none.
- Non-ignored untracked files: 38 (11 under `documents/`, 1 under `papers/`,
  and 26 under `prompts/`).
- Existing edits above are user state and must be preserved.

## 3. File inventory

Counts cover repository files outside `.git`. Generated candidates are a
subset of each status class, using paths under common output/cache/build
directories and common tensor, checkpoint, and serialized-data suffixes.

| Git status | Files | Logical bytes | Generated candidates |
|---|---:|---:|---:|
| Tracked | 2,984 | 784,044,704 | 152 files / 143,306,849 bytes |
| Untracked, not ignored | 38 | 1,442,992 | 0 |
| Ignored | 46,631 | 11,237,424,511 | 46,081 files / 7,541,579,820 bytes |

The ignored inventory is mainly `.venv` (45,370 files) and `data` (590 files).
The generated classification is a reproducible screening rule, not a claim
that every matching file is disposable or every generated artifact was found.

## 4. Backup

Backup: `/private/tmp/bachelor-thesis-phase1-backup-20261103`.

The backup was made before writing this phase report or counter. A recursive
file-count and byte-count check, excluding `.git` internals, matched the source:
49,653 files and 12,022,912,207 logical bytes. macOS reported that the
transient `.git/fsmonitor--daemon.ipc` socket could not be copied; Git metadata
files and all regular files were copied. The backup is on temporary storage,
so it is recoverable for this work session but is not durable project history.

## 5. Required methods and variants

The canonical experiment set follows the SMD matrix. THESIS is this
repository's runtime name for the MUSTARD method. `O0/O1/O2` are offline
ablation variants of THESIS/MUSTARD only. `A0/A1/A2` are its online ablation
variants only. Other methods remain separate comparison methods and use their
own method profile names.

| Method | Required offline variants | Required online variants | Evidence boundary |
|---|---|---|---|
| THESIS (MUSTARD) | `O0`, `O1`, `O2` | Each `O0/O1/O2` with `A0`, `A1`, `A2` | Matrix and offline/online ontologies; these ablations belong only to THESIS/MUSTARD |
| RedLamp | `main` | — | Matrix |
| STUMPY Channel AB | `main` | `main` | Matrix; runtime canonical name is `stumpy_channel_ab` offline and `stumpy` online |
| KMeansAD | `main` | `main` | Matrix; runtime config key is `kmeans_ad` |
| Isolation Forest | `main` | `main` | Matrix; runtime config key is `iforest` |
| CANDI | — | `reference_adapter_redlamp_encoder` | Matrix |
| M2N2 | — | `reference_adapter_redlamp_encoder` | Matrix |
| MOMENT adapter | No matrix variant | No matrix variant | Adapter and shape test exist; no active experiment config or matrix row found |

Config-file counts measure stored YAML files, not official experiment variants.
Baseline config families include older A-labeled files outside the matrix.

Older generated baseline configs also use `A0`, `A1`, or `A2` in filenames and
the generic `online_variant` field. Those labels do not make THESIS ablations
of CANDI, M2N2, or traditional methods. They are legacy baseline config
labels outside the official SMD matrix. Current generator profiles are
`reference_adapter_redlamp_encoder` for CANDI/M2N2 and `main` for traditional
methods. Keep legacy files until any cleanup is separately reviewed; do not
use their labels to infer equivalent behavior.

The local config audit found 270 such non-THESIS YAML files: 18 each for
`A0`, `A1`, and `A2` in CANDI, M2N2, STUMPY, KMeansAD, and Isolation Forest.
They remain on disk as legacy configs; the official generator does not include
them.

The matrix requires THESIS offline variants `O0`, `O1`, and `O2`, but the
offline config generator currently declares only `O0` and `O1`, and the
standard offline config directory contains no `O2` config. The direct-branch
helper is a separate path and does not establish the complete `O2` lifecycle
specified by the matrix. Keep `O2` in the preservation boundary and record its
standard runnable path as unresolved.

## 6. Runtime dependency and caller map

Shared data path: `src/data/loaders.py` builds SMD data using
`src/data/datasets/smd.py`, with windowing, scaling, and split logic in
`src/data/window.py`, `src/data/scalers.py`, and
`src/data/split_protocol.py`.

| Method | Runtime implementation | Config families | Entrypoints / callers | Focused evidence |
|---|---|---|---|---|
| THESIS offline | `src/models/thesis_multitask.py`; `src/engine/trainer.py`; checkpoint code in `src/engine/checkpoint.py` | `configs/experiment/{benchmark,benchmark_smoke,offline_benchmark}/thesis/`; standard variants found: `O0`, `O1` only | `scripts/experiments/run_two_stage_offline_pretraining.py`; `scripts/benchmarks/run_thesis_offline_benchmark.py`; CLI train/evaluate | `tests/models/test_*thesis*`; `tests/engine/test_checkpoint_roundtrip.py`; `tests/benchmarks/test_thesis_offline_artifact_exports.py`; no standard `O2` config found |
| THESIS online | `src/models/online_adaptation.py`; `src/engine/online_tta/`; `src/engine/online_loop.py` | `configs/experiment/online_benchmark/thesis/`; `online_diagnostic/thesis/` | `scripts/run_online_adaptation.py`; `scripts/benchmarks/run_online_streaming_benchmark.py`; `src/engine/online_tta/online_engine_run.py` | `tests/online/test_online_tta_variants.py`; `test_online_state_roundtrip.py`; `test_online_verification_buffer.py`; `test_threshold_artifact.py` |
| RedLamp | `src/models/redlamp_baseline.py`; `src/baselines/neural/redlamp.py` is an export; online encoder loader is `src/baselines/online/redlamp_encoder_checkpoint.py` | `configs/experiment/{benchmark,comparative}/baseline/` | CLI train/evaluate; comparative experiment runner; CANDI/M2N2 encoder initialization | `tests/models/test_redlamp_baseline.py`; `test_redlamp_baseline_runtime.py`; `tests/online/test_redlamp_encoder_checkpoint.py` |
| STUMPY Channel AB | `src/baselines/traditional/stumpy_channel_ab.py`; online frozen implementation in `src/baselines/online/frozen.py` | `configs/experiment/offline_benchmark/stumpy/`; `online_benchmark/stumpy/` | `scripts/benchmarks/run_offline_benchmark.py`; `run_online_streaming_benchmark.py` | `tests/evaluation/test_stumpy_channel_ab_contract.py`; `tests/online/test_online_streaming_baseline_contracts.py` |
| KMeansAD | `src/baselines/traditional/kmeans_ad.py`; online implementation in `src/baselines/online/frozen.py` | `configs/experiment/offline_benchmark/kmeans_ad/`; `online_benchmark/kmeans_ad/` | Same offline and online benchmark launchers | `tests/evaluation/test_kmeansad_contract.py`; online streaming baseline contract tests |
| Isolation Forest | `src/baselines/traditional/iforest.py`; online implementation in `src/baselines/online/frozen.py` | `configs/experiment/offline_benchmark/iforest/`; `online_benchmark/iforest/` | Same offline and online benchmark launchers | `tests/evaluation/test_iforest_contract.py`; online streaming baseline contract tests |
| CANDI | `src/baselines/online/candi.py` and `src/baselines/online/adaptive.py`; RedLamp encoder checkpoint loader | `configs/experiment/online_benchmark/candi/` | `scripts/benchmarks/run_online_streaming_benchmark.py` (`candi` builder) | `tests/online/test_online_method_fidelity.py`; streaming baseline contract tests |
| M2N2 | `src/baselines/online/m2n2.py` and `src/baselines/online/adaptive.py`; RedLamp encoder checkpoint loader | `configs/experiment/online_benchmark/m2n2/` | `scripts/benchmarks/run_online_streaming_benchmark.py` (`m2n2` builder) | `tests/online/test_online_method_fidelity.py`; streaming baseline contract tests |
| MOMENT adapter | `src/adapters/moment.py`; external `momentfm` package loaded on demand | No active experiment config found | Direct adapter use only; not in either benchmark builder map | `tests/models/test_moment_adapter_shapes.py` |

Checkpoint obligations are method-specific. THESIS uses Stage A/Stage B
initialization and selected checkpoints plus score/threshold identity.
RedLamp checkpoints supply encoder tensors to CANDI and M2N2. Traditional
methods use their own fitted/calibration state. A later move must inspect each
format and caller before changing ownership or serialization.

## 7. Size rule and baseline

Rule: file count includes each project-owned `.py` or `.sh` file below `src/`
and `scripts/` that has at least one code line after blank lines and comments
are ignored. Line and byte totals exclude blank lines and comment text,
including inline comments. Count UTF-8 bytes of retained source line contents,
excluding line endings. Docstrings remain because they are string literals,
not comments. Exclude tests, configs, vendored code, documents, generated
caches, and other research assets. Use the same selected files and rule for
every comparison.

Reusable counter: `count_owned_core_size.py`. It counts either the worktree or
tracked files at a requested Git revision.

| Baseline | Files | Lines | Bytes |
|---|---:|---:|---:|
| Revision `753ad03ed60b2d41b078e53bf421429bb0bd6c0f` | 243 | 42,278 | 1,705,839 |
| Current worktree at Phase 1 | 243 | 42,278 | 1,705,839 |

The matching counts confirm that existing local edits are outside the counted
core paths. This corrected code-only baseline replaces the earlier Phase 1
count, which included blank lines and comments. It does not establish that any
proposed reduction can preserve behavior.

## Open boundary item

Before any removal work, preserve the distinction between THESIS/MUSTARD
ablations (`O0/O1/O2`, `A0/A1/A2`) and each baseline's method profile. Legacy
baseline configs carrying A-labels remain outside the official matrix and
must not be counted as THESIS variants.

Also resolve the matrix-required THESIS `O2` runnable path. The current
standard config generator and config directory cover `O0/O1`; the separate
direct-branch helper does not establish the full `O2` lifecycle.

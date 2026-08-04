---
date: 2026-08-04
topic: "Full benchmark matrix and preparation checklist for the new THESIS online runtime smoke"
status: prepared
source_checkpoint_inventory: documents/inventories/detail-remote-gpu-checkpoints-inventory.md
source_runtime_detail: documents/logs/2026-08-04/detail/detail-online-runtime-desired-flow.md
---

# Online benchmark matrix and THESIS smoke checklist

## Main point

The official matrix has 18 THESIS offline runs and 72 online runs. The online
runs are 54 THESIS runs, 9 M2N2 runs, and 9 CANDI runs. Therefore the full
program has 90 run units.

A0, A1, and A2 are online variants of THESIS only. M2N2 and CANDI each have
one baseline online method without a THESIS A-variant. Their `online_variant`
config field must not create additional official baseline combinations.

Remote revalidation on 2026-08-04 found all 18 expected Stage-B `best.pt`
files. This only confirms their paths. It does not make old threshold artifacts
valid for the new THESIS runtime: the checked O1 / `machine_1_6` / seed6
artifact is V3 and has no checkpoint SHA-256, while the current THESIS online
runtime requires V4.

## Counts

| Group | Formula | Count |
| --- | --- | ---: |
| THESIS offline | 2 offline variants × 3 entities × 3 seeds | 18 |
| THESIS online | 2 offline variants × 3 online variants × 3 entities × 3 seeds | 54 |
| M2N2 online | 3 entities × 3 seeds | 9 |
| CANDI online | 3 entities × 3 seeds | 9 |
| Total online | 54 + 9 + 9 | 72 |
| Total run units | 18 offline + 72 online | 90 |

## Verified THESIS offline combinations and Stage-B checkpoints

The following 18 paths were verified read-only on the remote GPU checkout.
They are valid locations for Stage-B files only. Each online THESIS run still
needs a V4 threshold artifact from the same offline run.

| Offline variant | Entity | Seed | Verified remote Stage-B checkpoint |
| --- | --- | ---: | --- |
| O0 | `machine_1_6` | 6 | `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_1_6` | 8 | `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_1_6` | 36 | `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_3_4` | 6 | `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_3_4` | 8 | `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_3_4` | 36 | `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_3_9` | 6 | `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_3_9` | 8 | `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O0 | `machine_3_9` | 36 | `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_1_6` | 6 | `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_1_6` | 8 | `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_1_6` | 36 | `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_3_4` | 6 | `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_3_4` | 8 | `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_3_4` | 36 | `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_3_9` | 6 | `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_3_9` | 8 | `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |
| O1 | `machine_3_9` | 36 | `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt` |

See the revalidated source inventory in
[detail-remote-gpu-checkpoints-inventory.md](detail-remote-gpu-checkpoints-inventory.md).

## Full combination matrix

Every non-header cell is one concrete combination. A THESIS online cell uses
the Stage-B checkpoint in the previous table with the same O-variant, entity,
and seed.

### THESIS offline: 18 combinations

| Offline variant | `machine_1_6` / seed6 | `machine_1_6` / seed8 | `machine_1_6` / seed36 | `machine_3_4` / seed6 | `machine_3_4` / seed8 | `machine_3_4` / seed36 | `machine_3_9` / seed6 | `machine_3_9` / seed8 | `machine_3_9` / seed36 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O0 | `THESIS/O0/machine_1_6/seed6` | `THESIS/O0/machine_1_6/seed8` | `THESIS/O0/machine_1_6/seed36` | `THESIS/O0/machine_3_4/seed6` | `THESIS/O0/machine_3_4/seed8` | `THESIS/O0/machine_3_4/seed36` | `THESIS/O0/machine_3_9/seed6` | `THESIS/O0/machine_3_9/seed8` | `THESIS/O0/machine_3_9/seed36` |
| O1 | `THESIS/O1/machine_1_6/seed6` | `THESIS/O1/machine_1_6/seed8` | `THESIS/O1/machine_1_6/seed36` | `THESIS/O1/machine_3_4/seed6` | `THESIS/O1/machine_3_4/seed8` | `THESIS/O1/machine_3_4/seed36` | `THESIS/O1/machine_3_9/seed6` | `THESIS/O1/machine_3_9/seed8` | `THESIS/O1/machine_3_9/seed36` |

### THESIS online: 54 combinations

| Offline / online variant | `machine_1_6` / seed6 | `machine_1_6` / seed8 | `machine_1_6` / seed36 | `machine_3_4` / seed6 | `machine_3_4` / seed8 | `machine_3_4` / seed36 | `machine_3_9` / seed6 | `machine_3_9` / seed8 | `machine_3_9` / seed36 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O0 / A0 | `THESIS/O0/A0/machine_1_6/seed6` | `THESIS/O0/A0/machine_1_6/seed8` | `THESIS/O0/A0/machine_1_6/seed36` | `THESIS/O0/A0/machine_3_4/seed6` | `THESIS/O0/A0/machine_3_4/seed8` | `THESIS/O0/A0/machine_3_4/seed36` | `THESIS/O0/A0/machine_3_9/seed6` | `THESIS/O0/A0/machine_3_9/seed8` | `THESIS/O0/A0/machine_3_9/seed36` |
| O0 / A1 | `THESIS/O0/A1/machine_1_6/seed6` | `THESIS/O0/A1/machine_1_6/seed8` | `THESIS/O0/A1/machine_1_6/seed36` | `THESIS/O0/A1/machine_3_4/seed6` | `THESIS/O0/A1/machine_3_4/seed8` | `THESIS/O0/A1/machine_3_4/seed36` | `THESIS/O0/A1/machine_3_9/seed6` | `THESIS/O0/A1/machine_3_9/seed8` | `THESIS/O0/A1/machine_3_9/seed36` |
| O0 / A2 | `THESIS/O0/A2/machine_1_6/seed6` | `THESIS/O0/A2/machine_1_6/seed8` | `THESIS/O0/A2/machine_1_6/seed36` | `THESIS/O0/A2/machine_3_4/seed6` | `THESIS/O0/A2/machine_3_4/seed8` | `THESIS/O0/A2/machine_3_4/seed36` | `THESIS/O0/A2/machine_3_9/seed6` | `THESIS/O0/A2/machine_3_9/seed8` | `THESIS/O0/A2/machine_3_9/seed36` |
| O1 / A0 | `THESIS/O1/A0/machine_1_6/seed6` | `THESIS/O1/A0/machine_1_6/seed8` | `THESIS/O1/A0/machine_1_6/seed36` | `THESIS/O1/A0/machine_3_4/seed6` | `THESIS/O1/A0/machine_3_4/seed8` | `THESIS/O1/A0/machine_3_4/seed36` | `THESIS/O1/A0/machine_3_9/seed6` | `THESIS/O1/A0/machine_3_9/seed8` | `THESIS/O1/A0/machine_3_9/seed36` |
| O1 / A1 | `THESIS/O1/A1/machine_1_6/seed6` | `THESIS/O1/A1/machine_1_6/seed8` | `THESIS/O1/A1/machine_1_6/seed36` | `THESIS/O1/A1/machine_3_4/seed6` | `THESIS/O1/A1/machine_3_4/seed8` | `THESIS/O1/A1/machine_3_4/seed36` | `THESIS/O1/A1/machine_3_9/seed6` | `THESIS/O1/A1/machine_3_9/seed8` | `THESIS/O1/A1/machine_3_9/seed36` |
| O1 / A2 | `THESIS/O1/A2/machine_1_6/seed6` | `THESIS/O1/A2/machine_1_6/seed8` | `THESIS/O1/A2/machine_1_6/seed36` | `THESIS/O1/A2/machine_3_4/seed6` | `THESIS/O1/A2/machine_3_4/seed8` | `THESIS/O1/A2/machine_3_4/seed36` | `THESIS/O1/A2/machine_3_9/seed6` | `THESIS/O1/A2/machine_3_9/seed8` | `THESIS/O1/A2/machine_3_9/seed36` |

### M2N2 and CANDI online: 18 combinations

| Baseline method | `machine_1_6` / seed6 | `machine_1_6` / seed8 | `machine_1_6` / seed36 | `machine_3_4` / seed6 | `machine_3_4` / seed8 | `machine_3_4` / seed36 | `machine_3_9` / seed6 | `machine_3_9` / seed8 | `machine_3_9` / seed36 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2N2 | `M2N2/machine_1_6/seed6` | `M2N2/machine_1_6/seed8` | `M2N2/machine_1_6/seed36` | `M2N2/machine_3_4/seed6` | `M2N2/machine_3_4/seed8` | `M2N2/machine_3_4/seed36` | `M2N2/machine_3_9/seed6` | `M2N2/machine_3_9/seed8` | `M2N2/machine_3_9/seed36` |
| CANDI | `CANDI/machine_1_6/seed6` | `CANDI/machine_1_6/seed8` | `CANDI/machine_1_6/seed36` | `CANDI/machine_3_4/seed6` | `CANDI/machine_3_4/seed8` | `CANDI/machine_3_4/seed36` | `CANDI/machine_3_9/seed6` | `CANDI/machine_3_9/seed8` | `CANDI/machine_3_9/seed36` |

## Config status and required correction

The THESIS generator correctly creates 54 main configurations. The current
baseline generator instead creates 27 main configurations for M2N2 and 27 for
CANDI because it assigns A0, A1, and A2 to each baseline. This conflicts with
the decision above. Do not launch those 54 baseline configs as the official
matrix.

Before the official benchmark, the baseline configuration contract needs one
clear correction: produce one main config per baseline/entity/seed, or state a
separate baseline-specific variant name. Do not silently use the THESIS A0/A1/A2
names for baselines. This document records the matrix decision only; it does
not modify source code or configs.

## Checklist: prepare one smoke run for the new THESIS online runtime

### 1. Fix the smoke scope

- [ ] Use one concrete functional smoke: `THESIS / O1 / A2 / machine_1_6 / seed6`.
- [ ] Stream `[5608,5909)`, not the first 16 windows of the whole test series.
  The range has 301 points. With window size 20 and stride 1, it creates 282
  causal windows.
- [ ] Set both `absolute_start_index: 5608` and `absolute_end_index: 5909`.
- [ ] Set `max_online_steps: null` or at least 282.
- [ ] Do not use this smoke's metric values in the performance comparison.

The online runner supports the range and retains its absolute offset in each
window. See [range selection](../../src/engine/online_tta/online_engine_run.py#L55-L95)
and [runtime wiring](../../src/engine/online_tta/online_engine_run.py#L553-L562).

### 2. Check remote state safely

- [ ] Re-check the remote host key, source revision, `git status --short`, CUDA,
  free disk space, and active GPU jobs just before the write run.
- [ ] Confirm that remote source still matches the local revision containing the
  new flow. The last read-only check found revision
  `2200db1ed49024e06b5a85d09634f93e19febc71` on both sides.
- [ ] Do not copy a local patch, reset the remote worktree, or overwrite an
  output directory merely to execute the smoke. If the source differs, stop
  for an approved deployment action.

### 3. Create a new matching offline smoke artifact

- [ ] Start from
  [smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml](../../configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml).
- [ ] Keep its 2 Stage-A epochs and 1 Stage-B epoch.
- [ ] Run offline once on the remote. It must produce a new Stage-B `best.pt`
  and a threshold artifact under the same `benchmark_smoke` offline run.
- [ ] Verify the new artifact has schema version 4, four canonical thresholds,
  a non-empty checkpoint SHA-256, entity `machine_1_6`, seed 6, offline variant
  O1, window size 20, and a hash equal to the new Stage-B `best.pt` hash.
- [ ] Record the exact Stage-B and threshold-artifact paths. THESIS requires an
  explicit `task.threshold_artifact_path`; it does not infer one. See
  [artifact resolution](../../src/engine/online_tta/checkpoint_resolution.py#L101-L126).

### 4. Prepare the functional A2 config

- [ ] Use the existing range config as a starting point:
  [smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml](../../configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml).
- [ ] Point its checkpoint and `task.threshold_artifact_path` to the new O1
  offline smoke artifacts from step 3.
- [ ] Keep `device: cuda`, window size 20, and stride 1.
- [ ] Set `debug_timing: false` for the functional smoke. Make a separate,
  labelled diagnostic run with `debug_timing: true` only when measuring a
  bottleneck. The logger synchronizes CUDA around every timed component and
  therefore changes runtime. See
  [timing implementation](../../src/engine/online_tta/timing_debug.py#L13-L46).
- [ ] Use a new empty smoke output directory. Do not reuse a directory that
  contains another artifact identity.

### 5. Validate flow coverage

- [ ] Run focused online tests before remote execution: artifact, range,
  runtime-state, A0/A1/A2, verification, and benchmark-wrapper tests.
- [ ] Run a dry-run first. Confirm it resolves the intended checkpoint and V4
  artifact paths.
- [ ] In A2 event records, check `causal_window.absolute_indices`,
  `window_point_scores`, `current_window_ewma_point_scores`, and
  `triage_region`.
- [ ] Check the implemented order: score and vector EWMA, triage, current
  hard-old action when admissible, then gray-zone admission and verification.
  See [per-window flow](../../src/engine/online_tta/online_engine_window_core.py#L180-L230).
- [ ] Check whether this data visits a gray zone and triggers verification. If
  it does not, record the uncovered PNN/adaptation branch. A successful process
  does not prove that branch ran.
- [ ] Confirm runtime state uses schema version 2 and the active point-EWMA map.
- [ ] Preserve only report-ready artifacts: the Stage-B checkpoint, V4
  threshold artifact, resolved config/provenance, final online checkpoint,
  report, metrics, and selected event diagnostics.

## Known blockers before this functional A2 smoke

1. The revalidated checkpoint inventory locates Stage-B files, but the checked
   corresponding artifact is V3 and invalid for the current V4 runtime.
2. The generic generated THESIS smoke config has `max_online_steps: 16`; it is
   not a substitute for the `[5608,5909)` range smoke.
3. Current baseline generated configs incorrectly reuse A0/A1/A2. They need a
   separate correction before the official 90-run matrix starts.
4. `runtime_protocol_status` remains `full_spec_v2` until focused tests and one
   real smoke path pass. A smoke run must not silently change that status.

## Evidence boundary

Checkpoint paths were verified on 2026-08-04. The V3 finding was directly
checked only for O1 / `machine_1_6` / seed6. Check every other checkpoint and
artifact pair independently before its online run.

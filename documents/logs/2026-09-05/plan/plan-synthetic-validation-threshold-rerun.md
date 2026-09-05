---
date: 2026-09-05T17:09:13+07:00
planner: OpenAI Codex
topic: "Synthetic-validation normal-score q99 threshold rerun"
status: ready
revision: 5529a0c3f1ab9f4b7f013543aa7e61922b74b56d
branch: dev
related_research: /Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/2026-09-05/research/research-synthetic-validation-threshold-change-surface.md
---

# Implementation Plan: Synthetic-validation normal-score q99 threshold rerun

## Summary

Add an opt-in offline evaluation mode that computes one point threshold as the q99 of finite anomaly scores from covered normal samples in `val_synth`, then reuses that point threshold for synthetic validation metrics and real test metrics. Preserve the current clean-validation protocol and output paths when the new option is absent. Write the rerun artifacts under a separate output directory and provide a command for all Stage-B best checkpoints on `cloud-gpu`.

## Request

Change the evaluation code and write the rerun command. The threshold is `q99`. The scope is every Stage-B `best.pt` checkpoint on `cloud-gpu`. The results must go to a new rerun directory. The implementation must remain backward-compatible.

## Current state

The active runner selects the offline point threshold from clean-validation scores in `scripts/benchmarks/run_thesis_offline_benchmark.py:452-544`. The evaluator already supports `val_synth`, covered-point reconstruction, and an explicit point threshold in `src/engine/evaluator.py:421-433` and `:537-669`. Artifact export and the report currently use `experiment_config["output_dir"]` at `scripts/benchmarks/run_thesis_offline_benchmark.py:923-970`. The default protocol locks `offline_threshold_split: clean_validation` at `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:6`.

## Desired end state

With an explicit new protocol/configuration, the runner evaluates synthetic validation once to collect scores, keeps only covered points with normal synthetic labels, removes non-finite scores, computes q99, evaluates synthetic validation and test with that fixed point threshold, and records the source in the offline threshold record. Existing commands without the new option retain the current clean-validation behavior and artifact contract.

## Scope

### In scope

- A small threshold-selection helper or equivalent focused logic for covered synthetic normal scores.
- Opt-in protocol configuration for `synthetic_validation_normal` as the offline point source.
- Re-evaluation flow and threshold provenance in the active benchmark runner.
- Optional output-directory override for isolated reruns.
- Focused regression and contract tests.
- A command template for all 18 Stage-B best checkpoints on `cloud-gpu`, with one preflight before the matrix.

### Out of scope

- Changing model training, Stage-A/Stage-B checkpoint selection, or online EWMA calibration source.
- Changing the default clean-validation protocol or existing output trees.
- Modifying the evaluator's `val_synth` label mapping unless a test proves it is required.
- Running the remote matrix in this planning task.

## Terminology and compatibility decision

Keep `offline_threshold_split: clean_validation` and the top-level threshold-artifact `calibration_split: clean_validation` as the legacy/default contract. Add an optional, explicitly named `offline_point_threshold_source_split` field. Its absent value falls back to the current clean-validation path; its new value `synthetic_validation_normal` selects covered normal synthetic points. The offline point threshold record may report this source separately, while online threshold records and legacy artifacts remain clean-validation based.

## Phases

### Phase 1: Establish the opt-in contract

Define the new protocol field, allowed value, source terminology, and output-directory override without changing existing defaults.

**Tools:** YAML, Python parser inspection, existing protocol and artifact validators.

**Result:** Old configs still validate unchanged; a new config can request synthetic-normal q99 and a separate output root.

### Phase 2: Implement synthetic-normal threshold selection

Update the active offline split flow to collect `val_synth` scores before thresholding, filter normal covered points, reject an empty usable set, compute q99, and reuse one point threshold for synthetic/test evaluation.

**Tools:** Python, NumPy, existing `Evaluator`, `.venv/bin/python`.

**Result:** The requested threshold is selected from the intended score population, without using test labels or uncovered timeline positions.

### Phase 3: Preserve artifact and output compatibility

Thread the selected threshold source through threshold artifact construction and route artifacts, retention bundles, and reports to the opt-in rerun directory.

**Tools:** Python, JSON/YAML artifact writers, existing retention helpers.

**Result:** New runs are isolated and auditable; old artifacts remain readable and unchanged.

### Phase 4: Add focused regression coverage

Test the helper/filtering rules, two-pass synthetic evaluation, source metadata, output override, and legacy default behavior.

**Tools:** `pytest`, `.venv/bin/python`, existing benchmark and artifact fixtures.

**Result:** Automated checks prove both new behavior and backward compatibility.

### Phase 5: Prepare the cloud rerun command

Write a short command sequence for all 18 Stage-B best checkpoints, using the new protocol and a new rerun directory. Re-read the current SSH endpoint and checkpoint inventory before execution; run one concrete preflight before the matrix.

**Tools:** Bash, SSH, `tmux`, `.venv/bin/python`, remote read-only checks.

**Result:** An inspectable command is ready for the authorized remote rerun; no remote job starts in this planning task.

## Verification

### Automated

- Run focused threshold, protocol, artifact, and benchmark tests with `.venv/bin/python -m pytest`.
- Run `git diff --check` after implementation.
- Parse the new protocol and confirm the legacy protocol still validates.
- Verify a synthetic fixture produces `np.quantile(normal_covered_scores, 0.99)` and excludes anomalies, uncovered points, NaN, and infinity.

### Manual

- Confirm the new rerun path differs from every existing output path.
- On `cloud-gpu`, inspect one checkpoint/config pair and one completed artifact bundle before launching the remaining 17 combinations.

## Risks and recovery

- **Legacy behavior changes accidentally:** keep the new field optional and default to the existing clean-validation branch; run legacy tests. **Recovery:** omit the new field and use the unchanged protocol/config.
- **Synthetic anomalies enter calibration:** filter by synthetic normal labels after covered-point reconstruction; test a mixed fixture. **Recovery:** stop before remote matrix if the source metadata or counts are wrong.
- **Artifacts overwrite prior evidence:** require an explicit new output root and verify it before running. **Recovery:** stop the exact job and do not remove existing trees.
- **Remote inventory is stale:** perform a read-only inventory immediately before command generation/execution. **Recovery:** regenerate the command from the current inventory.

## Completion condition

The implementation plan is complete when Phases 1–4 are implemented and tested, the old default path remains green, the new protocol and output-root behavior are documented, and Phase 5 contains a command that targets the current 18 Stage-B best checkpoints without overwriting existing outputs.

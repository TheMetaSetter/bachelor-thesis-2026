---
date: 2026-09-12 19:42:46 +0700
researcher: OpenAI Codex
topic: "Audit THESIS runtime flows with current code and verify the A2 verification path"
status: complete
revision: b8fff201cdfdb27aa9cec377d607487c4c2755cb
branch: dev
---

# Research: Audit THESIS runtime flows with current code

## Summary

The current `A2` runtime does contain a `VerificationBuffer` and a verification trigger.

The trigger runs only for `gray_zone` windows after the buffer reaches capacity and has new entries.

The current `A2` `hard_old_normality` path updates the projector directly after `NonOverlapGuard` approval.

Therefore, the existing `A2` diagram is incomplete because it omits triage, gray-zone admission, buffer capacity, verification, and `pnn_mask` creation.

The current `A1` runtime also differs from the matrix policy because it updates only through verified non-empty PNN reconstruction and does not use online-to-source contrastive loss.

This audit researched the current implementation only and did not modify runtime code.

## Research question

Audit the runtime flows in the THESIS component matrix against the current codebase.

Confirm whether every `<...>-A2` flow contains a verification buffer or a verification trigger.

## System context

`run_thesis_online_benchmark.py` accepts `A0`, `A1`, and `A2`, resolves the matching Stage-B checkpoint, validates the matching threshold artifact, and calls `run_thesis_online_tta_experiment`.

`online_engine_run.py` creates the online model, creates an optimizer only for non-`A0`, creates a `VerificationBuffer`, and streams one causal window at a time.

`online_engine_window_core.py` performs scoring, EWMA, triage, hard-old gating, adaptation, gray-zone admission, and verification.

`online_engine_step.py` selects the loss and performs a projector update when the caller supplies an optimizer and the variant-specific gate accepts the event.

## Confirmed execution path

### Common startup for `O0-A*`, `O1-A*`, and `O2-A*`

The online wrapper validates the online variant and resolves the matching Stage-B checkpoint at `scripts/benchmarks/run_thesis_online_benchmark.py:292-325`.

The runtime validates threshold-artifact identity against entity, offline variant, seed, window size, checkpoint hash, EWMA weights, score space, and score transform at `src/engine/online_tta/online_engine_run.py:81-123`.

The runtime constructs the online model from the offline checkpoint and configures the optimizer only when the online variant is not `A0` at `src/engine/online_tta/online_engine_run.py:164-188`.

The runtime creates `VerificationBuffer(max_size=64, non_overlap_gap=0)` and `NonOverlapGuard(max_size=1)` at `src/engine/online_tta/online_engine_run.py:207-250`.

The online model freezes both inherited encoders and makes only `online_mlp_projector` trainable for `A1` and `A2` at `src/models/online_impl/online_adaptation.py:95-112` and `src/models/online_impl/online_adaptation.py:218-229`.

### Common per-window path

The runtime builds a causal stream and processes windows in temporal order at `src/engine/online_tta/online_engine_run.py:293-343`.

The runtime moves the window to the target device, runs either the frozen source path or the projected online path, extracts scores, and applies point-level EWMA at `src/engine/online_tta/online_engine_window_metrics.py:87-151`.

For `A1` and `A2`, the runtime classifies the window into `normal`, `hard_old_normality`, `gray_zone`, or `strong_anomaly` at `src/engine/online_tta/online_engine_window_core.py:79-90` and `src/engine/online_tta/triage.py:17-41`.

### Actual `A0` flow

```text
A0 config
  -> Stage-B checkpoint and threshold artifact
  -> frozen source model
  -> causal window
  -> source forward pass
  -> point/window scores
  -> point-level EWMA
  -> prediction and metric record
```

`A0` uses `forward_source` and does not create an adaptation step through the online projector at `src/engine/online_tta/online_engine_window_metrics.py:154-162` and `src/models/online_impl/online_adaptation.py:315-335`.

`A0` skips buffer admission and verification because `_process_online_window` calls the buffer path only when `online_variant != "A0"` at `src/engine/online_tta/online_engine_window_core.py:243-255`.

### Actual `A1` flow

```text
A1 config
  -> Stage-B checkpoint and threshold artifact
  -> frozen source model + trainable projector
  -> causal window
  -> projected forward pass
  -> point/window scores
  -> point-level EWMA
  -> four-region triage
  -> gray-zone admission into VerificationBuffer
  -> capacity and new-entry trigger
  -> source-prototype verification
  -> pnn_mask
  -> if pnn_mask is non-empty: masked PNN reconstruction update
  -> prediction and metric record
```

The `A1` step accepts only `triage_decision == "pnn_verified"` at `src/engine/online_tta/online_engine_step.py:119-144`.

The current `A1` step does not add online-to-source contrastive loss because its contrastive term is explicitly set to zero at `src/engine/online_tta/online_engine_step.py:143-144`.

Gray-zone windows enter the buffer at `src/engine/online_tta/online_engine_window_metrics.py:208-242`.

The verification controller runs only when the buffer reaches capacity and has new entries at `src/engine/online_tta/verification_cycle.py:21-36`.

The verification adapter filters known anomaly tokens, finds recurrent signatures, and creates `pnn_mask` at `src/engine/online_tta/verification_adapter.py:114-161`.

Verified entries with a non-empty `pnn_mask` call `execute_online_tta_step` with `triage_decision="pnn_verified"` at `src/engine/online_tta/online_engine_window_metrics.py:42-77`.

### Actual `A2` flow

```text
A2 config
  -> Stage-B checkpoint and threshold artifact
  -> frozen source model + trainable projector
  -> causal window
  -> projected forward pass
  -> point/window scores
  -> point-level EWMA
  -> four-region triage
      -> normal: no update
      -> strong_anomaly: no update
      -> hard_old_normality + non-overlap approval:
           hard-old hinge loss + online-to-source contrastive loss
           -> projector update
           -> record hard-old interval
      -> gray_zone:
           VerificationBuffer admission
           -> capacity and new-entry trigger
           -> frozen-source prototype verification
           -> known-anomaly filtering + recurrent-signature filtering
           -> pnn_mask
           -> if pnn_mask is non-empty:
                masked PNN reconstruction loss
                + online-to-source contrastive loss
                -> projector update
  -> prediction and metric record
```

The hard-old branch is selected after `classify_online_window` and `NonOverlapGuard.accept` at `src/engine/online_tta/online_engine_window_core.py:79-90`.

The accepted hard-old event calls `execute_online_tta_step` with an optimizer and `triage_decision="hard_old_normality"` at `src/engine/online_tta/online_engine_window_core.py:108-134`.

The `A2` hard-old update combines hard-old reconstruction loss with contrastive loss at `src/engine/online_tta/online_engine_step.py:145-180`.

The hard-old interval is added to the separate non-overlap guard only after a successful update at `src/engine/online_tta/online_engine_window_core.py:132-134`.

The `A2` gray-zone path calls `_admit_and_verify_gray_zone` after the current-window action at `src/engine/online_tta/online_engine_window_core.py:243-255`.

That helper admits only `gray_zone` entries and invokes `VerificationCycleController.maybe_run` at `src/engine/online_tta/online_engine_window_metrics.py:208-242` and `src/engine/online_tta/online_engine_window_core.py:137-171`.

When verification returns a non-empty PNN mask, the `A2` PNN update combines masked PNN reconstruction loss with online-to-source contrastive loss at `src/engine/online_tta/online_engine_step.py:145-180`.

## Variant-flow audit

| Matrix flow | Current code result | Audit status |
| --- | --- | --- |
| `O0-A0` | Source forward pass, EWMA, no triage, no buffer, no verification, no update | Confirmed for online behavior |
| `O1-A0` | Same online path as `O0-A0`; only the loaded offline checkpoint differs | Confirmed for online behavior |
| `O2-A0` | No separate O2 runtime branch; it would use the same path if an O2 checkpoint resolved | Not executable from current O2 inventory |
| `O0-A1` | Triage, gray-zone buffer, verification cycle, non-empty PNN reconstruction update | Conflicts with matrix policy |
| `O1-A1` | Same A1 online path after loading the O1 checkpoint | Conflicts with matrix policy |
| `O2-A1` | No separate O2 runtime branch; current O2 checkpoint/config inventory is incomplete | Not executable from current O2 inventory |
| `O0-A2` | Hard-old direct update or gray-zone buffer-triggered verification and PNN update | Matrix diagram incomplete |
| `O1-A2` | Same A2 online path after loading the O1 checkpoint | Matrix diagram incomplete |
| `O2-A2` | No separate O2 runtime branch; current O2 checkpoint/config inventory is incomplete | Not executable from current O2 inventory |

`O0`, `O1`, and the proposed `O2` are not selected by separate online adaptation branches in the current online engine.

The online engine receives the resolved offline checkpoint and uses its inherited model state, while `online_variant` selects the online behavior.

## Answer to the A2 verification requirement

The current `A2` implementation satisfies the weaker requirement that an `A2` runtime contains a verification buffer and a verification trigger.

The current `A2` implementation does not satisfy the stronger requirement that every `A2` adaptation update must first pass through verification.

The hard-old `A2` update bypasses `VerificationBuffer` and uses `NonOverlapGuard` instead.

The gray-zone `A2` update uses `VerificationBuffer`, `VerificationCycleController`, prototype verification, and `pnn_mask` before updating.

The matrix `A2` diagram must therefore show two distinct adaptation branches: direct guarded hard-old adaptation and verification-triggered gray-zone PNN adaptation.

## Evidence from tests

The focused online test set passed with `12 passed in 5.93s`.

`tests/online/test_online_tta_variants.py:176-193` verifies that a direct `A2` hard-old event updates the projector and records contrastive loss.

`tests/online/test_online_verification_buffer.py:35-69` verifies gray-zone admission and confirms that normal, hard-old, and strong-anomaly decisions do not enter the verification buffer through that helper.

`tests/online/test_verification_cycle.py:7-22` verifies capacity-triggered verification and one TTL tick.

These tests establish the two separate A2 paths, but they do not provide one end-to-end test proving that a full A2 stream reaches a verification cycle.

## Conflicts and uncertainties

The matrix policy says every `A1` has EWMA, online-to-source contrastive loss, and hard-old-normality adaptation, but the authoritative online ontology says `A1` uses only verified non-empty PNN reconstruction at `documents/spec/online_tta_terminology_ontology.md:80-88`.

The matrix policy says `A2` adds pseudo-new-normality adaptation, while the current code represents this through `pnn_mask` produced only by the verification path at `src/engine/online_tta/verification_adapter.py:146-155`.

The current metric builder sets `online/num_pseudo_new_normality_points` to `0` at `src/engine/online_tta/online_engine_window_metrics.py:310-323`, so the available code does not prove that this count is reported after a successful verification update.

The current `O2` generator, ontology, and checkpoint inventory are outside this online runtime audit and remain incomplete according to the matrix note.

## Open questions

The available code does not establish whether the intended revised policy requires hard-old A2 updates to pass through the verification buffer.

The available code does not establish whether `A1` should retain its current PNN reconstruction contract or be changed to the revised matrix contract.

The available code does not establish whether pseudo-new-normality adaptation should be counted as a PNN reconstruction update or as a separate adaptation operation.

## Evidence index

- `scripts/benchmarks/run_thesis_online_benchmark.py:292-325` — validates the online variant, resolves the Stage-B checkpoint, and creates the logger.
- `src/engine/online_tta/online_engine_run.py:164-188` — builds the model and optimizer boundary.
- `src/engine/online_tta/online_engine_run.py:207-250` — creates runtime state, verification buffer, and hard-old guard.
- `src/engine/online_tta/online_engine_run.py:293-363` — streams and processes causal windows.
- `src/engine/online_tta/online_engine_window_core.py:79-171` — performs triage, hard-old gating, buffer admission, and verification triggering.
- `src/engine/online_tta/online_engine_window_core.py:243-262` — invokes buffer and verification only for non-`A0` variants.
- `src/engine/online_tta/online_engine_step.py:119-193` — defines the A1 and A2 update branches and losses.
- `src/engine/online_tta/online_engine_window_metrics.py:27-84` — applies verified results and calls the update step.
- `src/engine/online_tta/online_engine_window_metrics.py:208-242` — admits only gray-zone entries.
- `src/engine/online_tta/verification_buffer.py:8-80` — defines non-overlap admission, capacity, TTL, and cycle state.
- `src/engine/online_tta/verification_cycle.py:12-36` — triggers and commits a verification cycle.
- `src/engine/online_tta/verification_adapter.py:114-161` — produces verification results and `pnn_mask`.
- `documents/spec/online_tta_terminology_ontology.md:284-415` — documents the authoritative triage, verification, and loss contracts.
- `documents/notes/smd-all-machines-experiment-matrix.md:200-224` — contains the audited target A2 diagram.

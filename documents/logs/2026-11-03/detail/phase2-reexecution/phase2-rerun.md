# Phase 2 rerun: design and capture behavior

Date: 2026-11-03

Basis: updated `phase1-boundary.md` and `simplification-lessons.md`.
THESIS is the runtime name for MUSTARD. `O0/O1/O2` and `A0/A1/A2` are
THESIS/MUSTARD ablations only; baseline profiles are method-specific.

No runtime source or test files were changed during this rerun. The earlier
Phase 2 artifacts were left in place. This folder contains fresh outputs.

## Ordered checklist

- [x] 8. Regenerate the proposed file-level tree in `proposed-next-version-tree.md`.
- [x] 9. Map all 2,984 tracked paths in `path-map.csv`. The map records four
  implementation moves, two public forwarder replacements, and two internal
  package markers proposed for removal. All other paths stay unchanged.
- [x] 10. Audit runnable method paths in `step-10-runnable-path-audit.md`.
  Required baseline paths use their own profiles. The standard THESIS O2 path
  remains unresolved.
- [x] 11. Run the unchanged full suite with
  `.venv/bin/python -m pytest`: **508 passed, 2 skipped, 33 warnings in
  49.71 seconds**. Full output: `pytest.log`.
- [ ] 12. Capture and repeat seeded cases for every currently executable
  method and branch. The supported cases matched exactly; the complete step
  remains open because no standard O2 path or matching checkpoint exists.

## Step 7 size-baseline check

The revised comment and blank-line rules were applied with the reusable
counter. Both `HEAD` and the worktree count to **243 code files, 42,278 code
lines, and 1,705,839 UTF-8 code bytes**.

## Seeded capture results

`capture-models.py` saved a fresh snapshot and compared an immediate repeat.
It covered THESIS MLP/CNN Stage A and Stage B with both fusion routes; THESIS
O0/O1 objectives; A0, A1, and both A2 decision routes; RedLamp MLP/CNN;
online-adaptation training; and the MOMENT adapter's padding path using a
deterministic model fixture. Result: **18 cases and 3,025 tensors matched
exactly**. This checks adapter and runtime behavior, not parity with the
external pretrained MOMENT model.

`capture-baselines.py` covered offline STUMPY, KMeansAD, and Isolation Forest;
STUMPY's constant-channel case; and online CANDI, M2N2, STUMPY, KMeansAD, and
Isolation Forest using their supported method profiles. Result: **9 cases and
10 saved arrays matched exactly**. Logs: `model-capture-with-moment.log` and
`baseline-capture.log`; snapshots are stored in this folder.

The first nested-script invocation lacked the repository import path and
stopped before running a case. Re-running with `PYTHONPATH=.` succeeded. The
successful commands and outputs are recorded in the capture logs.

The captures are seeded synthetic comparisons, not real-data experiments.
They do not establish THESIS O2 behavior. Do not infer it from the separate
direct-branch helper. Phase 2 Step 12 remains incomplete until the standard
O2 lifecycle and matching checkpoints are resolved and captured.

## Final review

A fresh review found no Critical or Important issues. The path map, proposed
tree, full-suite result, seeded case totals, and O2 blocker agree with their
artifacts.

Minor deferred: the online-adaptation training snapshot key is `online/A2`.
It could include the THESIS/OnlineAdaptationModel scope to make its ownership
clear. This naming change does not affect the captured values.

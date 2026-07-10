---
date: 2026-07-10T22:41:39+0700
researcher: Codex
git_commit: c5341061779cfc00b6e2fc3b88fd09a754b2d2e7
branch: dev
repository: bachelor-thesis-2026
topic: "Is the current codebase ready to run the complete full-spec-v2 experiment?"
tags: [research, time-series, anomaly-detection, full-spec-v2, readiness]
status: complete
last_updated: 2026-07-10
last_updated_by: Codex
---

# Research: `full-spec-v2` experiment readiness

**Date**: 2026-07-10T22:41:39+0700  
**Researcher**: Codex  
**Git Commit**: `c5341061779cfc00b6e2fc3b88fd09a754b2d2e7`  
**Branch**: `dev`

## Research Question

Is the current codebase ready to run the complete experiment specified by
`documents/spec/full-spec-v2.md`,
`documents/logs/07-09-2026/detail/detail-thesis-first-full-spec-v2-offline-online-benchmark-demo.md`,
and
`documents/logs/07-10-2026/detail/detail-full-spec-v2-gap-remediation.md`?

## Summary

No. The repository has a complete-looking experiment matrix, active entrypoints,
local smoke artifacts, and a green active test collection. However, the active
THESIS online runtime still differs from the locked `full-spec-v2` computation
in several result-changing ways. The main online configs also stop after 200
online windows, so they do not process the complete test stream. GPU-server and
readability acceptance gates remain explicitly unchecked.

The matrix preflight reports 18 THESIS offline configs, 54 THESIS online
configs, 9 RedLamp configs, 27 traditional offline configs, and 81 online
baseline configs. This proves structural coverage only. The preflight validates
file counts, epochs, checkpoint-path suffixes, and protocol fields; it does not
validate the online loss equations or event-order semantics.

## Detailed Findings

### Data Preparation

- The active benchmark uses SMD entities `machine-1-6`, `machine-3-4`, and
  `machine-3-9`, seeds `6`, `8`, and `36`, and window length 20.
- Offline THESIS main configs use 30 epochs split into Stage A 25 epochs and
  Stage B 5 epochs.
- Clean-validation threshold sources and label-use policy are represented in
  the shared protocol and preflight report.
- The current online threshold collector uses one stride-1 score collection for
  both online calibration and the `offline_point_threshold` value. Therefore,
  the THESIS online artifact builder does not independently execute the
  non-overlapping offline calibration rule stated in `full-spec-v2`.

### Modeling and Training

- The offline model exposes the required hidden, reconstruction,
  classification, continuous-memory, and discrete-codebook surfaces.
- The public THESIS model lifecycle is still distributed through setup, state,
  routing, and loss mixins. This directly conflicts with the current
  no-lifecycle-mixin policy in `codebase_preferences.md`.
- The active online forward encodes `batch["view_a"]` and `batch["view_b"]`.
  The locked specification requires one input window, with `Z_online =
  Z_source`. Main configs set view noise to zero, but smoke configs set it to
  `0.01`; the runtime contract itself still requires two views.
- The adapter computes a prototype-distance latent value internally, but the
  public online output writes reconstruction `window_scores` into
  `aux["latent_window_score"]`. Triage and calibration read this public field,
  so the active latent threshold is not using the computed memory distance.
- A2 receives every gray-zone event through the ordinary event update after
  buffer admission. `_run_online_variant_update()` treats gray-zone as an A2
  update and applies reconstruction plus contrastive loss before the window has
  passed buffer verification. The specification says gray-zone windows enter
  the verification buffer and adapt only after PNN verification.
- A2 hard-old events currently receive reconstruction hinge only because the
  contrastive branch is restricted to `gray_zone` and `pnn_candidate`. The
  specification requires hard-old reconstruction plus contrastive
  regularization.
- The function named `compute_token_multi_positive_info_nce()` implements a
  same-position cross-entropy matrix. It does not receive recurrent-signature
  groups, anomalous discrete codewords, known-anomaly tokens, or the PNN mask.
  Therefore the locked A2 positive, negative, and ignored-token sets are not
  implemented.
- A0 disables optimizer updates, but inference still passes through the
  near-identity projector. The locked A0 definition says to use the source
  model only.

### Online Execution and Reporting

- Verification-buffer capacity, per-entry TTL, non-overlap admission, anomaly
  radius filtering, recurrent-signature filtering, and checkpoint state
  helpers exist and have focused tests.
- Every generated THESIS `__main.yaml` online config contains
  `max_online_steps: 200`. The generator intentionally writes 200 for main
  runs and 16 for smoke runs. Consequently a nominal main run is a truncated
  online evaluation rather than a complete test-stream experiment.
- Local smoke artifacts exist for the four minimum combinations O0-A0, O0-A2,
  O1-A0, and O1-A2 on `machine-1-6`, seed 6.
- No CUDA device is available in the audited environment. The gap-remediation
  checklist still leaves GPU smoke execution and artifact-path verification
  after server resume unchecked.

### Evaluation and Demo

- The active test collection contains 390 tests, and the detail checklist
  records a green active suite after legacy tests were archived.
- The demo can build static offline/online replay images and has a queue-based
  callback helper. `demo/app.py` is still a command-line image exporter; it
  does not implement the specified Streamlit-style live controls, speed slider,
  selected-channel control, or continuously updated live plot.
- Label isolation is tested for the live callback, but the complete visual demo
  acceptance surface is not present.

### Readability and Deployment Gates

- The latest AST audit records 12 source files above 500 lines and 74 callables
  above 50 lines.
- `detail-full-spec-v2-gap-remediation.md` remains
  `status: implementation_in_progress` and explicitly leaves four acceptance
  items unchecked: two GPU/server checks and two AST checks.
- The local preflight result is `status: ready`, but this status describes
  matrix/config coverage. It is not evidence that the locked THESIS A2 runtime
  semantics or full-stream execution are correct.

## Code References

- `documents/spec/full-spec-v2.md:819` - one-window online forward contract.
- `documents/spec/full-spec-v2.md:914` - gray-zone buffer-only action.
- `documents/spec/full-spec-v2.md:1002` - hard-old contrastive contract.
- `documents/spec/full-spec-v2.md:1138` - PNN multi-positive and negative sets.
- `src/models/online_adaptation.py:435` - active two-view online forward.
- `src/models/online_adaptation.py:481` - latent score overwritten by window reconstruction score.
- `src/engine/online_tta/online_engine.py:489` - gray-zone admission followed by ordinary event dispatch.
- `src/engine/online_tta/online_engine.py:833` - current A2 branch behavior.
- `src/engine/online_tta/online_losses.py:73` - current same-position contrastive implementation.
- `scripts/generate_online_benchmark_configs.py:112` - 200-step main-run limit.
- `scripts/preflight_full_benchmark_matrix.py:57` - structural matrix checks.
- `demo/app.py:21` - static replay-image entrypoint.
- `documents/logs/07-10-2026/detail/detail-full-spec-v2-gap-remediation.md:889` - remaining GPU and AST gates.

## Pipeline Documentation

The implemented high-level path is:

```text
SMD train/validation/test
  -> train-only scaling
  -> L=20 windows
  -> O0/O1 two-stage offline model
  -> Stage-B checkpoint
  -> clean-validation threshold artifact
  -> online window batcher with view_a/view_b
  -> projector scoring and triage
  -> hard-old or verification-buffer update
  -> point-score EWMA and metrics
```

The intended locked online segment instead requires:

```text
one window
  -> one frozen source latent
  -> projector
  -> exact triage
  -> hard-old update OR gray-zone buffer verification
  -> A2 contrastive keys including anomalous codewords
  -> complete test stream
```

Those two online segments are not yet equivalent.

## Historical Context

The July 9 detail plan established THESIS-first implementation and a full
offline/online/baseline matrix. The July 10 gap-remediation detail closed many
helper and integration gaps, but it remains explicitly in progress. Its checked
items accurately show substantial implementation progress; they do not override
the runtime mismatches traced above.

## Open Questions

There is no open question affecting the binary readiness conclusion. The
current code and locked specification are sufficiently explicit to conclude
that the complete experiment should not yet be treated as specification-valid.

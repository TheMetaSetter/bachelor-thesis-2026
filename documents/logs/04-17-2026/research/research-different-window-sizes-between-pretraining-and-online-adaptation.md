---
date: 2026-04-17 16:44:37 +0700
researcher: TheMetaSetter
git_commit: ce9c92c9c052f39818c1186016886d1c9d0b12dd
branch: dev
repository: bachelor-thesis-2026
topic: "Whether the current codebase supports different input window sizes between pre-training and online adaptation for the multi-task prototypical autoencoder"
tags: [research, time-series, anomaly-detection, windowing, online-adaptation, multitask]
status: complete
last_updated: 2026-04-17
last_updated_by: TheMetaSetter
---

# Research: Whether the current codebase supports different input window sizes between pre-training and online adaptation for the multi-task prototypical autoencoder

**Date**: 2026-04-17 16:44:37 +0700
**Researcher**: TheMetaSetter
**Git Commit**: ce9c92c9c052f39818c1186016886d1c9d0b12dd
**Branch**: dev

## Research Question

With the current status of the codebase, can the main multi-task prototypical autoencoder use one non-overlapping window length during pre-training, such as 100 timesteps, and a different non-overlapping window length during online adaptation, such as 10, 15, or 20 timesteps?

## Summary

The answer must be split into two layers.

At the architectural level, the current offline `ThesisMultitaskModel` is length-agnostic with respect to the time dimension. Its encoder, prototype branches, fusion logic, reconstruction head, and classification head all operate on tensors of shape `[B, L, D]` and do not hard-code `L = 100` into learned layer shapes. The data windowing and online stream code are also configuration-driven and accept arbitrary positive `window_size` values, provided that `stride <= window_size`.

At the current repository-runtime level, however, the intended thesis path is still documented around `L = 100` for the main model, and the first online adaptation implementation is not presently dependable enough to claim this workflow is usable end to end. In the current environment, `OnlineAdaptationModel` fails during construction because it deep-copies a loaded `ThesisMultitaskModel` that contains a `torch.Generator` inside the synthetic anomaly injector state. Therefore, the repository does not currently provide a reliable, tested end-to-end path for pre-training at 100 timesteps and then running online adaptation at 10, 15, or 20 timesteps.

## Detailed Findings

### Data Preparation

- The design documents consistently describe the offline thesis path around windows of length `L = 100`. This appears explicitly in `documents/design/idea.md`, which defines the thesis input contract as `X in R^{B x L x D}` with `L = 100`.
- The design starter also repeats this assumption for the canonical offline windowing story, describing window construction as turning full sequences into windows of length `100`.
- The implemented windowing code is more flexible than that prose. `src/data/window.py` accepts `window_size` and `stride` as arguments and slices any valid window length. It returns an empty list only when `window_size` exceeds the sequence length.
- The implemented online stream is likewise configuration-driven. `src/data/stream.py` accepts `window_size` and `stride`, builds index records from those values, and emits windows whose `meta["window_size"]` matches the configured online value.
- Configuration validation in `src/core/config.py` requires `window_size` to be a positive integer and only enforces `stride <= window_size`. It does not require the online `window_size` to match the window size used by the offline checkpoint.

### Modeling and Training

- The main encoder in `src/models/thesis_multitask.py` applies `nn.Linear(input_dim, encoder_dim)` and `nn.Linear(encoder_dim, hidden_dim)` on the feature dimension, not on a flattened `[L * D]` vector. That means the learned weights depend on `D`, `encoder_dim`, and `hidden_dim`, but not on `L`.
- The forward path of `ThesisMultitaskModel` remains shape-generic over the time dimension. It encodes `batch["x"]`, runs prototype lookup token by token, computes `MeanPool(dim=1)` for classification, reconstructs back to `[B, L, D]`, and computes `window_scores` as the mean of `point_scores` across `dim=1`.
- The clean-batch preparation logic in the same file reads `window_size` dynamically from `prepared_batch["x"].shape[1]`, which is another sign that the offline model itself does not assume a fixed temporal length during forward or loss computation.
- A direct local runtime probe confirmed that the offline `ThesisMultitaskModel` can process a batch with shape `[2, 15, 38]` and returns `hidden` of shape `[2, 15, 32]`, `recon` of shape `[2, 15, 38]`, and `window_scores` of shape `[2]`.
- The online model is designed to reuse the offline encoder geometry from a multitask checkpoint and then score the projected online hidden states through the frozen offline heads. This design also does not insert an explicit equality check between checkpoint pre-training window size and online window size.
- However, the online runtime is currently blocked by a separate implementation issue. `src/models/online_adaptation.py` deep-copies the loaded multitask model inside `ThesisMultitaskEncoderAdapter`. In the current environment, that deep copy fails because the offline model includes a `torch.Generator` through the synthetic anomaly injector state.
- A direct local runtime probe and the repository test `tests/test_online_adaptation_step.py` both fail at online-model construction with `TypeError: cannot pickle 'torch._C.Generator' object`. This failure happens before any practical comparison between 100-step and 10/15/20-step online windows can be exercised in the accepted online path.

### Evaluation

- The design documents for the first online slice define the online batch contract as `x`, `view_a`, and `view_b` all sharing the same shape `[B, L, D]`. This requires internal agreement within one online batch, but it does not impose equality between online `L` and offline pre-training `L`.
- The implementation follows that contract. `src/core/contracts.py` validates that `view_a.shape == x.shape` and `view_b.shape == x.shape`, but it does not compare those shapes against any reference-checkpoint metadata.
- The online entrypoint constructs the stream with `experiment_config["data"]["window_size"]` and passes those windows directly into the online batcher and model. There is no logic that reads the offline checkpoint config and rejects a different online window length.
- Therefore, the repository currently lacks both an explicit safeguard and a passing integration test for "pre-train with 100, adapt online with 10/15/20".

## Code References

- `documents/design/idea.md:6` - thesis design states windows of length `L = 100`
- `documents/design/idea.md:65` - online adaptation is framed as operating on incoming windows under the same `[B, L, D]` contract
- `documents/design/design_starter.md:172` - design starter repeats window construction around length `100`
- `documents/design/design_starter.md:676` - native window contract uses generic `[L, D]` with `window_size` in metadata
- `documents/logs/04-02-2026/plan/plan-phase-4-online-adaptation-implementation.md:160` - online batch contract uses `x`, `view_a`, and `view_b` with the same `[B, L, D]` shape
- `src/data/window.py:16` - offline window slicing is parameterized by `window_size` and `stride`
- `src/data/stream.py:36` - online stream is parameterized by `window_size` and `stride`
- `src/core/contracts.py:92` - online batch validation enforces only intra-batch shape equality
- `src/core/config.py:91` - config validation requires positive `window_size` and `stride`
- `src/core/config.py:181` - config validation only enforces `stride <= window_size`
- `src/models/thesis_multitask.py:30` - encoder layers depend on `input_dim`, not on fixed `window_size`
- `src/models/thesis_multitask.py:492` - clean-batch preparation reads `window_size` dynamically from tensor shape
- `src/models/thesis_multitask.py:532` - forward path is generic over `L`
- `src/models/online_adaptation.py:31` - online encoder adapter deep-copies the offline multitask model
- `src/models/online_adaptation.py:182` - online model loads multitask checkpoint config but does not compare window sizes
- `src/models/online_adaptation.py:311` - online forward uses generic `[B, L, D]` online batches
- `scripts/run_online_adaptation.py:130` - online stream window size comes from the online experiment config
- `tests/test_online_adaptation_step.py:99` - current online-step test is the intended proof surface for online runtime behavior

## Pipeline Documentation

The implemented repository behavior relevant to this question is:

```text
offline data config
-> choose one window_size for offline loaders
-> train ThesisMultitaskModel on [B, L_pre, D]
-> save multitask checkpoint
-> online experiment config
-> choose one window_size for online stream
-> build online windows [B, L_online, D]
-> validate only that x, view_a, and view_b agree with one another
-> load multitask checkpoint into OnlineAdaptationModel
-> attempt projector-first online adaptation
```

The important current-state distinction is:

- `L_pre` and `L_online` are not tied together by config validation or by an explicit runtime assertion.
- The offline model code can operate on different `L` values.
- The accepted online runtime currently fails earlier for a separate model-construction reason, so this workflow is not yet dependable in practice.

## Historical Context (from documents/)

The design documents are slightly stricter than the implementation in their thesis-facing prose. `documents/design/idea.md` and `documents/design/design_starter.md` present the main thesis model around `L = 100` as the canonical offline setting. That makes sense as the intended default and as the language used throughout the repository design.

At the same time, the lower-level contract documents and the implemented code speak in generic `[B, L, D]` terms. The stream-design plan for the first online slice requires consistent shapes within each online batch, but it does not define a formal rule that online `L` must equal offline `L`. The implementation follows this more generic contract.

This means the repository currently sits in an in-between state:

- design intent still treats `100` as the main thesis window length;
- implementation is mostly length-generic;
- end-to-end online adaptation is not stable enough yet to treat mixed pre-train and online window lengths as an accepted supported workflow.

## Open Questions

- Should the repository formally support `L_pre != L_online`, or should config validation reject it to preserve the thesis-facing `L = 100` contract?
- If mixed window lengths are intended, should there be an explicit integration test for `100 -> 10`, `100 -> 15`, and `100 -> 20` in the online path?
- The current online runtime fails before this question can be exercised end to end because `OnlineAdaptationModel` deep-copies a multitask model that contains a non-pickleable `torch.Generator`.

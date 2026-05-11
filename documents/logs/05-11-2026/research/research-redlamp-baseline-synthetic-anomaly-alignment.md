---
date: 2026-05-11 13:33:33 +07 +0700
researcher: TheMetaSetter
git_commit: 0420abc3fb938056386cadba556d2c681f77296f
branch: dev
repository: bachelor-thesis-2026
topic: "RedLamp baseline synthetic anomaly and MLP fairness alignment"
tags: [research, time-series, anomaly-detection, multi-class, redlamp, candi]
status: complete
last_updated: 2026-05-11
last_updated_by: TheMetaSetter
---

# Research: RedLamp Baseline Synthetic Anomaly and MLP Fairness Alignment

**Date**: 2026-05-11 13:33:33 +07 +0700
**Researcher**: TheMetaSetter
**Git Commit**: 0420abc3fb938056386cadba556d2c681f77296f
**Branch**: dev

## Research Question

The user wants to run the RedLamp baseline to obtain baseline statistics for the proposed method in this repository. The user also wants the proposed method and `bsc-thesis-ref-codebases/RedLamp` to inject exactly the same classes of synthetic anomalies, to have classification heads output softmax probabilities over the same synthetic anomaly classes, to use the same number of MLP layers for fairness, and to use a small input window size similar to `bsc-thesis-ref-codebases/CANDI-main`.

## Summary

The active repository already contains the RedLamp anomaly taxonomy as the default synthetic anomaly family list in `src/data/augment.py`. The list contains eleven synthetic anomaly families: `spike`, `flip`, `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`, `upsidedown`, and `mixture`. This matches the anomaly families dispatched by RedLamp, excluding RedLamp's `normal` class.

The active repository does not currently train the thesis classification head on the same multi-class label space as RedLamp. The active thesis method uses the RedLamp family names to choose corruption mechanisms, but `SyntheticAnomalyInjector.augment_batch` writes binary `classification_labels`: `0` for clean windows and `1` for synthetic anomalous windows. The default thesis model config also sets `num_classes: 2`. RedLamp, in contrast, builds a label dictionary from `anomaly_types`, produces one-hot labels, and its classifier ends with `Softmax(dim=1)` over `len(args.anomaly_types)`. With RedLamp's default argument, this is a twelve-class space: `normal` plus the eleven synthetic anomaly classes.

For MLP fairness, the active thesis model exposes a shared `mlp_num_linear_layers` setting and the default config sets it to `3`. CANDI's MLP reference uses three linear layers in the encoder and three linear layers in the decoder. The active repository also has a window-10 SMD configuration that matches CANDI's small default window length.

## Detailed Findings

### Data Preparation

The active repository uses a standardized batch contract in which windows are shaped as `batch["x"]` with shape `[B, L, D]`. The thesis synthetic anomaly injector validates this three-dimensional shape before augmentation and then returns an augmented batch with added multitask fields.

The active default RedLamp family taxonomy is defined in `src/data/augment.py` as:

- `spike`
- `flip`
- `speedup`
- `noise`
- `cutoff`
- `average`
- `scale`
- `wander`
- `contextual`
- `upsidedown`
- `mixture`

The default task config `configs/task/multitask_tsad.yaml` uses the same eleven families. The active window-10 task config `configs/task/multitask_tsad_window10_binary.yaml` is intentionally smaller, using only `spike`, `noise`, `cutoff`, `scale`, and `contextual`.

RedLamp's `Loader_aug.select_anomalies` dispatches `normal` plus the same eleven synthetic anomaly families. It also has a `random` option that samples one of the eleven non-normal families. RedLamp creates `Y` as the injected window, `Z` as the original window, `anomaly_mask` as a cell-level mask, and `label` as a one-hot anomaly-type vector.

Important deviation: the current thesis injector records the selected family in `augmentation_metadata["anomaly_family"]` and `augmentation_metadata["anomaly_family_index"]`, but it does not use that family index as the classification label. Instead, every injected window receives class `1`, regardless of whether the injected family was `spike`, `flip`, or another RedLamp family.

### Modeling and Training

The thesis model is implemented in `src/models/thesis_multitask.py`. It builds an MLP encoder with the helper `build_multilayer_perceptron`, then computes continuous prototype context, discrete codebook context, task-specific fusion, reconstruction, and classification logits. The classification head consumes the mean-pooled classification fusion representation and returns logits.

The default thesis model config sets:

- `mlp_num_linear_layers: 3`
- `num_classes: 2`
- `input_dim: 38`
- `encoder_dim: 64`
- `hidden_dim: 32`

The training loss uses `F.cross_entropy(outputs["logits"], batch["classification_labels"].long())` unless binary label refurbishment is enabled. Label refurbishment explicitly requires `num_classes == 2`. Therefore, the current implementation is structurally binary by default, even though its anomaly corruption families follow RedLamp naming.

RedLamp's baseline model is `ConvAEC`, not an MLP. It uses a convolutional encoder, convolutional decoder, and `NonLinClassifier`. The classifier has two linear layers, optional normalization, ReLU, dropout, and a final `Softmax(dim=1)`. RedLamp computes `classes = len(args.anomaly_types)`, so the classifier output dimension is controlled directly by the anomaly type list. With the default CLI value, this is twelve classes.

There is a notable implementation detail in RedLamp: `MetaAEC.calculate_loss` applies `nn.CrossEntropyLoss(reduction="none")` to `pred_label`, where `pred_label` is already the output of a `Softmax`. In standard PyTorch usage, cross entropy normally expects unnormalized logits. This research note documents the current behavior only; it does not change or reinterpret RedLamp.

### Evaluation

The active thesis model reports reconstruction scores from mean squared error over reconstructed windows and reports classification metrics from logits and `classification_labels`. Current metrics are binary-oriented: `src/metrics/pointwise.py` computes binary classification metrics by applying `torch.softmax(logits, dim=-1)[:, 1]`.

RedLamp's test path collects reconstructed windows, anomaly masks, one-hot labels, predicted softmax labels, and latent encodings. It also combines reconstruction error with classifier-derived anomaly score components in `anomaly_scoreing`.

For baseline statistics, the current repository can run binary synthetic-anomaly classification statistics for the proposed method, but it cannot yet produce RedLamp-equivalent multi-class anomaly-type classification statistics without changing the thesis label construction, metric aggregation, and default `num_classes`.

## Code References

- `src/data/augment.py:21` defines `REDLAMP_ANOMALY_FAMILIES` with the eleven RedLamp synthetic anomaly families.
- `src/data/augment.py:81` registers the active family-to-injection-function dispatch table.
- `src/data/augment.py:637` samples `anomaly_family_index`, and `src/data/augment.py:666` stores it in metadata.
- `src/data/augment.py:723` creates binary `classification_labels`, and `src/data/augment.py:743` assigns all injected windows to class `1`.
- `src/models/thesis_multitask.py:32` defines the shared MLP builder and `src/models/thesis_multitask.py:50` derives the number of linear layers from `num_linear_layers`.
- `src/models/thesis_multitask.py:1434` mean-pools the classification representation and `src/models/thesis_multitask.py:1435` passes it to `classification_head`.
- `configs/model/thesis_multitask.yaml:5` sets `mlp_num_linear_layers: 3`, and `configs/model/thesis_multitask.yaml:6` sets `num_classes: 2`.
- `configs/task/multitask_tsad.yaml:12` lists the eleven default RedLamp anomaly families.
- `bsc-thesis-ref-codebases/RedLamp/loaders/loader_aug.py:191` dispatches `normal` plus the eleven synthetic anomaly families.
- `bsc-thesis-ref-codebases/RedLamp/loaders/loader_aug.py:920` builds the anomaly dictionary, and `bsc-thesis-ref-codebases/RedLamp/loaders/loader_aug.py:929` creates one-hot labels.
- `bsc-thesis-ref-codebases/RedLamp/models/classifier.py:25` sets the final classifier linear layer to `n_class`, and `bsc-thesis-ref-codebases/RedLamp/models/classifier.py:26` applies `Softmax(dim=1)`.
- `bsc-thesis-ref-codebases/CANDI-main/config.py:25` sets the CANDI default window length to `WIN_SIZE = 10`.
- `bsc-thesis-ref-codebases/CANDI-main/models/mlp/modeling_mlp.py:87` defines an encoder with three linear layers.
- `bsc-thesis-ref-codebases/CANDI-main/models/mlp/modeling_mlp.py:107` defines a decoder with three linear layers.
- `configs/data/smd_rtx3090_machine_2_1_10.yaml:4` sets `window_size: 10`, and `configs/data/smd_rtx3090_machine_2_1_10.yaml:5` sets `stride: 10`.

## Pipeline Documentation

The active thesis pipeline takes clean SMD windows with shape `[B, L, D]`, optionally injects synthetic anomalies in the offline multitask path, and returns the same batch with `classification_labels`, `synthetic_anomaly_mask`, and `augmentation_metadata`. The injector chooses one anomaly family for each injected window and corrupts a contiguous subsequence across selected channels. The training model reconstructs the augmented input and classifies the window from the fused prototype representation.

The RedLamp pipeline constructs windows in `[batch, n_features, window_size]`, injects each requested anomaly type as a separate set of windows, and returns one-hot labels over the anomaly type dictionary. RedLamp then transposes inputs to `[batch, window, n_features]` before feeding them to `ConvAEC`.

CANDI's MLP pipeline uses window length ten by default and flattens `[B, L, C]` into `[B, L * C]`. Its encoder has three linear layers and its decoder has three linear layers. The active thesis MLP encoder and task heads can be configured with the same linear-layer count through `mlp_num_linear_layers: 3`.

## Historical Context (from documents/)

`documents/design/idea.md` states that the thesis design originally used windows of length one hundred and planned to inject artificial anomalies for anomaly-type classification, with the repository default using the eleven RedLamp anomaly types while retaining CARLA as a subsequence-oriented mechanism reference. It also states that the default prediction paths should be reconstruction from the reconstruction fusion representation and classification from the classification fusion representation.

`documents/design/design_starter.md` defines the standard batch contract as `batch["x"]: Tensor[B, L, D]`, the model output contract including `hidden`, `recon`, `logits`, `point_scores`, and `window_scores`, and the principle that model-specific losses should remain inside the model file.

## Open Questions

- Should the aligned multi-class label space be twelve classes (`normal` plus eleven synthetic anomaly classes), matching RedLamp's default `anomaly_types`, or eleven classes over synthetic anomaly families only with clean windows excluded from the classification loss?
- Should the RedLamp baseline be run unchanged as `ConvAEC`, or should a separate RedLamp-style MLP baseline be created for the user's fairness condition about MLP layer count?
- Should `label_refurbishment` be disabled or generalized before moving the thesis model from binary classification to multi-class anomaly-type classification?
- Should window length ten become the default for the RedLamp comparison experiments, or should it remain an explicit experiment config such as `smd_rtx3090_machine_2_1_10.yaml`?

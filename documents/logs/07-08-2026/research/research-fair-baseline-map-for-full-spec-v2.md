---
date: 2026-07-08 22:58:52 +07
researcher: TheMetaSetter
git_commit: dd34f141eb6b82911a45483e55061dd3c4a105d1
branch: dev
repository: bachelor-thesis-2026
topic: "Fair baseline map for full-spec-v2 on SMD machine-1-6, machine-3-4, and machine-3-9"
tags: [research, time-series, anomaly-detection, baseline, fairness, smd, redlamp, candi, m2n2]
status: complete
last_updated: 2026-07-08
last_updated_by: Artificial Intelligence Agent
---

# Research: Fair baseline map for full-spec-v2

**Date**: 2026-07-08 22:58:52 +07
**Researcher**: TheMetaSetter
**Git Commit**: dd34f141eb6b82911a45483e55061dd3c4a105d1
**Branch**: dev

## Research Question

The target experiment set is SMD `machine-1-6`, `machine-3-4`, and `machine-3-9`.

For offline two-stage pre-training, RedLamp is the main baseline. Three additional traditional machine-learning baselines are required. The original MatrixProfile reference is replaced by a direct STUMPY implementation following `documents/notes/stumpy_tsad_fairness_guide.md`; KMeansAD and IForest remain sourced from CANDI/TSB-AD behavior.

For online test-time adaptation, CANDI and M2N2 are the two main baselines.

The implementation plan for `documents/spec/full-spec-v2.md` must maximize fairness between all baselines and the main method, named THESIS.

## Summary

The current repository already supports the three target SMD entities for RedLamp baseline and THESIS offline benchmark runs. The existing benchmark config family includes RedLamp baseline, THESIS two-stage base, and THESIS two-stage point-score variants for `machine-1-6`, `machine-3-4`, and `machine-3-9`, each with seeds `6`, `8`, and `36`.

The current mixed-method launcher only accepts `redlamp_baseline` as a baseline model. MatrixProfile, KMeansAD, IForest, CANDI, and M2N2 are present only as reference-code implementations under `bsc-thesis-ref-codebases/`, not as first-class experiment configs in the repo runtime.

The main fairness issue is not only model architecture. It is protocol control. Each reference codebase has its own windowing, normalization, thresholding, score aggregation, and online-update semantics. For fair comparison against THESIS, the future plan should keep one shared protocol layer for data splits, scaling, threshold calibration, metrics, run metadata, and output files, while wrapping baseline algorithms behind small adapters.

Follow-up on 2026-07-08:

The main THESIS epoch budget is now locked to `30` total epochs: `25` epochs for `stage_a_multitask_pretraining` and `5` epochs for `stage_b_fusion_finetuning`. A1 should be implemented fully and run, not left as an optional code-only variant. The demo should wait until benchmark scripts and artifacts are stable.

Second follow-up on 2026-07-08:

The official clean validation, synthetic validation, and test windowing protocol is now locked to non-overlapping windows with one tail exception. If a leftover final segment is shorter than the window size, one final full window may overlap with the previous window to cover that tail. Points covered by both windows average their anomaly scores. The offline anomaly threshold is a point-level threshold inferred from clean validation only. Before online TTA, the online point-level threshold must be recalibrated by simulating sliding windows with stride `1` on clean validation, computing anomaly scores, applying EWMA, and selecting the clean-validation threshold from that online score stream.

Implementation readability is also a protocol requirement. New code should be highly readable for a high-school student, include small ASCII diagrams that show how a component fits the larger pipeline, and may use cute kaomoji markers such as `₍^. .^₎⟆` or `( ˶˘ ³˘)♡` sparingly in comments to make educational code easier to scan.

## Detailed Findings

### Data Preparation

The shared SMD parser reads train, test, and test-label files, then splits the original train sequence into clean train and clean validation using `validation_split_ratio`. Train and validation labels are zeros; test labels come from SMD `test_label`.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/datasets/smd.py:13` - `SMDDatasetParser`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/datasets/smd.py:100` - train/validation/test sequence construction

The active benchmark entity configs are single-entity runs with `window_size: 20`, `train_stride: 1`, `val_stride: 20`, and `test_stride: 20`.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/data/smd_benchmark_machine_1_6_window20.yaml:1`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/data/smd_benchmark_machine_3_4_window20.yaml:1`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/data/smd_benchmark_machine_3_9_window20.yaml:1`

The scaler fits only on train sequences, then transforms train, validation, and test before windowing. This is the correct shared normalization surface for fair baseline adapters.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/scalers.py:8` - `SequenceStandardScaler`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/window.py:14` - `slice_sequence_into_windows`

### Synthetic Anomaly Quality

The active synthetic anomaly injector owns the 12-class contract: one `normal` class and 11 RedLamp-style anomaly families. It samples one contiguous segment per anomalous window, then applies the chosen family to a subset of channels.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/augment.py:14` - `REDLAMP_ANOMALY_FAMILIES`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/augment.py:28` - `SyntheticAnomalyInjector`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/augment.py:178` - `_sample_segment_bounds`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml:11` - active balanced multiclass synthetic config

The active benchmark config uses `window_size: 20`, `min_segment_fraction: 0.2`, `max_segment_fraction: 0.3`, `spike_scale: 3.0`, and `anomaly_visibility_boost: 1.5`. Therefore most synthetic segments are short, usually 4 to 6 timesteps per anomalous window. This makes some anomaly classes visually subtle, especially classes that preserve the local mean or local shape.

User-observed quality issues to preserve in the implementation plan:

- `spike`: should be slightly stronger, but must remain distinguishable from `flip`.
- `flip`: should reverse a longer segment.
- `speedup`: currently hard to distinguish from the clean window.
- `spike` and `flip`: sometimes visually confusable.
- `cutoff`: not visually clear enough.
- `average`: not visually clear enough.
- `scale`: sometimes not visually clear enough.
- `wander`: acceptable but can be stronger.
- `contextual`: acceptable but can be stronger.
- `upsidedown`: acceptable but can be stronger.
- `mixture`: already clear enough and should not be strengthened first.

The next plan should add class-specific intensity controls instead of relying only on global `spike_scale` and `anomaly_visibility_boost`. This keeps the 12-class taxonomy stable while making weak classes more visible.

### Offline Modeling and Training

The current in-repo RedLamp baseline is RedLamp-inspired and follows the repository batch/output contract. It is not the original RedLamp CNN architecture directly copied from the reference folder. It supports MLP or simple CNN encoder families, synthetic RedLamp multiclass augmentation, reconstruction loss, classification loss, and optional gradient-conflict profiling.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/redlamp_baseline.py:86` - `RedLampBaseline`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/redlamp_baseline.py:343` - forward output contract
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/redlamp_baseline.py:611` - reconstruction/classification training loss

The original RedLamp reference uses `ConvAEC`, a strided convolutional autoencoder, multiclass pseudo-anomaly labels, optional anomaly masking, and convex loss weighting through `c_loss_ratio`.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/RedLamp/main.py:17` - RedLamp model parameters
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/RedLamp/models/meta.py:31` - `MetaAEC.forward`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/RedLamp/models/meta.py:41` - `calculate_loss`

The THESIS public model is a small shell class whose implementation is split across mixins. The current active two-stage runtime is driven by stage names `stage_a_multitask_pretraining` and `stage_b_fusion_finetuning`.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:37` - `ThesisMultitaskModel`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_routing_mixin.py:471` - forward pass
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_loss_mixin.py:310` - point-wise score loss implementation
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_routing_mixin.py:346` - two-view contrastive loss

The two-stage runner materializes Stage A and Stage B YAMLs, trains Stage A, initializes memories for Stage B from the train loader, then trains Stage B and evaluates the final checkpoint.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_two_stage_offline_pretraining.py:23` - canonical stage names
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_two_stage_offline_pretraining.py:146` - manifest materialization
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_two_stage_offline_pretraining.py:227` - Stage B initialization from Stage A checkpoint

### Traditional Machine-Learning Baselines

The old MatrixProfile reference should not be used as the main baseline. It is a short wrapper around `stumpy.stump` and uses `X.ravel()`, so the reference implementation is naturally univariate and loses explicit channel semantics. The STUMPY fairness guide instead recommends a direct per-channel AB-join implementation with train-normal reference sequences.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/TSB-AD/TSB_AD/models/MatrixProfile.py:5` - `MatrixProfile`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/notes/stumpy_tsad_fairness_guide.md:1` - STUMPY fairness guide

The main fair STUMPY baseline should be `STUMPY-ChannelAB-FrozenTrainRef`. It scores each channel independently against train-normal reference subsequences, calibrates channel scores on clean validation, aggregates channels by robust normalized maximum, then exports point scores under the same offline or online point-score rule as THESIS.

KMeansAD is present as a sliding-window KMeans detector. It preprocesses data into sliding windows, optionally z-scores each window, predicts cluster-center distances, then reverse-window maps scores back to the point timeline.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/TSB-AD/TSB_AD/models/KMeansAD.py:13` - `KMeansAD`

IForest is present as a PyOD-style wrapper around scikit-learn `IsolationForest`. It converts the time series into windowed matrix format, optionally normalizes windows, inverts the sklearn decision function so larger means more anomalous, and pads window scores back to point length.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/TSB-AD/TSB_AD/models/IForest.py:25` - `IForest`

None of these three traditional baselines is currently registered in the repository experiment runtime. The current mixed runner supports only `redlamp_baseline` as a baseline model.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_comparative_smd_experiments.py:54` - `SUPPORTED_BASELINE_MODEL_NAMES = {"redlamp_baseline"}`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_comparative_smd_experiments.py:218` - run-family routing

### Online Adaptation Baselines

CANDI reference code computes train, validation, and test scores, calibrates thresholds from validation scores when threshold type is `ratio`, and then optionally adapts during test. The CANDI adapter selects hard and moderate samples using validation score strata and latent Mahalanobis similarity, then adapts SANA residual modules or the model depending on config.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/predictor.py:22` - `Predictor`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/threshold.py:5` - `Thresholder`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:133` - `CANDIAdapter`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/tta/candi/adapter_candi.py:353` - `MLPAdapter.calculate_loss`

M2N2 reference code computes offline scores, supports thresholding by train-score quantiles or oracle `off_f1_best`, and adapts online by masking predicted anomalies and updating the model on predicted-normal reconstruction loss. Its Detrender updates running mean statistics.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/M2N2-master/Exp/Tester.py:45` - `prepare_stats`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/M2N2-master/Exp/MLP.py:166` - online adaptation loop
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/M2N2-master/models/Normalizer.py:5` - `Detrender`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/tta/m2n2/adapter_m2n2.py:13` - CANDI repo's M2N2 adapter variant

The current THESIS online implementation is an earlier conservative slice. It uses a frozen reference encoder, frozen online encoder, and trainable residual projector. It optimizes alignment, optional prototype alignment, and anchor drift. It does not yet implement `full-spec-v2` A0/A1/A2 triage, EWMA point thresholding, hard-old adaptation, PNN verification buffer, TTL buffer, or online contrastive negatives from anomalous codewords.

Code references:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/online_adaptation.py:136` - `OnlineAdaptationModel`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/online_adaptation.py:389` - online forward pass
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_loop.py:37` - online loop
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/stream.py:37` - `SMDOnlineStream`

### Evaluation

The evaluator merges window-level `point_scores` back to full entity timelines by summing scores and counts over covered points, then averages overlapping scores before metrics.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:76` - `accumulate_pointwise_window_payload`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:126` - `reconstruct_pointwise_records_from_window_payload`

Current threshold fallback uses positive-support `q0.99` from the point scores being evaluated unless a checkpoint/provided threshold is passed. `full-spec-v2` requires clean-validation-only calibration for both offline point threshold and online EWMA threshold. Therefore, fair baseline planning should promote threshold artifacts to first-class shared outputs rather than allowing each method's evaluator to choose thresholds independently.

Code reference:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/thresholding.py:7` - positive-support quantile selection
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/thresholding.py:22` - `resolve_evaluation_threshold`

## Pipeline Documentation

Current offline benchmark flow:

```text
SMD raw files
  -> SMDDatasetParser
  -> train/val split from original train file
  -> SequenceStandardScaler fit on train only
  -> Windowizer
  -> DataLoader
  -> model.train/val/test step
  -> Evaluator timeline merge
  -> metrics and artifacts
```

Current supported offline experiment families:

```text
RedLamp baseline
  -> train.py
  -> evaluate.py

THESIS O0/O1
  -> run_two_stage_offline_pretraining.py
  -> Stage A
  -> train-only memory initialization
  -> Stage B
  -> evaluate.py
```

Required future fair-baseline flow:

```text
One shared SMD protocol
  -> same entity ids
  -> same train/val/test split
  -> same train-fit scaler state
  -> same official threshold artifacts
  -> same point-score export schema
  -> same metric implementation

Baseline algorithm adapters
  -> RedLamp
  -> STUMPY-ChannelAB-FrozenTrainRef
  -> KMeansAD
  -> IForest
  -> CANDI online TTA
  -> M2N2 online TTA
  -> THESIS O0/O1/A0/A1/A2
```

## Fairness Constraints for the Next Plan

1. Entity scope must be identical: `machine-1-6`, `machine-3-4`, `machine-3-9`.

2. Split policy must be identical: train-only fitting, clean validation-only threshold calibration, test-only final evaluation.

3. Scaling must be decided centrally. The current repo uses train-fit `SequenceStandardScaler`. Reference CANDI and M2N2 have their own normalization choices, and KMeansAD/IForest can z-score internal windows. These internal options must be disabled or documented as method-specific if they cannot be disabled without changing the algorithm.

4. Thresholding must be central. CANDI `best_f1`, M2N2 `off_f1_best`, and any test-label threshold tuning are incompatible with maximum fairness unless explicitly marked oracle-only and excluded from main comparison.

5. Traditional ML baselines must produce point scores with the same polarity: larger score means more anomalous.

6. Window-to-point aggregation must be centralized. KMeansAD and IForest already reverse-map internally, while THESIS evaluator merges window point scores. The plan must prevent double aggregation. The official clean validation, synthetic validation, and test rule is non-overlapping windows with one final tail-overlap exception. Only the final tail-covering window may overlap the previous window, and overlapped points average their anomaly scores.

7. Online baselines must share the same stream order and no-future-data rule. Any adaptation must be causal.

8. THESIS online A0 must be a real no-update baseline over the same online scoring path, not merely the offline evaluator renamed.

9. Output artifacts must be schema-compatible across methods: scores, predictions, labels, thresholds, metrics, config snapshot, and run metadata.

10. The mixed-method runner must stop hard-coding `redlamp_baseline` as the only supported baseline before adding the new baselines.

11. STUMPY, KMeansAD, and IForest should be frozen streaming scorers in the main online benchmark. Their train-time fit or train-reference state should not be updated by test stream data. This gives a leakage-safe online comparison. Any adaptive traditional variant should be reported separately.

12. STUMPY self-join variants should be added only after `STUMPY-ChannelAB-FrozenTrainRef` and the shared benchmark protocol are stable. Full-test self-join is not causally comparable to online THESIS, so it should not be part of the main online table.

13. Offline and online thresholds are distinct artifacts. Offline threshold comes from point-level clean validation scores under the official non-overlap plus tail-overlap protocol. Online threshold comes from clean validation simulated as a stride-1 stream, followed by EWMA score aggregation.

14. New implementation code should include short educational ASCII diagrams for protocol-heavy components so readers can see how local functions fit into the full benchmark flow.

## Historical Context

The prompt expected `documents/design/idea.md` and `documents/design/design_starter.md`, but this checkout does not contain a `documents/design/` directory. The equivalent files currently exist under:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/abstract-design-notes/idea.md`
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/abstract-design-notes/design_starter.md`

The active two-stage offline contract is recorded in:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/abstract-design-notes/offline_pretraining_two_stage_kmeans_memory_design.md`

The current implementation target is:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/spec/full-spec-v2.md`

## Resolved and Remaining Questions

1. The MatrixProfile open question is resolved for the main benchmark: use direct STUMPY `STUMPY-ChannelAB-FrozenTrainRef`, not the univariate `X.ravel()` wrapper.

2. CANDI and M2N2 update more than THESIS projector under their native implementations. The plan must decide whether the fairness target is native-method fairness or equal-trainable-surface fairness.

3. The epoch budget is resolved: use 30 total epochs for main THESIS runs, with 25 Stage A epochs and 5 Stage B epochs.

4. Current online repo code is a conservative alignment slice, not full A0/A1/A2. The next plan should implement the full `full-spec-v2` online engine first, including A1, then wrap CANDI/M2N2 as external baselines.

5. The repo still has no `documents/design/` directory despite instructions calling it SSOT. Either the docs path should be migrated back, or the instructions should be updated to name `documents/abstract-design-notes/` and `documents/spec/` explicitly.

6. The synthetic anomaly plan must decide class-specific visibility controls, especially for `speedup`, `cutoff`, `average`, `scale`, and the visual separation between `spike` and `flip`.

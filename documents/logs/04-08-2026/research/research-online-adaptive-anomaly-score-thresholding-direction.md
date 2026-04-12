---
date: 2026-04-08 15:16:24 +0700
researcher: TheMetaSetter
git_commit: 33f0e4ef21ad9862ee5d979ae9143084497736e4
branch: dev
repository: bachelor-thesis-2026
topic: "Research direction note for online adaptive anomaly score thresholding"
tags: [research, anomaly-detection, thresholding, online-adaptation, uncertainty]
status: complete
last_updated: 2026-04-08
last_updated_by: TheMetaSetter
---

# Research: Research direction note for online adaptive anomaly score thresholding

**Date**: 2026-04-08 15:16:24 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: 33f0e4ef21ad9862ee5d979ae9143084497736e4  
**Branch**: dev

## Research Question

Record the user's research direction exactly and preserve a faithful working understanding of it for later implementation, baseline selection, and experiment tracking.

## Summary

The intended research direction is online adaptive anomaly score thresholding. The threshold is not treated as a fixed constant. Instead, the thresholding mechanism should be learned during pre-training and then adapted during evaluation under two testing regimes: batch testing and stream testing. The threshold is intended to be context-dependent, with conditioning information coming from the current data, the history of thresholds, and the dynamics of threshold change over time.

The user also wants the threshold predictor to be probabilistic rather than deterministic. The current preference is a Gaussian predictive distribution parameterized by a mean and a standard deviation so that uncertainty in threshold selection can be quantified explicitly. The prior for this threshold distribution should come from what is learned during pre-training, while the posterior should be inferred during testing from observed test batches or streaming observations.

The project is explicitly organized into three modes: pre-training, batch testing, and stream testing. Baseline methods are needed for all three modes so that the proposed method can be compared against them through a common Weights and Biases measurement and reporting pipeline.

## Exact User Notes

The following text is preserved verbatim:

```text
I want to do research about online adaptive anomaly score thresholding, which is learning how to choose threshold for anomaly score from pre-training stage, and then, adaptive this choosing mechanism on online streaming data.

I want to do it like data-dependent threshold or we can call it more broadly context-dependent threshold, which is given the current data and history of thresholds, and threshold changing dynamics, predict the threshold for the current anomaly score. I assume the prediction should be something like a Gaussian distribution with parameters as mean and standard deviation, so I can quantify the uncertainty behind the action of choosing the threshold. I will give the threshold a prior distribution based on what was learnt from pre-training stage, from training dataset, involving train set, validation set. The posterior Gaussian distribution will be infered from samples from batch-like test set or stream test set, which are 2 main testing modes of this project.

Side note: This project has 3 modes in total: pre-training, batch testing and stream testing. So I need to find baseline methods for 3 modes, so that I can compare my methods with them after running and receiving all the measurements, all the metrics, through Weights and Biases.
```

## Working Understanding

### Core research objective

The primary goal is to learn a threshold-selection mechanism for anomaly scores rather than selecting a fixed threshold by hand. This mechanism should be initialized from offline pre-training and then adapted online when new data arrives.

### Thresholding view

The threshold is intended to be context-dependent. At minimum, the conditioning signal should include:

- the current data or current anomaly score context;
- the history of previously selected thresholds;
- the temporal dynamics of how the threshold has been changing.

### Probabilistic formulation

The current target formulation is a Gaussian predictive distribution over the threshold:

- the mean represents the central threshold estimate;
- the standard deviation represents uncertainty in the threshold decision.

This means the thresholding mechanism is expected to produce both an action and a calibrated uncertainty signal, rather than only a single scalar threshold.

### Prior and posterior roles

- The prior threshold distribution should be learned from the pre-training stage using the training and validation data.
- The posterior threshold distribution should then be inferred during evaluation from test-time evidence.
- The evaluation evidence differs by mode:
  - batch testing uses batch-like test samples;
  - stream testing uses sequential streaming observations.

### Project operating modes

The project is understood as having exactly three modes:

1. Pre-training
2. Batch testing
3. Stream testing

These modes should be treated as first-class experimental settings, not as minor variations of one another.

### Baseline requirement

Baseline methods are required for all three modes so that comparisons remain fair and complete:

- baselines for pre-training;
- baselines for batch testing;
- baselines for stream testing.

All resulting measurements, metrics, and runs should be tracked in Weights and Biases so that the proposed method and the baselines can be compared under a single experiment history.

## Immediate implications for later implementation

- The thresholding component should eventually be modeled as a learned module rather than a post hoc static threshold.
- The module should expose uncertainty-aware outputs.
- The experimental pipeline should separate offline learning from online adaptation.
- Evaluation design should explicitly distinguish batch adaptation from stream adaptation.
- Baseline selection is part of the core thesis workflow, not a secondary reporting task.

## Open points preserved for future research

- The exact conditioning features for the context-dependent threshold are not yet fixed.
- The exact update rule for converting the pre-training prior into a test-time posterior is not yet fixed.
- The precise definition of the baseline families for each of the three modes is not yet fixed.
- The calibration and evaluation criteria for threshold uncertainty are not yet fixed.

## Follow-up Note at 2026-04-08 15:18:42 +0700

### Exact follow-up text

The following text is preserved verbatim:

```text
Maybe, just maybe, we can use classical machine learning methods to predict (or more precisely say, forecast) the next threshold, given the current data and history thresholds, threshold dynamics, as context, contextual information.
```

### Follow-up understanding

This follow-up adds a concrete modeling possibility: the threshold forecaster does not necessarily need to be a deep neural module in every setting. Classical machine learning methods may also be valid candidates for predicting the next threshold from contextual information.

The contextual information named here remains consistent with the earlier note:

- current data;
- threshold history;
- threshold dynamics over time.

This introduces an important practical implication for the project design:

- classical machine learning threshold forecasters should be considered as candidate baselines;
- they may also serve as lightweight main-model variants, especially for batch testing or conservative online adaptation settings;
- the problem can be framed explicitly as next-threshold forecasting conditioned on contextual features.

### Additional open point created by this follow-up

- The classical forecasting baseline family is not yet fixed and should later be enumerated explicitly for pre-training, batch testing, and stream testing.

## Follow-up Note at 2026-04-08 15:23:37 +0700

### Topic

Open-ended mathematical assumption behind the threshold time series

### Note

The current research idea should not be stated as assuming that the threshold time series is globally stable, globally stationary, or nearly constant over time.

An open-ended and safer assumption is the following:

- the threshold sequence may contain structure that is learnable;
- the next threshold may be conditionally predictable from contextual information;
- the contextual information may include current data, threshold history, and threshold dynamics;
- the threshold process may still drift or change regime over time.

In this framing, the intended mathematical assumption is closer to conditional predictability than to strict stability. A suitable high-level expression is:

$$
T_t \sim p\left(T_t \mid \text{current data}, \text{threshold history}, \text{threshold dynamics}, \text{context}\right).
$$

This leaves the thesis direction open-ended in a useful way:

- the threshold process may be locally stable without being globally stationary;
- the process may be piecewise stable, regime-dependent, or slowly drifting;
- the useful requirement is not strict stability itself, but the existence of exploitable structure for forecasting or probabilistic inference.

### Working thesis wording

The threshold sequence is assumed to be context-dependent and conditionally forecastable, rather than necessarily globally stationary.

### Open point

- The exact strength of the time-series assumption remains to be decided later, for example conditional predictability, local stability, piecewise stationarity, or another related formulation.

## Follow-up Note at 2026-04-08 15:37:50 +0700

### Topic

Open-ended Gaussian assumption for the threshold series

### Note

The current idea should not be written as assuming that the entire threshold time series is globally Gaussian.

A safer and more precise assumption is the following:

- the next threshold may be modeled as Gaussian only after conditioning on local context;
- the local context may include current data, recent thresholds, threshold dynamics, and related contextual information;
- the resulting Gaussian statement is therefore local and conditional, not global and unconditional.

A concise mathematical form is:

$$
T_t \mid \mathcal{C}_t \sim \mathcal{N}(\mu_t, \sigma_t^2),
$$

where \(\mathcal{C}_t\) denotes the local context available at time \(t\).

This keeps the research direction open-ended in an appropriate way:

- the overall threshold process may still be nonstationary;
- the global marginal distribution of thresholds may still be non-Gaussian;
- different regimes or contexts may induce different local Gaussian parameters;
- the Gaussian assumption is being used as a practical conditional forecasting model, not as a claim about the full threshold sequence.

### Working thesis wording

We may assume that the next threshold is conditionally Gaussian given local context, rather than assuming that the full threshold time series is globally Gaussian.

### Open point

- The thesis still needs to decide whether the conditional Gaussian assumption is only an initial modeling approximation or a core probabilistic commitment of the final method.

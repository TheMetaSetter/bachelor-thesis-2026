---
date: 2026-06-11 17:28:40 +0700
researcher: TheMetaSetter
git_commit: 4a4e23939b0b8961fa27919282c1622e44840d66
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed interpretation of single-task two-view contrastive pretraining for DMTRL-LAF warm-start"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-06-11
last_updated_by: TheMetaSetter
---

# Research: Detailed interpretation of single-task two-view contrastive pretraining for DMTRL-LAF warm-start

**Date**: 2026-06-11 17:28:40 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `4a4e23939b0b8961fa27919282c1622e44840d66`  
**Branch**: `dev`

## Research Question
Use `prompts/1_research_prompt.md` to understand the repository context more deeply, then document in detail the intended experiment in which each task-specific encoder receives two views, a normal window and its corresponding synthetic anomalous window, uses time-step-level InfoNCE, and later serves as the warm-start source for DMTRL-LAF factorization.

## Summary
The active repository already fixes the synthetic anomaly taxonomy, the two-view contrastive batch contract, the window size, and the multi-class classification label space. Under the current task configuration, classification labels use `redlamp_multiclass`, which means a twelve-class space consisting of `normal` plus the eleven RedLamp anomaly families. The synthetic anomaly injector is already the single owning surface for building the anomalous view `x'`, the time-step anomaly mask `M`, and the per-window `classification_labels`.

The user's clarified experiment should be understood as a single-task pretraining protocol layered on top of those existing contracts. It is not the same as the active shared-encoder multitask code path in `src/models/thesis_multitask.py`. Instead, the intended protocol trains two separate encoders first. The classification encoder learns a twelve-class window-level objective while also receiving a time-step-level two-view InfoNCE term. The reconstruction encoder learns to reconstruct the clean normal window from both the clean and synthetic views, again with a time-step-level two-view InfoNCE term. After these two single-task encoders converge, their learned encoder weights become the source tensors for the later DMTRL-LAF stacking and factorization stage.

## Detailed Findings

### Data Preparation
- The active synthetic anomaly taxonomy is fixed in `src/data/augment.py` as eleven RedLamp anomaly families: `spike`, `flip`, `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`, `upsidedown`, and `mixture`.
- The active multi-class class-name surface is:

$$
\mathcal{Y}_{\text{multiclass}}
=
\{\text{normal}\}\ \cup\ \{\text{11 anomaly families}\}
$$

so the classification label space has cardinality

$$
|\mathcal{Y}_{\text{multiclass}}| = 12.
$$

- In the current task configuration, `classification_label_mode: redlamp_multiclass` is already active. Therefore, clean windows carry class label `0`, and anomalous windows carry one of the anomaly-family indices in `1,\dots,11`.
- The contrastive design note already fixes the two-view batch contract:

$$
x \in \mathbb{R}^{B \times L \times D},
\qquad
x' \in \mathbb{R}^{B \times L \times D},
\qquad
M \in \{0,1\}^{B \times L},
$$

with active window length

$$
L = 20.
$$

- The current injector implementation already produces:
  - `classification_labels`,
  - `classification_class_names`,
  - `synthetic_anomaly_mask`,
  - `augmentation_metadata`.

So the proposed experiment does not need a new label taxonomy or a new augmentation metadata format. It reuses the repository's current synthetic data contract.

### Modeling and Training

#### 1. Intended single-task decomposition
- The user's clarified experiment should be read as a pre-DMTRL-LAF warm-start stage with two independent encoders:
  - a classification encoder `E_cls`,
  - a reconstruction encoder `E_rec`.
- This differs from the active `thesis_multitask` implementation, which still runs one multitask model with a shared encoder-facing path and then splits into reconstruction and classification heads downstream.

#### 2. Classification branch interpretation
- For each sample pair, define:

$$
x = \text{normal window},
\qquad
x' = \text{corresponding synthetic anomalous window}.
$$

- The classification encoder produces:

$$
z_{\text{cls,normal}} = E_{\text{cls}}(x),
\qquad
z_{\text{cls,synth}} = E_{\text{cls}}(x').
$$

- The classification head produces logits over the twelve-class space:

$$
\hat{y}_{\text{cls,normal}} = H_{\text{cls}}(z_{\text{cls,normal}}),
\qquad
\hat{y}_{\text{cls,synth}} = H_{\text{cls}}(z_{\text{cls,synth}}),
$$

with target labels

$$
y_{\text{normal}} = 0,
\qquad
y_{\text{synth}} \in \{1,\dots,11\}.
$$

- Therefore, the intended classification loss for one pair is:

$$
\mathcal{L}_{\text{cls-head}}
=
\operatorname{CE}(\hat{y}_{\text{cls,normal}}, y_{\text{normal}})
+
\operatorname{CE}(\hat{y}_{\text{cls,synth}}, y_{\text{synth}}).
$$

This is a twelve-class anomaly-type classification objective, not a binary clean-versus-anomalous objective.

#### 3. Reconstruction branch interpretation
- The reconstruction encoder produces:

$$
z_{\text{rec,normal}} = E_{\text{rec}}(x),
\qquad
z_{\text{rec,synth}} = E_{\text{rec}}(x').
$$

- The reconstruction head produces:

$$
\hat{x}_{\text{rec,normal}} = H_{\text{rec}}(z_{\text{rec,normal}}),
\qquad
\hat{x}_{\text{rec,synth}} = H_{\text{rec}}(z_{\text{rec,synth}}).
$$

- Under the user's clarified contract, both reconstructions are supervised against the clean normal target `x`:

$$
\mathcal{L}_{\text{rec-head}}
=
\ell_{\text{rec}}(\hat{x}_{\text{rec,normal}}, x)
+
\ell_{\text{rec}}(\hat{x}_{\text{rec,synth}}, x).
$$

This is a denoising-style reconstruction objective. The synthetic view is not reconstructed back to `x'`. It is reconstructed back to the clean view `x`.

#### 4. Time-step-level InfoNCE on each task-specific encoder
- The active design note now documents a task-specific two-view InfoNCE interpretation with:

$$
z_{\text{normal}} = f_\theta(x_{\text{normal}}),
\qquad
z_{\text{synth}} = f_\theta(x_{\text{synth}})
$$

and with time-step-level positives built only from timesteps not touched by the synthetic injector.

- For a task-specific encoder, define:

$$
q_{b,t} = \operatorname{norm}(z_{\text{normal},b,t}),
\qquad
k^+_{b,t} = \operatorname{norm}(z_{\text{synth},b,t}),
$$

with positive anchor set

$$
\mathcal{I}_+ = \{(b,t)\mid M_{b,t}=0\}.
$$

- The InfoNCE objective is:

$$
\mathcal{L}_{\text{ctr}}
=
-\frac{1}{|\mathcal{I}_+|}
\sum_{(b,t)\in\mathcal{I}_+}
\log
\frac{
\exp(\operatorname{sim}(q_{b,t}, k^+_{b,t})/\tau_c)
}{
\exp(\operatorname{sim}(q_{b,t}, k^+_{b,t})/\tau_c)
+
\sum_{(b',t')\in\mathcal{N}_{b,t}}
\exp(\operatorname{sim}(q_{b,t}, \tilde{k}_{b',t'})/\tau_c)
},
$$

where tokens with `M_{b,t}=1` do not form positives and participate only as negatives.

- In the user's intended protocol, this term should be instantiated separately for each task-specific encoder:

$$
\mathcal{L}_{\text{ctr,cls}},
\qquad
\mathcal{L}_{\text{ctr,rec}}.
$$

#### 5. Per-task total objectives
- The intended single-task classification pretraining objective is:

$$
\mathcal{L}_{\text{task,cls}}
=
\mathcal{L}_{\text{cls-head}}
+
\lambda_{\text{ctr}}\mathcal{L}_{\text{ctr,cls}}.
$$

- The intended single-task reconstruction pretraining objective is:

$$
\mathcal{L}_{\text{task,rec}}
=
\mathcal{L}_{\text{rec-head}}
+
\lambda_{\text{ctr}}\mathcal{L}_{\text{ctr,rec}}.
$$

#### 6. Forward-pass chronology for one paired sample batch
- Classification branch:
  1. build `(x, x', M, y_normal, y_synth)`,
  2. compute `z_cls_normal` and `z_cls_synth`,
  3. compute `\mathcal{L}_{\text{ctr,cls}}` at the time-step level,
  4. compute classification logits for both views,
  5. compute `\mathcal{L}_{\text{cls-head}}`,
  6. sum to `\mathcal{L}_{\text{task,cls}}`.
- Reconstruction branch:
  1. build `(x, x', M)`,
  2. compute `z_rec_normal` and `z_rec_synth`,
  3. compute `\mathcal{L}_{\text{ctr,rec}}` at the time-step level,
  4. decode both latent tensors,
  5. compare both reconstructions against the same clean target `x`,
  6. sum to `\mathcal{L}_{\text{task,rec}}`.

#### 7. Backward-pass interpretation
- For the classification encoder:

$$
\frac{\partial \mathcal{L}_{\text{task,cls}}}{\partial \theta_{\text{cls}}}
=
\frac{\partial \mathcal{L}_{\text{cls-head}}}{\partial \theta_{\text{cls}}}
+
\lambda_{\text{ctr}}
\frac{\partial \mathcal{L}_{\text{ctr,cls}}}{\partial \theta_{\text{cls}}}.
$$

- For the reconstruction encoder:

$$
\frac{\partial \mathcal{L}_{\text{task,rec}}}{\partial \theta_{\text{rec}}}
=
\frac{\partial \mathcal{L}_{\text{rec-head}}}{\partial \theta_{\text{rec}}}
+
\lambda_{\text{ctr}}
\frac{\partial \mathcal{L}_{\text{ctr,rec}}}{\partial \theta_{\text{rec}}}.
$$

- The important consequence is:
  - `cls head` gradients come only from the twelve-class classification objective,
  - `rec head` gradients come only from the denoising reconstruction objective,
  - both encoders receive head gradients and contrastive gradients together.

#### 8. Relationship to the active code path
- The active repository already supports:
  - twelve-class synthetic anomaly labels through `redlamp_multiclass`,
  - synthetic anomaly masks `M`,
  - two-view pair construction for contrastive training,
  - reconstruction and classification loss computation in the offline multitask model.
- However, the active code path does **not** currently implement the user's clarified single-task protocol literally:
  - `src/models/thesis_multitask.py` still prepares clean and augmented pair batches for a shared multitask pipeline rather than two separately trained task-specific encoders,
  - the active reconstruction loss compares `outputs["recon"]` to `batch["x"]` from the same batch, with optional normal-time-step masking, rather than explicitly decoding `x'` toward a separate clean paired target `x`,
  - the active classification loss already supports multiclass supervision but is embedded inside the shared multitask model rather than inside an isolated single-task classifier warm-start stage.

### Evaluation
- For the intended single-task warm-start stage, the natural validation breakdown is:
  - classification branch: twelve-class accuracy, macro-F1, or confusion-matrix diagnostics on `normal + 11 anomaly families`,
  - reconstruction branch: reconstruction loss on clean windows and denoising reconstruction loss on synthetic windows against clean targets,
  - both branches: `\mathcal{L}_{\text{ctr}}` monitored separately from head loss.
- The design note already states that synthetic validation pairs are part of Experiment 2 monitoring, so the user's proposed warm-start stage is consistent with the repository's existing validation philosophy.

## Code References
- `prompts/1_research_prompt.md:1` - required research-note workflow and output format
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md:45` - two-view batch contract with `x`, `x'`, and `M`
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md:104` - time-step-level InfoNCE definition
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml:5` - active task config uses `classification_label_mode: redlamp_multiclass`
- `src/data/augment.py:21` - eleven active RedLamp anomaly families
- `src/data/augment.py:34` - twelve-class surface `normal + 11 anomaly families`
- `src/data/augment.py:786` - class-index surface depends on label mode
- `src/data/augment.py:812` - label sampling for clean and anomaly-family windows
- `src/data/augment.py:849` - batch augmentation path that writes `classification_labels` and `synthetic_anomaly_mask`
- `src/models/thesis_multitask.py:1651` - clean batch defaults classification labels to `0` and mask to all-zero
- `src/models/thesis_multitask.py:1696` - current code builds clean/augmented pairs for two-view contrastive mode
- `src/models/thesis_multitask.py:1900` - current reconstruction loss compares reconstruction to `batch["x"]`
- `src/models/thesis_multitask.py:1964` - current classification loss uses `classification_labels`
- `src/models/redlamp_mlp_baseline.py:99` - baseline classifier output dimension defaults to twelve classes

## Pipeline Documentation
The intended warm-start experiment should be documented as the following sequence:

`sample clean window x -> inject one synthetic family to create x' and mask M -> train classification encoder on (x, x') with twelve-class supervision plus time-step InfoNCE -> train reconstruction encoder on (x, x') with clean-target denoising supervision plus time-step InfoNCE -> export the two learned encoder parameter tensors -> stack corresponding task weights -> factorize them for DMTRL-LAF initialization -> continue with multitask factorized fine-tuning`

So, hiểu nôm na thì, đây là một protocol ba tầng:

1. pair generation and synthetic labeling,
2. two independent single-task warm-start trainings,
3. DMTRL-LAF merging and later multitask fine-tuning.

## Historical Context (from documents/)
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md` already established the active two-view contrastive contract for the offline phase.
- `documents/brainstorming-notes/brainstorming-notes-dmtrl-laf.md` already established that DMTRL-LAF should start from separately trained single-task kernels, then stack, flatten, factorize, and fine-tune.
- `documents/logs/06-11-2026/research/research-dmtrl-laf-model-architecture-description.md` already recorded that the DMTRL-LAF note is an encoder-parameterization design rather than a full implemented training path in `src/`.
- This new note adds the missing bridge between those two documents: it spells out how the single-task warm-start stage can be interpreted when the classification target space is the active twelve-class RedLamp taxonomy and the reconstruction target is the clean paired window.

## Open Questions
- The current repository does not yet define a dedicated training entrypoint for these two isolated task-specific encoders before DMTRL-LAF factorization.
- The current repository does not yet expose a stored artifact contract for "paired clean target used by synthetic-view reconstruction" because the active reconstruction path reconstructs the batch's own `x`.
- The current repository does not yet define whether the two single-task encoders should share architecture exactly layer-by-layer before factorization, although that is the natural assumption for later parameter stacking.
- The current repository does not yet define whether `\lambda_{\text{ctr}}` should be identical for classification and reconstruction warm-start stages or tuned separately.

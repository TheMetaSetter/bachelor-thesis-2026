---
date: 2026-05-19T14:01:11+07:00
researcher: Artificial Intelligence Agent
git_commit: unknown
branch: unknown
repository: bachelor-thesis-2026
topic: "Comparison of Brainstorming, Design, and Codebase Axes"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-19
last_updated_by: Artificial Intelligence Agent
---

# Research: Comparison of Brainstorming, Design, and Codebase Axes

**Date**: 2026-05-19T14:01:11+07:00
**Researcher**: Artificial Intelligence Agent
**Git Commit**: unknown
**Branch**: unknown

## Research Question
Đối chiếu lại ý tưởng mới trong `brandstorming-notes` với các thiết kế hiện tại trong `design` và tình hình thực tế hiện tại của codebase. Sử dụng `1_research_prompt.md` để scan các luồng tính toán chính và báo cáo dạng bảng so sánh sự khác nhau giữa 3 trục tài liệu.

## Summary
The pipeline involves a time-series anomaly detection system utilizing continuous and discrete prototypes. The analysis reveals a progression from early brainstorming ideas (token-level contrastive loss on injected anomalies, L=20) to a more structured multi-task design (L=100, task-specific fusion, modular objective), which is accurately reflected in the current codebase. However, the contrastive idea from the brainstorming notes has not been implemented in the main `ThesisMultitaskModel` yet, highlighting a divergence between the newly brainstormed ideas and the current implementation.

## Detailed Findings

### Data Preparation
- **Brainstorming Notes**: Proposes a window length of $L = 20$. Relies heavily on normal $x$ and anomalous $x'$ generated via anomaly injection for contrastive learning.
- **Design Documents**: Sets window length to $L = 100$. Focuses on a streaming pipeline with `River`, custom wrappers for SMD, MSL, SMAP, SWaT, UCR, and controlled synthetic drift/anomaly injection (11 RedLamp classes) using CARLA's subsequence mechanisms.
- **Codebase**: `SyntheticAnomalyInjector` in `src/data/augment.py` implements the 11 RedLamp anomaly families (spike, flip, speedup, noise, cutoff, average, scale, wander, contextual, upsidedown, mixture). Window length is configurable via `ThesisMultitaskModelConfig`.

### Modeling and Training
- **Brainstorming Notes**: 
  - Suggests a token-level contrastive loss ($\mathcal{L}_{\text{contrast}}$) that pulls normal positions together and pushes injected anomalous positions apart.
  - Continuous prototypes heavily assigned to reconstruction.
  - Discrete prototypes heavily assigned to classification.
- **Design Documents**: 
  - Fused representations for tasks: $H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)}$ and $H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)}$.
  - Default objective is $\mathcal{L}_{\text{recon}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$. Additional losses (diversity, variance, covariance, usage, gate) are strictly optional to maintain objective modularity.
  - Online adaptation uses a frozen reference encoder, partially trainable online encoder, and a near-identity residual projector.
- **Codebase**:
  - `ThesisMultitaskModel` in `src/models/thesis_multitask.py` strictly follows the Design Documents. It uses `alpha` and `beta` fusion scalars and implements all optional losses as toggles. The contrastive loss from the brainstorming notes is **not** present.
  - `OnlineAdaptationModel` in `src/models/online_adaptation.py` mirrors the design: employs `ThesisMultitaskEncoderAdapter`, `ResidualProjector` (zero-initialized final layer), and uses alignment loss (cross-entropy on pooled representations), prototype alignment loss, and anchor loss.

### Evaluation
- **Design Documents**: Emphasizes causal thresholding. Warns against future-leakage (evaluating thresholds using quantiles from the test set). Evaluates under clean streaming, real streaming with drift, and synthetic streaming.
- **Codebase**: The online loop evaluator processes sequential windows, tracking projector drift, anchor loss, and alignment loss over time.

## Pipeline Documentation
The current pipeline centers on a fixed-length window sequence feeding into an MLP-based `MultitaskWindowEncoder`. The hidden representation queries continuous (attention-based) and discrete (Gumbel-Softmax) prototype banks. The retrieved contexts are fused via learned scalars ($\alpha, \beta$) into separate reconstruction and classification states. Online adaptation freezes this model as a reference, duplicates the encoder, and adapts only a `ResidualProjector` to map shifted incoming data back to the reference geometry.

## Comparison Table

| Feature / Axis | Brainstorming Notes (`brainstorming-notes-contrastive-prototype-ts.md`) | Design Documents (`idea.md`, `stream_design.md`) | Current Codebase (`src/models/`) |
| :--- | :--- | :--- | :--- |
| **Window Length (L)** | 20 time-steps | 100 time-steps | Configurable via `MultitaskArchitectureConfig` |
| **Primary Objective** | Token-level contrastive loss between $x$ and $x'$ (pull normal, push anomalous). | Modular objective: $\mathcal{L}_{\text{recon}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$ (default). | Modular objective implemented. No token-level contrastive loss. |
| **Prototype Usage** | Continuous -> Reconstruction; Discrete -> Classification. | Task-specific fusion using learned weights $\alpha$ (cls) and $\beta$ (rec). | Implemented via `alpha_logit` and `beta_logit` fusion parameters. |
| **Anomaly Injection** | Mentioned as generic `InjectAnomaly(x_t)`. | 11 RedLamp anomaly types with subsequence mechanics. | Fully implemented in `SyntheticAnomalyInjector` (11 families). |
| **Online Adaptation** | N/A (Focuses on contrastive views). | Frozen reference, trainable online encoder, near-identity projector. Alignment loss. | Implemented `OnlineAdaptationModel` with `ResidualProjector`, `alignment_loss`, `anchor_loss`. |

## Historical Context (from documents/)
The codebase accurately reflects `idea.md`, indicating successful implementation of the "Phase 4" offline multi-task model and the first online slice. However, the `brandstorming-notes-contrastive-prototype-ts.md` introduces a novel token-level contrastive loss relying on explicit anomaly index sets ($A$ and $A^c$), which represents a conceptual shift from the sequence-level classification loss ($\mathcal{L}_{\text{cls}}$) currently driving the anomaly detection in the codebase.

## Open Questions
- Should the token-level contrastive loss from the brainstorming notes be integrated into the modular objective as an optional regularizer, or replace the sequence-level classification loss entirely?
- How should the distance function for the contrastive loss (squared Euclidean vs Cosine) interact with the existing Gumbel-Softmax and attention mechanisms?

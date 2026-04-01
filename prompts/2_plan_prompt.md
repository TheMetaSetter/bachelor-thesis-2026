Using the research note at `documents/logs/MM-DD-YYYY/research/research-<kebab-case-topic>.md`, create a detailed implementation plan for the multivariate time-series anomaly detection research codebase. The plan must reflect the thesis intent in `documents/design/idea.md` and the engineering guidance in `documents/design/design_starter.md`. Ground all recommendations in the current repository state and the documented risks and mitigations.

Your plan must include specific and detailed instructions for programming and building within this codebase. Use formal, academic language and complete sentences.

---

Based on my research, here is what I found:

## Current State
- [Key discovery about existing code, data pipelines, and configuration structure]
- [Established contracts or interfaces that the code already follows]
- [Existing model, task, or engine modules that must be preserved]

## Design Options
- [Option A: backbone with publicly released model weights and an explicit hidden-state contract implemented by an adapter module]
- [Option B: self-trained spectral-temporal encoder with the same contract and adapter module]
- [Option C: online adaptation strategy comparisons such as dual encoder versus teacher-student alignment with a moving average teacher]

## Risk and Mitigation
- [Risk: continuous and discrete prototype branches are redundant; Mitigation: ablation plan and explicit logging]
- [Risk: fusion collapses to one branch; Mitigation: balanced losses and monitoring]
- [Risk: adaptation contamination by anomalous batches; Mitigation: gated updates]
- [Risk: projector drift and poor initialization; Mitigation: residual adapter, warm-start, anchor regularization]
- [Risk: high-variance updates from a single batch; Mitigation: conservative parameter updates]
- [Risk: evaluation metric inflation; Mitigation: explicit metric definitions and reporting]

## Open Questions
- [Technical uncertainty that blocks implementation or evaluation]
- [Design decision needed about encoders, datasets, or adaptation strategy]

Which approach aligns best with your vision?

---

Plan requirements:
- Specify file paths, module names, class names, and interfaces to be added or modified.
- State how the batch contract, encoder contract, and model output contract are enforced.
- Use software engineering principles such as separation of concerns, single responsibility, and stable interfaces.
- Apply design pattern principles such as composition over inheritance, adapter pattern for encoders, strategy pattern for tasks, and a registry or factory for datasets and models.
- Include test plans, configuration changes, and validation procedures.
- Ensure that the plan follows a minimal vertical slice before advanced prototype modules and online adaptation.

Place your findings under:

`documents/logs/MM-DD-YYYY/plan/`

Example filename:

`documents/logs/MM-DD-YYYY/plan/plan-<kebab-case-topic>.md`

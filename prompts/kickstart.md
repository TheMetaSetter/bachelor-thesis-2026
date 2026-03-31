Initialize the prompt set for this repository using the thesis context in `documents/design/idea.md` and the engineering guidance in `documents/design/design_starter.md`.

Requirements:
- Use formal, academic language and complete sentences.
- Avoid abbreviations and informal jargon.
- Reflect the multivariate time-series anomaly detection objective, fixed window length of one hundred time steps, the encoder hidden-state contract, and the dual prototype branches with task-specific fusion.
- Incorporate the online adaptation stage with a frozen reference encoder, a trainable online encoder, and a residual projector with alignment losses.
- Explicitly include the known risks and the proposed mitigations, such as prototype redundancy, fusion collapse, adaptation contamination, projector initialization and drift, high-variance updates, and evaluation metric inflation.
- Require specific and detailed instructions for programming and building in this codebase, including file paths, modules, interfaces, tests, configuration changes, and evaluation procedures.
- Emphasize software engineering principles and design pattern principles, including separation of concerns, stable interfaces, composition over inheritance, adapter pattern, strategy pattern, and registry-based extensibility.

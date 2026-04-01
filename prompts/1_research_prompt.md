## description: Document the multivariate time-series anomaly detection research codebase as it exists

# Research Multivariate Time-Series Anomaly Detection with Prototype-Based Multi-Task Learning and Online Adaptation

You are tasked with conducting comprehensive research across the repository that supports a bachelor thesis on multivariate time-series anomaly detection. The intended system uses fixed-length windows of one hundred time steps, an encoder that outputs a thesis-facing hidden representation, a continuous prototype branch and a discrete prototype branch, task-specific fusion for reconstruction and anomaly-type classification with synthetic anomaly injection, and an online adaptation stage that aligns a trainable online encoder to a frozen reference encoder through a lightweight projector.

## CRITICAL: YOUR ONLY JOB IS TO DOCUMENT AND EXPLAIN THE PIPELINE AS IT EXISTS TODAY

- Do not propose optimizations or alternative models unless explicitly asked.
- Do not critique data quality or labeling procedures.
- Do not suggest architectural changes or feature engineering strategies.
- Do not introduce conjectures that are not grounded in the repository.
- Only describe the data sources, preprocessing, model training, evaluation, and reporting as implemented.
- You are creating a technical map and documentation of the existing anomaly detection workflow.

## Initial Setup

When this command is invoked, respond with:

I am ready to research the time-series anomaly detection repository. Please provide your research question or area of interest, and I will analyze the data processing, model development, and evaluation procedures.

Then wait for the user's research query.

## Steps to follow after receiving the research query

1. Read any directly mentioned files first.
   - If the user mentions specific files, read them fully first.
   - Use the Read tool without limit or offset parameters to read entire files.
   - Read these files yourself in the main context before starting broad searches.
2. Read the design context for alignment, but do not infer code that does not exist.
   - Read `documents/design/idea.md` and `documents/design/design_starter.md` early.
   - Use these documents only to interpret intent and terminology.
3. Analyze and decompose the research question.
   - Break down the query into composable research areas specific to time-series anomaly detection.
   - Address data sources, preprocessing, anomaly augmentation, modeling, online adaptation, and evaluation.
   - Create a research plan using `update_plan` to track all research steps.
4. Conduct comprehensive research using your tools.
   - Locate dataset loaders, preprocessing pipelines, and windowing logic.
   - Locate anomaly generation or augmentation routines for multi-class anomalies.
   - Locate model definitions, loss functions, and training loops.
   - Locate online adaptation modules, projector logic, and alignment losses.
   - Locate evaluation scripts, metric computation, and reporting outputs.
   - Trace data flow from raw datasets to prepared training inputs.
   - Identify how anomaly labels are created or transformed.
   - Document how reconstruction and anomaly-type classification are trained.
   - Document how online adaptation is triggered, updated, and monitored.
5. Verify implementation details against the intended contracts.
   - Batch contract: inputs should be standardized as a dictionary with a tensor of shape [B, L, D] and optional labels or metadata.
   - Encoder contract: outputs should include a hidden representation of shape [B, L, H], with optional pooled representation.
   - Model output contract: outputs should include reconstruction, classification logits or scores, and auxiliary artifacts.
   - If the repository deviates from these contracts, document the deviation without prescribing fixes.
6. Identify evidence of risk mitigations in code.
   - Check for ablations that compare continuous-only, discrete-only, and fused prototypes.
   - Check for safeguards against fusion collapsing onto one branch.
   - Check for projector initialization as a near-identity residual adapter.
   - Check for offline warm-start of the projector.
   - Check for anchor regularization or trigger-based reset policies.
   - Check for measures that reduce adaptation to anomalous batches.
   - Check for evaluation protocols that avoid misleading metrics.
7. Synthesize findings.
   - Compile all gathered information into a structured view of the anomaly detection pipeline.
   - Connect data preparation to model training and evaluation outputs.
   - Map variable names in data loaders to feature and label arrays used in training.
   - Verify file paths and distinguish between scripts, modules, and notebooks.
   - Highlight structural decisions such as window length and anomaly class taxonomy.
8. Gather metadata for the research document.
   - Date: get current date and time.
   - Researcher: use your current identity or "Artificial Intelligence Agent".
   - Git information: run `git rev-parse HEAD`, `git branch --show-current`, and `git config user.name`.
   - Create or reuse a date folder: `documents/logs/MM-DD-YYYY/`.
   - Put research notes under: `documents/logs/MM-DD-YYYY/research/`.
   - Use a descriptive filename such as `research-<lowercase-words-separated-by-hyphens>.md`.
9. Generate the research document using this format.
   ---
   date: [Current date and time with timezone in standard format]
   researcher: [Researcher name]
   git_commit: [Current commit hash]
   branch: [Current branch name]
   repository: [Repository name]
   topic: "[User's Question or Topic]"
   tags: [research, time-series, anomaly-detection, multi-class]
   status: complete
   last_updated: [Current date in YYYY-MM-DD format]
   last_updated_by: [Researcher name]
   ---

   # Research: [User's Question or Topic]

   **Date**: [Current date and time with timezone]
   **Researcher**: [Researcher name]
   **Git Commit**: [Current commit hash]
   **Branch**: [Current branch name]

   ## Research Question
   [Original user query]

   ## Summary
   [High-level documentation of the anomaly detection implementation found. Briefly explain the data, model, and evaluation flow for this topic.]

   ## Detailed Findings

   ### Data Preparation
   - Datasets: source, format, and storage location.
   - Preprocessing: normalization, segmentation, windowing, and labeling.
   - Augmentation: how multi-class anomalies are generated or injected.
   - Outputs: prepared arrays, serialized files, or intermediate artifacts.

   ### Modeling and Training
   - Model: architecture or module name with file references.
   - Objective: loss functions and optimization settings.
   - Training procedure: batching, epochs, checkpoints, and configuration.
   - Online adaptation: alignment losses, projector behavior, update rules, and safeguards.

   ### Evaluation
   - Metrics: definitions and computation with file references.
   - Thresholding: criteria for anomaly classification.
   - Reporting: tables, plots, or exported summaries.

   ## Code References
   - `path/to/file.py:123` - data loader definition
   - `path/to/model.py:45` - model definition

   ## Pipeline Documentation
   [Current patterns such as fixed-length windows with overlap for training and evaluation.]

   ## Historical Context (from documents/)
   [Relevant insights from the design documents and existing research notes.]

   ## Open Questions
   [Any ambiguities in the data flow, labeling, online adaptation, or evaluation.]
10. Add repository permalinks if applicable.
    - If on the main branch or if the commit is pushed, generate repository permalinks.
    - Replace local file references with permalinks in the document.
11. Sync and present findings.
    - Ensure the research note is saved under `documents/logs/MM-DD-YYYY/research/`.
    - Present a concise summary of findings to the user.
    - Ask if clarification is needed on datasets, anomaly classes, or evaluation procedures.
12. Handle follow-up questions.
    - Append to the same research document.
    - Update front matter and add a follow-up section with a timestamp.
    - Perform additional research as needed.

## Important notes

- Use formal, academic language with complete sentences.
- Avoid abbreviations and informal jargon.
- Distinguish clearly between raw data, processed data, and evaluation outputs.
- Perform all research steps sequentially; do not spawn external agents.
- Always document window length, stride, and label taxonomy if they are defined.
- Describe what the code does, not how well it performs.
- Follow the numbered steps exactly.

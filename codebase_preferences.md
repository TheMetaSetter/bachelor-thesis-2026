# Codebase Preferences for `bachelor-thesis-2026`

## 1. Reference mechanisms and source code alignment

### For data loaders
1. Data loaders need to follow mechanisms under TSLib codebase. Refer to code under `bsc-thesis-ref-codebases/Time-Series-Library`.
2. Synthetic anomaly injection needs to follow mechanisms under CARLA codebase and RedLamp codebase. Refer to code under `bsc-thesis-ref-codebases/CARLA-main/data/augment.py` and `bsc-thesis-ref-codebases/RedLamp/loaders/loader_aug.py`.

### For discrete prototypes or codebook
- Discrete prototypes or codebook follow mechanisms under DALL-E codebase. Refer to code under `bsc-thesis-ref-codebases/DALL-E-master/dall_e`.

---

## 2. Logging and task organization

- Place logs under `documents/logs/MM-DD-YYYY/<task_type>` with `task_type` being one of the four following: `research`, `plan`, `structure`, or `detail`.
    - Research is following prompt under `prompts/1_research_prompt.md`
    - Plan is following prompt under `prompts/2_plan_prompt.md`
    - Structure is following prompt under `prompts/3_structure_prompt.md`
    - Detail is following prompt under `prompts/4_detail_prompt.md`

- Use Weights & Biases to log artifacts and statistics from experiments, and history of experiments, to make sure each experiment in the history can be reproduced easily without effort.

### Experimental result directory and artifact constraints
- Treat top-level output directories such as `outputs/benchmark` and `outputs/benchmark_smoke` as experiment roots. The ideal result-path depth is 3 levels below the experiment root; keep paths shallow and within approximately 3–6 levels, with 6 levels as the absolute maximum. Six levels is a hard limit, not the usual target.
- Design audit-facing outputs to minimize cognitive overload for other researchers. Persist the task-appropriate summary statistics, essential checkpoints, evaluation protocol, configuration, provenance, and explicitly selected diagnostics; do not retain large raw tensors, traces, intermediate logs, tool-generated internals, or duplicated artifacts by default unless a documented analysis or audit need requires them.

### Canonical experiment-result hierarchy and storage policy

- Use the canonical hierarchy:
  `outputs/<experiment_type>/<dataset_name>/<entity_name>/<seed_value>/<method_name>/<phase_name>/<stage_name>/`.
  For example, `outputs/benchmark/smd/machine_1_6/seed6/thesis/offline/stage_a/`.
- `benchmark` and `benchmark_smoke` are experiment types. The folders below
  them represent dataset, entity, seed, method, phase, and stage in that
  order. Existing historical trees may differ; discovery and adapters must
  handle them without redefining the canonical hierarchy.
- A `[stage_name]` folder stores the artifacts for one stage of one phase of
  one method, entity, dataset, and seed combination.
- Minimize experiment data written to disk. Compute intermediate values
  on-the-fly and persist only summary statistics required for final reports,
  uncertainty analysis, provenance, reproducibility, and explicitly selected
  diagnostics.
- Do not persist every neural-network forward-pass output by default. Raw
  per-pass traces, tensors, and duplicated retention bundles require an
  explicit downstream analysis or audit justification.
- Every stage must retain the checkpoint used to initialize the stage and the
  best checkpoint selected by the stage's monitoring rule. Record their paths,
  roles, and checksums in the stage provenance/manifest.

---

## 3. Planning and codebase workflow order

- `plan` to use which design patterns before `structure`.

- Traverse the codebase using the directory tree inside `documents/abstract-design-notes/design_starter.md`. This tree is planned to be fixed most of the time. But if you really need to change the structure of the directory, please remember to update the tree.

---

## 4. Model file organization and self-contained design

- Each model has exactly one public entrypoint in `src/models/`. The entrypoint owns the constructor, public inference/training API, configuration boundary, and checkpoint contract.

- Helpers may be placed in separate files only when they are small reusable primitives or immutable configuration objects. Helpers must not use mixins to distribute a model lifecycle, define a second public model, or hide phase-specific runtime behavior.

- A reader must still be able to start at the public model entrypoint and follow the complete runtime flow through explicitly named helpers without guessing.

---

## 5. Readability-first principles

This is important so I will repeat 3 times.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In this codebase, readability is king. So, DO repeat yourself in a thoughtful way.

- Stick to "least amount of codepaths" principle, which means configurations, models, pre-processing, all of that should be obvious to users. Reading should be obvious and configurations should be obvious.

- Minimalistic, easy to comprehend, and single-meaning are mandatory codebase-wide criteria.
  - Minimalistic means prefer the smallest workable design that preserves behavior.
  - Easy to comprehend means the reader should be able to infer the intended contract directly from file names, function names, comments, and config keys.
  - Single-meaning means one term, one contract, one runtime concept. If a term can plausibly mean two different things, rename or separate it.

- Keep methods and functions short.
  - A method inside a class must not exceed 50 lines.
  - A standalone function must not exceed 50 lines.

- Keep files small.
  - A code file must not exceed 500 lines.
  - If a file grows past that size, split it into smaller files or move non-core helpers into a narrower boundary.

- When writing CLI instructions or shell commands for the user, prefer one short command per line or per tmux `send-keys` step.
  - Avoid bundling many actions into one long shell block unless the user explicitly asks for a scripted batch.
  - Each command should be easy to inspect, rerun, and debug in isolation.

- When using SSH on a shared remote GPU, prefer a read-only check first, then scope any write or cleanup action to exact process IDs, exact run directories, or exact artifact paths.
  - Treat broad `rm -rf` cleanup as a last resort and only after confirming the target tree is safe to remove.
  - Before leaving the host, re-check that no unrelated jobs, logs, or temporary files were touched.

- Before running a full benchmark batch, always execute the full experiment flow on 1 concrete combination from the development specification first. If the project has enough time, this first-pass check may expand to 3 combinations. For this project, keep the first-pass check to 1 combination unless the user explicitly requests more. Only after that first end-to-end pass succeeds should you launch all combinations.

- Do not write overly long `for` loops. When a loop starts handling many responsibilities at once, split the work into small, clearly named helper functions or methods so each step can be read, reused, tested, and modified independently.

- Write explanatory comments to support user reading code. Comments should be updated in parallel with code or implementations.

- Write explanatory comments to support user reading code. Comments should be updated in parallel with code or implementations.

- Make code pedagogical-first. The primary audience for every code review, code comment, and explanation is a high-school student who should be able to understand the intent, the data flow, and the runtime contract without guessing.

- When adding or changing code, explain the surrounding computation in a way that is easy to follow from first principles. Prefer short, explicit comments that say what each step does and why it exists.

- Add cute ASCII flow diagrams regularly when they help readers understand how a snippet fits into the larger processing pipeline. The diagram should focus on the local snippet's place in the end-to-end flow, not on decoration.

- If a change cannot be explained clearly enough for a high-school student to follow, treat that as a code-quality failure and simplify the code or the explanation before considering the work done.

- When writing, reviewing, or refactoring Python code in this repository, strictly follow `python_semantics.md` as a required semantic checklist for slicing, indexing, assignment, mutability, and function-call behavior.

---

## 6. Data versioning and reproducibility

- Each time we do synthetic anomaly injection on one particular dataset will result in a different version of that particular dataset, such as SMD.

- So, please use data version control techniques, use `dvc.yaml`, to control data version, and make sure all history experiments can be reproduced without effort.

---

## 7. Testing requirements

- Add small and minimalistic test cases to check for these things:
    - Input and output tensors of calculations, computations
    - Small test to run one forward pass and one backward pass on one batch of one specific dataset, such as SMD, or let the user choose.
    - TEST THE MECHANISM TO SAVE AND LOAD CHECKPOINTS. THIS IS SUPER IMPORTANT.
    - Test data loaders to make sure the trainers receive the right batch size for each dataset, and the right shapes of tensors.
    - Allow user to test injecting synthetic anomalies by injecting anomalies in one batch of sample in one specific dataset. Then, visualize the injected samples clearly to allow user easily examine the quality of injection.
    - Test the functions or methods within classes that are used to initialize configurations from definition files, such as `.yaml` files. Refer to format under `bsc-thesis-ref-codebases/CARLA-main/configs`. These are just examples; you do not need to do the exact same thing.
    - Design and implement simple test cases using Pytest

---

## 8. Ablation study friendliness

- Design the codebase such that components (e.g., modules, loss terms, preprocessing steps, augmentations) can be easily turned on or turned off without modifying core logic.

- Each component should be controllable via clear and explicit configuration (e.g., `.yaml`), avoiding hidden dependencies or implicit coupling.

- Avoid hard-coded interactions between components; instead, use modular design so that removing or disabling one component does not break the pipeline.

- Ensure that enabling or disabling components results in minimal changes in codepaths, so that ablation experiments are easy to run, compare, and reproduce.

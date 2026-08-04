# Repository Guidelines

Locally, the root of this repo is: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/`.

Use `.venv/bin/python` to run code regularly instead of only writing code.

On macos environment, please use `realpath` command to get the full file path everytime you cite a file or a folder. File paths may contain language-specific unicode characters (e.g., Vietnamese) so you need to be extra careful with file paths while reasoning and also when respond to me.

Responded file paths containing Vietnamese must be valid in NFC Unicode normalization. Please check using very quick Python command in CLI.

All contributors and agents must strictly follow every requirement in `codebase_preferences.md`.
The single source of truth for this codebase is the `documents/` directory.

## Project Structure and Module Organization
This repository is research-first and currently centers on notebooks, design documents, and reference codebases.
- `documents/` holds design notes and research logs. Logs belong in `documents/logs/MM-DD-YYYY/<task_type>`.
- `prompts/` contains the prompt templates used for research and planning.
- `data/` stores raw datasets (for example `data/SMD`, `data/NASA`). Keep derived artifacts out of version control unless explicitly required.
- `bsc-thesis-ref-codebases/` contains external reference implementations (TSLib, CARLA, DALL-E, and others).
- Root-level `*.ipynb` notebooks are used for exploratory analysis and visualization.

Canonical experiment-result layout:

```text
outputs/<experiment_type>/<dataset_name>/<entity_name>/<seed_value>/<method_name>/<phase_name>/<stage_name>/
```

`benchmark` and `benchmark_smoke` are experiment types. A stage directory
stores the artifacts for that stage. Keep disk usage minimal by computing
intermediate values on-the-fly and persisting report-ready summary statistics,
provenance, selected diagnostics, the stage initialization checkpoint, and the
stage best checkpoint. Do not persist every forward-pass output by default.

When adding new source code, follow the guidance in `codebase_preferences.md`: one model per file, with inference and training logic colocated for readability.

## Build, Test, and Development Commands
- `conda env create -f environment.yml` creates a conda environment aligned with the research stack.
- `pip install -r requirements.txt` installs Python dependencies if using a virtual environment.
- `jupyter lab` runs local notebooks.
- `pytest` runs the test suite once tests are added.

## Coding Style and Naming Conventions
- Prefer explicit, descriptive names and short, linear control flow.
- Use four-space indentation in Python and follow PEP 8 conventions.
- Follow `python_semantics.md` whenever writing, reviewing, or refactoring Python code, especially for slicing, indexing, assignment, and mutability.
- Keep model logic self-contained in a single file. Avoid fragmented class hierarchies.
- Prefer adapter-style wrappers when integrating reference codebases.

## Simplicity-First Requirement
Always start with the simplest solution that correctly satisfies the current requirement. Apply this rule to research, specifications, architecture, implementation, tests, debugging, and explanations.

- Use the existing data flow and natural control flow when they already produce the required behavior.
- Do not add explicit state, flags, guards, abstractions, classes, configuration options, or lifecycle steps unless a concrete requirement or observed failure needs them.
- Do not design for hypothetical future cases before the current case requires that support.
- Prefer a small extension of the current code path over a parallel framework or a general-purpose redesign.
- Add complexity only after identifying the specific problem it solves. Record that reason where future readers can verify it.
- When two solutions are both correct, choose the one with fewer concepts, fewer state transitions, and fewer code paths.
- Reconsider a proposed mechanism when the runtime's natural behavior already enforces the intended result. For example, if a point stops changing because later sliding windows no longer contain it, do not add a separate point-finalization mechanism.

## Testing Guidelines
Tests should use `pytest` and remain minimal and focused. Expected coverage includes data loader shapes, one forward and backward pass, checkpoint save and load, and synthetic anomaly injection behavior as described in `codebase_preferences.md`.

Tests should be minimal and focused while also being extra skeptical.

## Commit and Pull Request Guidelines
Commit messages in history use short, imperative summaries (for example, “Add data loaders”, “Fix type errors”). Follow the same pattern without prefixes or scopes.
Pull requests should include a concise summary, testing notes, and links to relevant design documents or notebooks. If data changes are introduced, describe the dataset versioning strategy.

## Plain Language Requirements for Documentation
Every document must follow the Plain Language Guide, whether it is written in English or Vietnamese. This requirement applies to specifications, research logs, notes, plans, reports, READMEs, and all other project documentation.

- Lead with the main point.
- Name the actor and action, and use concrete verbs.
- Explain each technical term when it first appears.
- Put one main idea in each sentence.
- Write natural Vietnamese instead of translating English word for word.
- Make instructions concrete enough that the reader knows what to do.
- Preserve important conditions, limitations, and uncertainty.
- Prefer clarity over elegance and accuracy over oversimplification.

## Research Workflow Notes
Follow the logging and planning conventions in `codebase_preferences.md`, including experiment tracking with Weights and Biases and the planned use of data version control for augmented datasets.
All contributors and agents must strictly follow every requirement in `codebase_preferences.md`.
Before any large benchmark run, first execute the full experiment flow on 1 concrete combination from the development specification. If time allows, you may expand that first pass to 3 combinations. For this project, default to 1 combination only unless the user explicitly asks otherwise. Only after those first-end-to-end checks pass should you run all combinations.
When writing CLI commands, do not rely on `--help` as the only source for defaults. Prefer the code parser, config schema, and explicit default assignments in source. Always state which arguments are intentional defaults versus explicit overrides.

## Specification Terminology Consistency
- Before writing, reviewing, or implementing a new experiment specification version, compare it with every relevant earlier version.
- Build an explicit object-name mapping across versions. Mark each object as unchanged, renamed, split, merged, newly introduced, or deprecated.
- Use one canonical name for one runtime object across specification versions. If a rename is necessary, document the old name, new name, semantic equivalence, ownership, migration boundary, and affected source/config/artifact fields.
- Similar names are not evidence that two objects are equivalent. Compare their schema, API, stored data, lifecycle, owner, callers, runtime decisions, and checkpoint contract.
- If identity remains ambiguous, stop before implementation and ask the human developer to clarify. Never silently map names such as `TTLBuffer` and `VerificationBuffer` to each other.
- Every new specification version must include a terminology-change section or explicitly state that no object names changed.

## SSH Safety Note
- When using SSH on a shared GPU server, keep the session read-only unless the user explicitly asks for writes.
- Before cleanup, stop only the exact jobs you started and verify the target paths first.
- Do not remove broad output trees or artifacts unless the user clearly asked for it.
- Before leaving the remote host, confirm that only the intended logs, processes, and temporary files were touched.

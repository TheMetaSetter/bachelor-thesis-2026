# Repository Guidelines
Respond in English by default.

On macos environment, please use `realpath` command to get the full file path everytime you cite a file or a folder. File paths may contain language-specific unicode characters (e.g., Vietnamese) so you need to be extra careful with file paths while reasoning and also when respond to me.

Responded file paths containing Vietnamese must be valid in NFC Unicode normalization. Please check using very quick Python command in CLI.

All contributors and agents must strictly follow every requirement in `codebase_preferences.md`.
The single source of truth for this codebase is the `documents/` directory.

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
- Use `.venv/bin/python` to run code regularly instead of only writing code.

## Coding Style and Naming Conventions
- Follow `python_semantics.md` whenever writing, reviewing, or refactoring Python code, especially for slicing, indexing, assignment, and mutability.
- Keep model logic self-contained in a single file. Avoid fragmented class hierarchies.
- Prefer adapter-style wrappers when integrating reference codebases.

## Simplicity-First Requirements
- Reconsider a proposed mechanism when the runtime's natural behavior already enforces the intended result.
For example, if a point stops changing because later sliding windows no longer contain it, do not add a separate point-finalization mechanism.

## Testing Guidelines
- Expected coverage includes data loader shapes, one forward and backward pass, checkpoint save and load, and synthetic anomaly injection behavior as described in `codebase_preferences.md`.
- Tests should be minimal and focused while also being extra skeptical.

## Plain Language Requirements for Documentation
- Prioritize using simple words and minimal sentences, paragraphs when respond to user.

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

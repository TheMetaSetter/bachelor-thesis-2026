# Cross-codebase terminology audit plan

Goal: check whether bachelor-thesis-2026 and tsad-lib use precise, consistent wording for online test-time adaptation.

## Phase 1: search both codebases

- [x] Search specs, docs, source, configs, scripts, and tests for relevant term variants. Tools: `rg --files`, `rg`.
- [x] Read each relevant authoritative spec, current note, and runtime interface in context; distinguish prose, API identifiers, experiment labels, and separate behavior. Tools: `sed`, `rg`.

## Phase 2: report findings

- [x] Separate actual terminology confusion from valid code/API names and distinct runtime behavior. Tool: Markdown evidence table.
- [x] State findings, evidence, and scope limits for both codebases. Tool: Markdown review. No code/spec behavior changed; no tests run.

# Codex Cleandocs Hook Implementation Plan

> Inline implementation, following the approved design in `documents/spec/codex-cleandocs-hook-design.md`.

**Goal:** Add an opt-in Codex SessionStart hook that reports each repository's documentation tree and date-organized key-file context.

**Architecture:** Keep the Python hook in `scripts/cleandocs_hook.py`; keep local hook registration and settings under ignored `.codex/`. It reads `.codex/cleandocs.toml`, discovers the documentation root, invokes `tree`, writes a report to the system temporary folder, and emits a short plain-English summary. It never changes source documents.

**Tech Stack:** Python 3 standard library, Codex `hooks.json`, `tree`, TOML-like one-key opt-in file.

**Spec:** `documents/spec/codex-cleandocs-hook-design.md`

## Global Constraints

- Run only in `bachelor-thesis-2026` and `tsad-lib` initially.
- `.codex/` must be ignored by each repository's `.gitignore`.
- Report-only; never delete or edit scanned documents.
- Discover `documents/` or configured alternate names such as `docs/`.
- Use `tree` to list all files before summaries.
- Use simple English and highlight the latest dated folder.

## Tasks

- [x] Add failing tests for opt-in behavior, documentation-root discovery, tree-first report order, and dated-folder summaries.
- [x] Implement the Python hook and confirm those tests pass.
- [x] Add ignored `.codex/` config, hook registration, and `cleandocs = "on"` opt-in to both repos.
- [x] Run the hook in both repos and a temporary repo with a differently named docs folder; inspect reports and verify source documents are unchanged.
- [x] Review Git status and confirm `.codex/` stays ignored in both repos.

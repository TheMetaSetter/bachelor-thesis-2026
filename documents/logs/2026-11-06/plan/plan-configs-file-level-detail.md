---
date: 2026-11-06 17:49 Asia/Ho_Chi_Minh
planner: OpenAI Codex
topic: "File-level design for next-version configs/"
status: ready
revision: 3041e44b
branch: simplifying
related_research: none
---

# Implementation Plan: file-level design for `configs/`

## Summary

Add a file-level proposal for `configs/` to `proposed-next-version-tree.md`. First map current YAML values to their real consumers and compare earlier config guidance. Do not move, rename, or delete current configs while preparing the design.

## Request

Plan the file-level detail under `configs/` in the proposed next-version tree. Keep the design simple and consistent with the four project protocol contexts. Distinguish settings stored in YAML from validation rules implemented in source.

## Current state

1. The current `configs/` tree has 808 YAML files in data, debug, experiment, generated, model, protocol, and task folders.
2. The proposed tree names four future groups: `data/`, `methods/`, `protocols/`, and `experiments/`.
3. The current protocol checks are spread across config, data, scoring, evaluation, and update code. A protocol YAML file may hold values used by checks; it is not automatically the rule implementation.
4. An earlier config guideline defines a purpose-first layout, naming pattern, and metadata for experiment files. Its fit with the next-version tree needs review.

## Desired end state

The proposed tree lists each canonical config file or reusable file pattern, states the values it owns, and shows how an experiment refers to data, method, and protocol settings. Every current config category has a supported destination or a recorded open question. No config migration is part of this plan.

## Scope

### In scope

- Read-only inventory of current YAML files, key structures, references, and consumers.
- A terminology and ownership map from current config groups to proposed groups.
- File-level design entries under `configs/` in the proposed tree.

### Out of scope

- Editing, moving, renaming, or deleting YAML files.
- Changing runtime code, config schemas, defaults, or validation behavior.
- Running experiments or claiming that the proposal is implemented.

## Evidence

1. `configs/` contains 808 YAML files across seven top-level groups, counted with `find` and `wc`.
2. `documents/logs/2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md:18-22` names the proposed config groups.
3. `documents/specs/current-protocol-validation-rules.md:3-5` says current checks are distributed and groups them by four contexts.
4. `documents/abstract-design-notes/experiment_config_organization_guideline.md:7-24` defines earlier config organization and naming guidance.
5. `AGENTS.md` and `codebase_preferences.md` require evidence-based terminology mapping and say `documents/` is the documentation source of truth.

## Implementation approach

Use the smallest structure that preserves current meanings. Trace YAML keys into their loaders and consumers before assigning file owners. Treat `data`, `model`, `task`, `protocol`, and run-definition settings as separate concepts until evidence supports mapping or merging them. Keep fixed validation rules in source; include a YAML value only when the runtime reads it as a setting.

## Phase 1: inventory files and schemas

### Goal

Know what the current YAML files contain and which files share a structure.

### Changes

#### 1. List the current config tree

- **File:** `configs/` (read only)
- **Symbol:** YAML file paths and directory groups
- **Change:** Record the full file list, folder list, counts, and extensions.
- **Reason:** The future tree must account for current config kinds.
- **Dependencies:** None.
- **Tools:** `tree`, `rg --files`, shell counts.

#### 2. Summarize each YAML structure

- **File:** every `configs/**/*.yaml` file (read only)
- **Symbol:** top-level keys and config references
- **Change:** Parse each file and record its keys, references, and repeated structure.
- **Reason:** Filenames alone do not establish a config's purpose.
- **Dependencies:** Step 1.
- **Tools:** `.venv/bin/python`, PyYAML, a read-only script.

### Verification

- [ ] The inventory accounts for all 808 YAML files.
- [ ] Parse failures and duplicate-key failures are listed without repairing the files.

## Phase 2: trace meaning and ownership

### Goal

Map each config and field to the runtime behavior that reads or validates it.

### Changes

#### 1. Trace config entry points

- **File:** `src/core/config.py`, `src/core/config_experiment_validation.py`, config-loading CLI and run entry points
- **Symbol:** loaders, merges, defaults, validators, and config consumers
- **Change:** Record how each config is loaded and which source symbols read its fields.
- **Reason:** Place a field with its actual owner, not by its current filename.
- **Dependencies:** Phase 1 inventory.
- **Tools:** `rg`, source reading, Python call tracing.

#### 2. Assign each current config group

- **File:** current config files and proposed config groups
- **Symbol:** `data`, `model`, `task`, `protocol`, run definitions, generated settings
- **Change:** Map each file and field to a proposed owner, or mark it unresolved with evidence needed.
- **Reason:** Some current groups may need to split across method, protocol, and run settings.
- **Dependencies:** Step 1 and YAML key summary.
- **Tools:** Python mapping table, `rg`, source call map.

#### 3. Check setting duplication

- **File:** YAML files with repeated keys or values
- **Symbol:** repeated dataset, method, protocol, and run fields
- **Change:** Mark each repeated value as shared, method-specific, protocol-specific, or run-specific; record conflicts and defaults.
- **Reason:** Avoid keeping one value in multiple future owners.
- **Dependencies:** Step 2.
- **Tools:** Python YAML comparison, `rg`.

#### 4. Compare earlier config guidance

- **File:** `documents/abstract-design-notes/experiment_config_organization_guideline.md` and any other relevant earlier config specifications
- **Symbol:** config groups, metadata, naming, and run identity
- **Change:** Write an object-name map with old name, proposed name, status, runtime owner, and evidence.
- **Reason:** Preserve meaning across the tree update and follow the repository's terminology rule.
- **Dependencies:** Steps 1-3.
- **Tools:** Markdown reading, `rg`, Git history if needed.

### Verification

- [ ] Each mapped field has a source consumer or an explicit unresolved status.
- [ ] The map marks names as unchanged, renamed, split, merged, new, or deprecated.
- [ ] Any unclear identity remains open; no names are merged by guesswork.

## Phase 3: draft the file-level tree

### Goal

Describe a small future config layout with clear ownership and references.

### Changes

#### 1. Choose canonical files or patterns

- **File:** `configs/data/`, `configs/methods/`, `configs/protocols/`, `configs/experiments/`
- **Symbol:** proposed YAML filenames or reusable filename patterns
- **Change:** Select only the file set supported by the ownership map.
- **Reason:** Avoid a separate file for every run when shared settings can be referenced safely.
- **Dependencies:** Phase 2 ownership and terminology maps.
- **Tools:** ownership table, current schemas, Markdown outline.

#### 2. Define each file's contract

- **File:** each proposed config file or pattern
- **Symbol:** owned fields and references to other config files
- **Change:** State what values the file owns and how an experiment resolves them.
- **Reason:** Readers should see one owner for each value and a clear path to a resolved run.
- **Dependencies:** Step 1.
- **Tools:** YAML key map, source consumer map, Markdown editor.

#### 3. Resolve special config groups

- **File:** current debug, generated, smoke, demo, and historical config paths
- **Symbol:** active, generated, temporary, or retained config roles
- **Change:** Place each group in the future tree or document why its destination is unresolved.
- **Reason:** Do not silently drop current config use cases.
- **Dependencies:** Phase 1 classification and Phase 2 caller map.
- **Tools:** `rg`, source reading, version history.

### Verification

- [ ] Each proposed config has one purpose, one owner, and a known consumer.
- [ ] The tree separates validation rules in source from values configured in YAML.
- [ ] The file set preserves required research settings without copying run-specific values into shared files.

## Phase 4: update and review the tree proposal

### Goal

Add the verified file-level entries to the existing proposal without changing current configs.

### Changes

#### 1. Update the `configs/` section

- **File:** `documents/logs/2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md`
- **Symbol:** proposed `configs/` tree and config ownership notes
- **Change:** Add filenames or patterns, short purpose comments, and references between files.
- **Reason:** Make the next-version config layout reviewable at file level.
- **Dependencies:** Phases 1-3.
- **Tools:** Markdown editor, Git diff.

#### 2. Record terminology changes

- **File:** `documents/logs/2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md`
- **Symbol:** terminology-change note
- **Change:** Include the approved name map, or state that no config object names changed.
- **Reason:** Keep old and proposed config meanings explicit.
- **Dependencies:** Phase 2 map.
- **Tools:** Markdown editor, Git diff.

### Verification

- [ ] `git diff --check` reports no whitespace errors.
- [ ] `rg` confirms the new plan and tree text use allowed terminology.
- [ ] The final tree agrees with `documents/specs/definitions.md` and `documents/specs/current-protocol-validation-rules.md`.
- [ ] No current YAML or source file changed.

## Testing strategy

This task changes documentation only. Use YAML parsing during inventory as a read-only check of current files. Do not run the experiment test suite for a tree proposal.

## Migration and rollback

No config migration is planned. Revert only the documentation diff if review rejects the proposed layout; leave all current configs intact.

## Assumptions and non-blocking uncertainties

1. The four proposed groups remain `data/`, `methods/`, `protocols/`, and `experiments/`.
2. The meaning of each current `task/` and `protocol/` YAML must be traced before assigning it to a future group.
3. It is not yet known which current run files are generated, hand-maintained, or still called by active entry points.

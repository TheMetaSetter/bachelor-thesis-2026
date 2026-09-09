---
date: 2026-09-09 21:35:18 +07:00
topic: "Apply terminology ontology naming conventions to smoke W&B run names and identity fields"
status: approved
revision: ed10fbb4db70f9ed17f98330c18f6e456ba5f2f5
related_documents:
  - documents/logs/09-09-2026/research/research-terminology-ontology-naming-code-modifications.md
  - documents/notes/terminology_ontology_naming_conventions.md
  - documents/spec/offline_pretraining_terminology_ontology.md
  - documents/spec/online_tta_terminology_ontology.md
---

# Implementation Structure: Apply terminology ontology naming conventions to smoke W&B run names and identity fields

## Summary

The implementation will centralize smoke display-name construction, expose canonical identity fields, update all active smoke generators and runtime callers, regenerate derived YAML, and verify the result across the full smoke inventory.

The phase order is contract first, source integration second, regeneration third, and verification last.

The user explicitly requested the complete research, plan, structure, and detail sequence in one instruction, so this structure is marked approved for detailed expansion.

## Request

Use `smk-<phase>-<method_or_variant>-<entity>-s<seed>` for smoke W&B display names.

Preserve canonical ontology names and full experiment identity outside the short W&B display name.

Do not implement code in this planning task.

## Confirmed context

- `offline_pretraining_phase` and `online_tta_phase` are canonical phase identifiers.
- `offline_variant` uses `O0` or `O1`.
- `online_variant` uses `A0`, `A1`, or `A2`.
- `KA` is the explicitly selected display token for `kmeans_ad`.
- 508 smoke configs currently have W&B enabled but 0 use the `smk-` prefix.
- Several runtime callers overwrite configured names with the older generic helper.
- `experiment_variant` is protocol detail and does not replace `offline_variant`.

## Scope

### In scope

- Canonical identity fields used by smoke naming.
- One smoke W&B display-name composition path.
- Benchmark generators and special smoke entry points.
- Runtime preservation of configured names.
- Derived smoke configuration regeneration.
- Naming and inventory tests.

### Out of scope

- Checkpoint, artifact, metric, output-path, or threshold semantics.
- Historical W&B run renaming.
- Broad online runtime compatibility-field renaming.
- Unapproved method abbreviation design.

## Proposed phases

### Phase 1: Establish canonical identity and the smoke display-name contract

**Result:** The codebase has one explicit, validated smoke display-name contract and canonical offline or online identity fields are available.

**Stages:**

1. **Stage 1.1: Confirm ontology mappings.** Record phase tokens, variant tokens, the selected `KA` method token, and the rule that full identity remains outside the display name.
2. **Stage 1.2: Expose canonical identity fields.** Update config validation and generators so `offline_variant` and `online_variant` are not silently inferred from `experiment_variant` or filename text.
3. **Stage 1.3: Add the display-name builder.** Compose and validate the exact `smk-` grammar from explicit inputs.

**Depends on:** Current ontology files, naming note, and research findings.

**Verification:** Unit tests assert all three selected examples and reject missing identity inputs.

**Risks:** Ambiguous identity could create a valid-looking but incorrect short name.

**Complete when:** The helper accepts only explicit canonical identity and the tests prove exact output.

### Phase 2: Route generators and runtime callers through the contract

**Result:** Every active smoke generator emits the selected name and runtime callers preserve it.

**Stages:**

1. **Stage 2.1: Update THESIS generators.** Cover offline THESIS, benchmark-smoke THESIS, and online THESIS generators.
2. **Stage 2.2: Update baseline generators.** Cover traditional offline and online streaming baseline generators.
3. **Stage 2.3: Update the remaining-SMD matrix generator.** Separate short display names from the generic `run_id` used for full experiment identity.
4. **Stage 2.4: Update runtime callers.** Stop replacing a valid configured smoke name with the generic artifact-style helper.
5. **Stage 2.5: Update special smoke entry points.** Cover direct-branch-routing smoke and runtime fallback paths.

**Depends on:** Phase 1 helper and identity fields.

**Verification:** Generator tests and fake-W&B logger tests observe the exact selected names.

**Risks:** One unupdated caller can reintroduce a legacy name after config generation.

**Complete when:** Every named source either calls the helper or preserves a valid configured smoke name.

### Phase 3: Regenerate derived smoke configurations and preserve provenance

**Result:** Tracked smoke YAML files agree with the source generators and retain full experiment identity.

**Stages:**

1. **Stage 3.1: Regenerate source-owned YAML.** Run the existing generator entry points.
2. **Stage 3.2: Check canonical fields.** Confirm phase, variant, method, entity, seed, stage, and protocol identity remain available.
3. **Stage 3.3: Check W&B metadata.** Confirm tags and job type retain details omitted from the display name.

**Depends on:** Phase 2 source changes.

**Verification:** Full smoke inventory loads and matches the grammar.

**Risks:** Hand-editing generated YAML can create generator drift.

**Complete when:** Regeneration is reproducible and no smoke YAML has a legacy name.

### Phase 4: Verify compatibility and finalize the change

**Result:** The smoke naming change is tested without changing local artifact paths or runtime compatibility fields.

**Stages:**

1. **Stage 4.1: Extend unit and generator tests.** Add exact examples and negative cases.
2. **Stage 4.2: Extend inventory and runtime tests.** Test all smoke files and the final `wandb.init` boundary.
3. **Stage 4.3: Run focused verification.** Run existing tests, config loading, and `git diff --check`.

**Depends on:** Phases 1 through 3.

**Verification:** All focused tests pass and the inventory contains only compliant smoke names.

**Risks:** A broad naming refactor could break historical readers.

**Complete when:** Only the W&B display-name surface changes and all canonical local identities remain stable.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 1 | Ontology and current helper | Deterministic smoke name construction |
| Phase 2 | Phase 1 contract | Consistent generator and runtime behavior |
| Phase 3 | Phase 2 source changes | Consistent tracked YAML |
| Phase 4 | Phases 1-3 | Evidence-backed completion |

## Decisions confirmed

- `off` and `on` are display tokens, not replacements for canonical phase identifiers.
- `O0`, `O1`, `A0`, `A1`, and `A2` retain their ontology meanings.
- `KA` is a display token for `kmeans_ad`, not a new ontology object.
- `KA-A2` is a combined display sequence and not a new `online_variant` value.
- Stage, checkpoint, window, protocol, and artifact identity remain outside the short display name.
- Generated YAML must be updated through generator sources.

## Non-blocking uncertainties

The user has not selected abbreviated tokens for methods other than `kmeans_ad`.

The detailed plan therefore preserves existing method tokens for those methods until a new mapping is explicitly selected.

The legacy `previous_ewma_point_scores` state parameter is recorded as a separate ontology mismatch and is not included in this W&B naming change.

## Feedback requested

The structure follows the explicitly requested complete workflow and can be reviewed before implementation begins.


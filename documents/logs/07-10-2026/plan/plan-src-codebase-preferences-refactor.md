# `src/` codebase-preferences refactor plan

## Goal

Make every owned Python source file stay within 500 lines and every function or
method within 50 lines, while preserving the current experiment, tensor,
checkpoint, and configuration contracts.

## Implementation order

1. Establish deterministic test discovery and repair stale public imports.
2. Extract reusable neural primitives so model files do not depend on each
   other.
3. Split large runtime, configuration, augmentation, and metrics files by one
   responsibility per helper.
4. Add an AST contract test only after all current violations are removed.

## Invariants

- Public registry names, YAML keys, output dictionaries, checkpoint keys, and
  seeded synthetic augmentation behaviour remain unchanged.
- The thesis model has one public entrypoint; helpers cannot be lifecycle
  mixins.
- Reference mechanisms remain covered by existing loader, injection, and
  prototype characterization tests.

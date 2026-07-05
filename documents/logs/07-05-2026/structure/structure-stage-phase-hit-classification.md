# Stage/Phase Semantic Rename Structure

**Objective:** Chuyển kế hoạch đổi tên `phase` và `stage` thành một cấu trúc thi công nhỏ, rõ ràng, và an toàn để codebase dạy đúng một nghĩa cho mỗi thuật ngữ.

**Scope:** Chỉ xử lý semantic drift giữa active two-stage, legacy three-stage, và runtime `stage_name`. Không mở rộng sang thay đổi behavior, không đổi public schema trừ khi một migration riêng được tách ra sau này.

**Core Constraint:** `offline pre-training` là phase lớn. Stage A/B là stage con trong phase đó. Legacy three-stage là compatibility hoặc historical context. Runtime `stage_name` như `train`, `val`, `test` là một lớp nghĩa khác, không được gộp với phase/stage taxonomy của offline pre-training.

---

## Overview

This refactoring stream should make the repository more legible without altering the active two-stage rerun contract. The implementation must preserve compatibility where it is already part of the repository contract, while reducing the number of places where one term carries multiple meanings.

---

## Implementation Phases

1. **Terminology lock and compatibility boundary**
   - Write a short terminology block into the active SSOT design note so that readers see the intended meaning of `phase`, `stage`, and `stage_name` before they reach the code.
   - Keep the distinction explicit between the active two-stage rerun, the legacy three-stage path, and ordinary runtime step naming.
   - Preserve the minimal vertical slice principle by changing only the documentation boundary first.

2. **Stage-first internal naming in the active two-stage runner**
   - Rename internal variables and comments in the two-stage runner so that the file reads stage-first instead of phase-first.
   - Keep public manifest keys and compatibility-shaped schema fields unchanged in this phase.
   - Preserve the batch contract and experiment contract by making only reader-facing internal names more precise.

3. **Active two-stage contract clarity in model helpers**
   - Update helper names and comments in the model state and setup mixins so they describe Stage A/B semantics directly.
   - Keep the model output contract stable and avoid changing serialization behavior.
   - Use composition-like clarity at the helper boundary: each helper should expose one meaning and one responsibility.

4. **Legacy three-stage fenced compatibility**
   - Mark legacy three-stage codepaths as historical or compatibility-only in comments and docstrings.
   - Keep old config aliases and legacy tests intact where they are still required for backward compatibility.
   - Make the boundary between active two-stage and legacy three-stage explicit so readers do not infer a single shared contract.

5. **Preserve ordinary runtime `stage_name` usage**
   - Leave runtime `stage_name` unchanged where it means execution splits such as `train`, `val`, `val_synth`, or `test`.
   - Review these occurrences only to confirm that they are not part of the offline pre-training taxonomy.
   - Do not rename them for symmetry, because that would reduce clarity rather than improve it.

6. **Tests and docs teach the same meaning**
   - Rename tests and helper names where they still teach the wrong meaning.
   - Update research notes and design notes so the same term always points to the same runtime concept.
   - Validate the final state with a grep-based classification pass that separates the three groups cleanly.

---

## Design Principles Preserved

- **Minimalistic:** make the smallest meaningful rename set first.
- **Easy to comprehend:** prefer names that reveal the contract directly.
- **Single-meaning:** one term, one runtime concept, one contract.
- **Separation of concerns:** keep active two-stage, legacy three-stage, and runtime naming separated.
- **Stable interfaces:** preserve public schema until a dedicated migration is planned.
- **Compatibility-first:** keep historical support explicit instead of silently blending it into the active path.

---

## Expected Validation Shape

- A terminology block in the SSOT doc.
- Stage-first internal naming in the active two-stage runner.
- Clear comments in model helpers.
- Explicit legacy markers in three-stage codepaths.
- No unnecessary renames in runtime `stage_name` files.
- Clean test and doc wording aligned with the same semantic buckets.

---

## Next Step

If this outline is accepted, the detailed implementation phase should split into one rename pass per semantic bucket, with a grep-based review after each pass.


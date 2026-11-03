# Lessons from the failed simplification attempt

## Purpose

Use this note before another large simplification of this research library. It records this session, including mistakes and limits. It is not a claim that the earlier rewrite is safe to reuse.

Writing checklist:

- [x] Read the simplification and restoration reports.
- [x] Record the scope errors, useful tactics, and evidence limits.
- [x] Write a safer plan for the next attempt.
- [x] Check this note against the session records.

## What the user wanted

A much smaller core codebase with fewer files, fewer folders, and fewer lines. Researchers, students, and general readers should find it easier to follow. Start with a file-level directory tree and execute a written checklist in order. Make decisions quickly and independently.

The requested targets grew to about 80% fewer files and at least 90% fewer core lines. Mathematical behavior had to stay the same throughout. Third-party code was outside the reduction target. Later instructions asked for fewer agents, faster decisions, and an end to the work.

Autonomy allowed implementation choices. It did not make scientific methods disposable. When a size target conflicts with preservation, report the gap. Do not quietly change the scope to make the percentage look better.

## What failed

### 1. I reduced the wrong things

I deleted files in `bsc-thesis-ref-codebases` and other research assets during the broad reduction pass. Those deletions did not simplify the owned core. They were restored. All 1,170 tracked reference files were checked against the original revision at that point.

I also removed scientific baselines from the core: IForest, KMeans-AD, STUMPY, MOMENT, CANDI, M2N2, and other online baseline paths. Calling these external adapters did not make them unnecessary. They were part of the library's scientific scope.

Keeping native THESIS and RedLamp numerical behavior was not enough. The requirement applied to the library, including comparison methods and their runnable paths.

### 2. I changed experiment behavior too

I retired campaign entrypoints, scheduler automation, checkpoint-selection paths, and related tests. Some operational code may be removable, but changing how an experiment trains, chooses a checkpoint, calibrates scores, or resumes can change its scientific result.

A smaller training script is not equivalent merely because it can run one optimizer step. Selecting minimum validation loss does not preserve an older selection rule based on another metric.

### 3. I treated passing reduced tests as too much evidence

The original recorded run had 641 passed and 2 skipped. The reduced suite had 160 passed and 1 skipped after many interfaces and tests were removed. These counts describe different coverage.

Deleting a feature and its tests can make the remaining suite pass while reducing the library's behavior. Test removal must be reviewed separately from code removal.

### 4. I optimized size before fixing the preservation boundary

The attempt reported core lines falling from 47,514 to 13,732: 71.1%, below the 90% target. That historical count included blank lines and comments, as well as wrong feature removals, and has been withdrawn after rollback. Do not compare it with the corrected code-only baseline.

The target was a goal, not evidence that every required method could fit within 4,751 lines. A large numerical reduction does not by itself show a better library.

### 5. File moves had hidden risks

Promoted source and tests matched broad ignore rules. Required files could exist locally but be absent from a future commit. Ignore exceptions were added after review found this.

Flattening configurations lost clear ownership. The separator `__` also appeared inside original filenames. Replacing every `__` with `/` during recovery produced incorrect paths. Those ambiguous copies were moved to the backup instead of being treated as restored originals.

Git restores tracked files. It does not restore every ignored or untracked local configuration. One such missing data configuration broke the restored suite and had to be recovered from the backup.

### 6. Reporting became stale

Earlier removal lists and verification results no longer described the code after assets were restored or the core was rolled back. They needed explicit notices that they were superseded.

A final report must describe the current tree. Historical test results and reduction figures must stay clearly labeled as historical.

## What helped, with limits

| Tactic | What it checked | Limit |
|---|---|---|
| Fixed original Git revision | Exact tracked-file restoration | Does not cover all local ignored files |
| Backup before rollback | Preserved the current attempt and local files | A temporary backup is not durable project history |
| Before/after numerical snapshots | Compared outputs, losses, gradients, state, and Adam updates | Covered selected native cases only |
| Exact seeded injection checks | Compared generated tensors, metadata, and random draw progression | Covered the chosen cases only |
| Split/resume stream comparison | Compared resumed and uninterrupted online state and records | One variant does not cover every online method |
| Small training/checkpoint test | Checked weight changes and checkpoint restoration | Not a full real-data experiment |
| Full test suite | Found missing dependencies and entrypoints | Passing tests do not prove every scientific path |
| Ignore-rule review | Found required files excluded from Git | Must be repeated for newly moved files |

The native numerical comparison recorded 1,835 exactly equal tensors across 11 cases. Timing fields were excluded. Injection checks recorded 144 exact seeded cases. These were useful checks of retained computations. They did not justify deleting other methods.

The comparison harness and snapshots were kept under temporary paths. A future attempt should retain a small reusable comparison tool and a clear record of inputs, seeds, versions, and exclusions.

Separate data, model, and online tasks helped local work, but integration remained a single shared problem. More agents also added coordination cost. Use one owner by default. Delegate only when permitted and when independent work has a clear benefit.

## Additional source: method-specific mathematics

Before simplifying a scientific method, consult the method notes in
`/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/method-mathematics/`
when that method is covered there. Its `checklist.md` requires a source-ordered
review, recording the method flow and assumptions, and tracking unresolved
sources. At the time of this review, 34 of 36 entries were checked; Matcha and
AFMNF/DRAF remained incomplete.

A checked source review is not proof of implementation parity. The checklist
explicitly marks many adapters as unverified or parity-untested. Keep these
claims separate:

- What the paper or thesis method computes.
- What the current adapter computes, including its input, preprocessing,
  target, score, calibration, and update order.
- What numerical comparisons have actually verified.

Use the method notes to define the mathematical behavior that a refactor must
preserve. Compare that behavior with the current source and its tests before
changing code. If a source, adaptation, or behavior branch is incomplete or
ambiguous, record it as unresolved and do not claim equivalence. This catalog
covers more methods than the current SMD matrix; it is a mathematical reference,
not automatic evidence that every catalog entry is in this repository's active
experiment scope.

For methods in both sources, carry the documented gaps forward: KMeansAD's
paper flow does not define this experiment's windowing or validation
calibration; the STUMPY channel-AB adaptation, RedLamp soft-label formula,
M2N2 adaptation, and CANDI threshold convention still need implementation-level
checks. Their existing method paths remain in scope while those checks are open.

## Retrieval wording and review requirements (2026-11-03)

Use **no retrieval / dense retrieval / sparse retrieval** for the proposed comparison. No retrieval passes encoder features directly to the task head. Dense retrieval corresponds to the **continuous attentional prototype bank**; sparse retrieval corresponds to the **discrete attentional codebook**. Both banks can contain real-valued vectors. Weighted top-k can return a continuous mixture; hard top-1 is a special case. Use “discrete attentional codebook,” not “anomaly codebook”; class-stratified contents can include normal patterns.

Review in this order:

1. Read the relevant specifications and map names to the owning runtime objects. Inspect the configured selection, weighting, normalization, temperature, Gumbel hard/soft behavior, gradients, and task routing. Similar names do not establish equivalent computations.
2. Record specified behavior, observed implementation, and numerical parity separately, with source revision, configuration, evidence, and unresolved gaps. A source-reading checkbox or passing reduced suite does not verify equivalence.
3. Check whether an ablation isolates retrieval. Match encoder, memory entries and size, similarity, normalization, classifier, examples, and tuning budget. Treat initialization pools, labels, losses, and routing as separate factors; distinguish frozen probes from end-to-end training.
4. State the classification target and label granularity. Separate familiar synthetic generators from unseen-family transfer. Check split leakage, paired comparisons, uncertainty, and actual test FPR. A synthetic classification gain does not establish real-anomaly detection quality.
5. Verify each mathematical claim's assumptions. Weighted top-k is not a finite hard code index; normalization and query bypasses affect bounds. Lower distortion, reconstruction error, or a retrieval-error bound does not establish lower classification error. Sparse support alone does not establish a speed gain.
6. Verify primary sources and publication status. Keep submission, workshop, and accepted main-conference claims distinct; unavailable evidence remains unresolved.

**Terminology-change statement:** Existing code, configuration, checkpoint, metric, and artifact names remain unchanged. The proposed comparison is not an approved change to the MUSTARD matrix or evidence of an implemented dense/sparse distinction. Preserve distinct retrieval mechanisms when simplifying. See the [two-stage specification](../../../spec/two-stage-offline-pretraining-spec.md#retrieval-wording-and-review-2026-11-03) and [implemented-flow note](../../../notes/experiment-flows-and-equations.md#retrieval-wording-and-review-2026-11-03).

## What I wish I had fixed before editing

1. A complete list of required methods, offline and online variants, and public entrypoints.
2. A written definition of preserved behavior: data splits, scaling, windows, losses, scores, thresholds, update order, random state, checkpoint selection, and resume.
3. A map from each method to its source, configs, loaders, runners, tests, and checkpoint format.
4. An inventory of tracked, ignored, untracked, and generated files, with a recoverable backup.
5. One stable counting rule for owned core code, separate from tests, configs, vendors, and research assets.
6. A small comparison case for each distinct method and behavior branch before changing it.
7. A proposed directory tree with one clear home for each runtime concept.
8. A stop rule: freeze edits, verify, and report unfinished work when time runs short.

## Rules for the next attempt

- Protect all scientific methods by default, including baseline wrappers.
- Keep method identity separate from ablation labels. `O0/O1/O2` and `A0/A1/A2` are THESIS (MUSTARD) ablations only; baseline profiles such as `main` or `reference_adapter_redlamp_encoder` are method-specific. Legacy baseline config filenames using A-labels do not define THESIS-like baseline ablations.
- Protect reference sources, specifications, research notes, datasets, and experiment artifacts.
- Simplify repeated wiring and wrappers before touching mathematical code.
- Merge files only when they belong to the same concept and remain easy to read.
- Keep each model's public training, inference, and checkpoint contract easy to follow.
- Preserve useful names and comments. Do not compress code to win a line count.
- Do not replace explicit contracts with hidden defaults, guessed paths, or broad fallback behavior.
- Remove an entrypoint only after mapping its callers and providing an equivalent path where required.
- Treat scheduler, checkpoint-selection, calibration, and online update changes as behavior changes.
- Do not delete tests just because the new code no longer supports what they test.
- Track file moves with an explicit old-path to new-path map. Do not infer paths from separators.
- Keep local work and original user edits safe. Do not use a broad reset or cleanup.
- Report a missed percentage honestly. Do not count vendor deletion, minification, archives, or lost features as successful core simplification.
- For the size counter, file count includes owned `.py` and `.sh` files with at least one code line after blank lines and comments are ignored. Line and byte totals exclude blank lines and comment text, including inline comments. Count UTF-8 bytes of retained source text, excluding line endings. Use this same rule for every baseline and comparison.

## Next attempt: execute in this order

### Phase 1: establish the boundary

- [ ] 1. Read repository guidance, relevant specifications, and applicable method-mathematics notes. Tools: file reads and `rg`.
- [ ] 2. Record the starting revision and existing edits. Tool: Git.
- [ ] 3. Inventory tracked, ignored, untracked, and generated files. Tools: Git and Python.
- [ ] 4. Save a recoverable backup of local files. Tool: Python or a checked archive.
- [ ] 5. List every required scientific method and its own variants or profile. Tool: a Markdown table that keeps THESIS/MUSTARD ablations separate from baseline method profiles.
- [ ] 6. Map each method to its runtime dependencies and callers. Tools: `rg` and source reads.
- [ ] 7. Define the size-counting rule and record the baseline. Tool: one reusable Python counter; count files with code after blank lines and comments are ignored, and exclude blank lines and comment text from line and byte totals.

### Phase 2: design and capture behavior

- [ ] 8. Write the simpler directory tree down to individual files. Tool: Markdown.
- [ ] 9. Map every old path to its planned path or explain why it is redundant. Tool: a move table.
- [ ] 10. Check that every required method still has a complete runnable path. Tools: the method table and source reads.
- [ ] 11. Run the existing full suite and record failures without changing tests. Tool: `.venv/bin/python -m pytest`.
- [ ] 12. Capture seeded comparison cases for each method and distinct behavior branch. Tools: Python and the method's existing dependencies.

### Phase 3: simplify in small groups

- [ ] 13. Choose one group of repeated wrappers or wiring. Tools: the dependency map and source reads.
- [ ] 14. Move or rewrite only that group. Tools: file edits and an explicit path map.
- [ ] 15. Update its imports, configs, documentation, and test paths. Tools: file edits and `rg`.
- [ ] 16. Check that new files are visible to Git and expected old paths have replacements. Tool: Git.
- [ ] 17. Run focused tests and numerical comparisons. Tools: pytest and the comparison tool.
- [ ] 18. Run the full suite. Tool: pytest.
- [ ] 19. Review the diff for lost features, changed defaults, and removed assertions. Tools: Git and source reads.
- [ ] 20. Recount size using the same rule. Tool: the Python counter.
- [ ] 21. Record the completed group before returning to step 13 for another group. Tool: the Markdown checklist.

### Phase 4: finish

- [ ] 22. Run one small full real-data experiment through the changed path, including checkpoint save/load and resume where relevant. Tool: the supported local runner.
- [ ] 23. Verify that every required method is still present and its evidence status is clear. Tools: the method table and recorded checks.
- [ ] 24. Update the directory tree and mark old reports as superseded where needed. Tool: Markdown.
- [ ] 25. Report current size, passing checks, failures, untested paths, and remaining work. Tool: one final report.

Do not launch a full campaign to validate a refactor. Start with one concrete experiment. Stop changing code when asked to end; preserve the work and report the exact state.

## Current outcome

The core simplification was rolled back. Tracked files in `src/`, `scripts/`, `configs/`, `tests/`, and `demo/`, plus the restored README and architecture notes, matched original revision `753ad03ed60b2d41b078e53bf421429bb0bd6c0f` when checked during restoration. Existing user edits to `AGENTS.md` and `.gitignore` were preserved.

The final restored test run recorded 508 passed and 2 skipped. This count differs from the earlier 641-pass record. The session did not establish the complete reason for that difference. Do not claim that every original local test was recovered from Git alone.

No full real-data experiment was run. Ambiguous flattened experiment configurations remain in the backup. The earlier percentage reductions are withdrawn. A next attempt starts from preservation of the whole scientific scope.

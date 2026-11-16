# Simplicity audit: proposed tree and desired CLI

Date: 2026-11-09  
Scope: static comparison of `documents/specs/desired-cli-20261109.md` and `documents/logs/2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md`. Both describe proposals, not confirmed runtime behavior. No source code or config files were changed.

## Main findings

1. **Conflicting run generation.** The desired CLI generates applicable method–dataset combinations; the tree permits only seed expansion of explicitly listed templates. Design one experiment expansion flow for both method–dataset combinations and shared seeds. Applicability, defaults, and how skipped combinations are shown remain open.
2. **Experiment management has no clear owner.** The CLI can create, list, inspect, and change experiments, but the tree does not identify who persists experiment definitions, resolves names, or builds the run matrix. Assign those responsibilities to a small experiment module or modules; exact file names and signatures remain design choices.
3. **Execution scripts overlap proposed CLI/runtime responsibilities.** `run_offline.py`, `run_online.py`, `run_comparison.py`, and `run_ablation.py` sit beside `src/cli.py` and the offline/online runtime. Candidate simplification: expose one CLI flow and keep script wrappers only when an existing caller requires them. Preserve variant selection, phase ordering, and run behavior; no script is proven redundant without a caller and behavior map.
4. **Reporting ownership overlaps.** `collect_results.py` and `summarize_results.py` overlap the stated aggregation and CSV responsibilities of `evaluation/reporting.py`. Candidate simplification: give result lookup, aggregation, and table output clear owners behind the reporting command; retain distinct scripts only for demonstrated extra duties or compatibility.
5. **Config generation and smoke have unclear roles.** The tree proposes both `generate_configs.py` and explicit experiment YAML, plus `configs/experiments/smoke.yaml`; the CLI creates configurations and has `--smoke`. Decide whether these are generators, reusable examples, or wrappers before retaining them. A smoke run must not overwrite or contaminate full-run results; its exact selection and identity are open.
6. **Backup and run tracking lack an owner.** The CLI exposes W&B logging and Kaggle backup, but the tree gives neither a home. Specify who owns each integration and its config, failure behavior, and recovery. Their absence from the displayed tree is not proof the current project lacks them.
7. **Completion and lookup lack a contract.** Shell completion and name-based experiment commands require a source of names, name-to-config lookup, and a behavior for missing or duplicate names. Reuse a suitable existing registry or config owner if it fits; a new layer is not automatically necessary. Dataset download is future work and does not require an implementation file now.
8. **The CLI entry point is under-specified.** The desired command is `tsad`, but the tree only proposes `src/cli.py`; it does not state the installed command registration or its relationship to `tsad-lib`. Define those before implementation; do not assume the two projects use the same command implementation.

## Preserve while simplifying

1. Keep the four agreed validation-rule contexts and the distinction between settings and rules.
2. Keep all selected methods, MUSTARD variants, data paths, offline/online scoring behavior, calibration, checkpoint and resume contracts, reporting, artifacts, and retained research records. MOMENT and O2 readiness remain unverified.
3. Keep CLI entry points thin, but do not merge offline/online execution, checkpoint, calibration, and recovery simply to reduce the file count.
4. Compare config profiles, reference equations, entry points, callers, artifacts, and representative runs before retiring any actual script or claiming parity.

## Suggested next design step

Map each desired command to one proposed owner and the config, run, or artifact it reads or changes. Use that map to revise the tree and design signatures. This audit does not authorize an implementation or settle the unresolved defaults, download mechanism, compatibility wrappers, smoke behavior, or config migration.

# Phase 2 rerun: runnable-path audit

Date: 2026-11-03

## Scope

THESIS is the runtime name for MUSTARD. `O0/O1/O2` and `A0/A1/A2` are
ablation variants of THESIS/MUSTARD only. Baselines keep their own method
profiles. Legacy baseline YAMLs carrying A-labels are excluded from this
method-path mapping.

## Audit

| Method | Offline path | Online path | Evidence inspected | Status |
|---|---|---|---|---|
| THESIS/MUSTARD | Public model, trainer, two-stage runner; standard configs currently provide O0/O1 | Online adaptation model and causal TTA engine; standard configs provide O0/O1 × A0/A1/A2 | `src/models/thesis_multitask.py`, `src/engine/trainer.py`, benchmark scripts/configs, model/checkpoint and online variant tests | O0/O1 path present; O2 blocked |
| RedLamp | `RedLampBaseline`, comparative and CLI paths; profile `main` | Not a matrix method | `src/models/redlamp_baseline.py`, comparative runner, RedLamp tests/configs | Present |
| STUMPY Channel AB | Traditional offline scorer; profile `main` | Frozen online scorer; profile `main` | Offline/online builder maps, configs, STUMPY contract tests | Present |
| KMeansAD | Traditional offline scorer; profile `main` | Frozen online scorer; profile `main` | Offline/online builder maps, configs, KMeansAD contract tests | Present |
| Isolation Forest | Traditional offline scorer; profile `main` | Frozen online scorer; profile `main` | Offline/online builder maps, configs, Isolation Forest contract tests | Present |
| CANDI | Not in the SMD matrix | Streaming adapter; profile `reference_adapter_redlamp_encoder` | Online builder/profile map, configs, method-fidelity and streaming contract tests | Present |
| M2N2 | Not in the SMD matrix | Streaming adapter; profile `reference_adapter_redlamp_encoder` | Online builder/profile map, configs, method-fidelity and streaming contract tests | Present |
| MOMENT adapter | No matrix configuration | No matrix configuration | `src/adapters/moment.py` and adapter shape test | Out of matrix; not counted as a required runnable experiment path |

The current online generator uses `reference_adapter_redlamp_encoder` for
CANDI/M2N2 and `main` for STUMPY, KMeansAD, and Isolation Forest. Its baseline
variant names do not use THESIS A-label semantics. Old generated A-labeled
baseline configs remain legacy artifacts and are not evidence of baseline
ablations.

## Open blocker

The required THESIS O2 standard run path remains unresolved. The standard
offline generator enumerates O0/O1, the standard offline config directory has
no O2 config, and the separate direct-branch helper does not establish the
full Stage A → memory initialization → Stage B → evaluation lifecycle. Do not
infer O2 parity from that helper. Phase 2 Step 12 cannot cover the complete
THESIS matrix until this path and matching checkpoints exist.

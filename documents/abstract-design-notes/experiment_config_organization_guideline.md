# Experiment Config Organization Guideline

Date: 2026-05-31
Status: Active guideline
Applies to: `configs/experiment/`

## Objective
Standardize experiment configuration organization so that humans and AI agents can discover, filter, and maintain runs with minimal ambiguity and minimal grep cost.

## Folder Taxonomy
Use purpose-first grouping.

1. `configs/experiment/smoke/`
- Fast checks for pipeline integrity and runtime health.

2. `configs/experiment/baseline/`
- Canonical baseline runs used for comparison anchors.

3. `configs/experiment/ablation/`
- Controlled component toggles (`no_gate`, `no_usage`, `no_covariance`, and related variants).

4. `configs/experiment/scale/`
- Long/full runs, multi-seed sweeps, hardware-specific runtime presets.

5. `configs/experiment/thesis/exp1/`, `exp2/`, `exp3/`
- Thesis experiment phases aligned with report structure.

6. `configs/experiment/archive/`
- Deprecated or superseded presets retained temporarily for traceability.

## File Naming Contract
Use a stable, grep-friendly, double-underscore naming contract:

`<dataset>__<model>__<goal>__<window>__<seed>__<runtime>.yaml`

Examples:
- `smd__thesis_multitask__baseline__w20__seed11__rtx3090.yaml`
- `smd__thesis_multitask__ablation_no_gate__w20__seed11__rtx3090.yaml`
- `smd__redlamp_mlp__smoke__w20__seed11__cpu.yaml`

Notes:
- Keep tokens lowercase and explicit.
- Avoid hidden abbreviations unless they are already codebase-standard.
- Prefer fixed token order for deterministic grep behavior.

## Required YAML Header Metadata
Each experiment config must start with compact comment metadata.

```yaml
# group: baseline
# stage: exp2
# status: active
# owner: thesis_multitask
# tags: [smd, w20, seed11, rtx3090, val_realistic]
```

Field semantics:
- `group`: one of `smoke|baseline|ablation|scale|thesis|archive`.
- `stage`: thesis phase or `general`.
- `status`: `active|draft|deprecated|archived`.
- `owner`: primary model family or experiment owner key.
- `tags`: normalized searchable tokens.

## Search Patterns (Human + Agent)
Use these canonical commands:

```bash
rg "^# group: baseline" configs/experiment
rg "^# stage: exp2" configs/experiment
rg "tags:.*val_realistic" configs/experiment
rg "__ablation_" configs/experiment
rg "__seed11__" configs/experiment
```

## Migration Rules
1. New files must follow folder taxonomy and naming contract.
2. Existing files should be migrated incrementally.
3. During migration, preserve content semantics; only move/rename when verified.
4. If a config is no longer compatible with active task schema, move it to `archive/` or delete it after confirmation.
5. Keep monitor metric naming consistent with active validation namespace (`val_realistic_*`).

## AI-Agent Collaboration Rules
1. Agents should locate candidate configs by `group`, then `stage`, then `tags`.
2. Agents should never infer experiment purpose from filename fragments alone when header metadata is present.
3. Agents should update metadata comments whenever experiment semantics change.

## Acceptance Checklist
- Folder path matches purpose taxonomy.
- Filename matches token contract and order.
- Header metadata is present and valid.
- Monitor metrics align with active namespace.
- Config loads successfully through `load_experiment_config`.

---
date: 2026-06-26 20:47:33 +07
researcher: Codex
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Whether evaluation_records can really contain multiple entity_id values"
tags: [research, evaluation, smd, anomaly-archive, configs]
status: complete
last_updated: 2026-06-26
last_updated_by: Codex
---

# Research: Whether `evaluation_records` Can Really Contain Multiple `entity_id` Values

**Date**: 2026-06-26 20:47:33 +07  
**Researcher**: Codex  
**Git Commit**: `89a598f643cf0c20b0ab540b926e6b71f27e975f`  
**Branch**: `dev`

## Research Question

Can `evaluation_records` in the current codebase actually contain multiple distinct `entity_id` values at runtime, when traced from experiment and data configs into the active loader and evaluator path?

## Summary

Yes, the current runtime can produce `evaluation_records` with multiple distinct `entity_id` values, but this depends on the resolved data config.

- For `smd`, if `data.entity_ids` is omitted, the parser selects every SMD machine file in the split, so `evaluation_records` can contain many entity ids.
- For `smd`, if `data.entity_ids` is present and contains one machine id, the resolved runtime path is single-entity, so `evaluation_records` will contain only one `entity_id`.
- For `anomaly_archive`, the active parser loads exactly one file path into one series, so the runtime path is single-entity.

Therefore, “can it happen?” is a code-level yes. “does it happen for a specific run?” depends entirely on the resolved config.

## Detailed Findings

### Config Resolution

Experiment configs load a referenced data config and then apply optional `data_overrides`, after which the resolved config is validated ([src/core/config.py:1405](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:1405), [src/core/config.py:1407](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:1407)).

The validator explicitly allows `data.entity_ids` to be absent, and only checks that it is a non-empty list of strings when provided ([src/core/config.py:945](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:945)).

### SMD: Multi-Entity Is Possible

The general SMD data config `configs/data/smd.yaml` does not define `entity_ids` at all, so it leaves the loader unfiltered ([configs/data/smd.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/data/smd.yaml:1)).

When the SMD parser receives `entity_ids=None`, it selects every available SMD entity file:

- `selected_entity_ids = sorted(train_files_by_entity.keys())`
- then loops over every selected id and appends one train, one val, and one test sequence per entity

([src/data/datasets/smd.py:87](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/smd.py:87), [src/data/datasets/smd.py:113](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/smd.py:113)).

Since the evaluator groups by `entity_id` and emits one evaluation record per entity, this path can produce many `evaluation_records` with distinct ids ([src/engine/evaluator.py:68](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:68), [src/engine/evaluator.py:115](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:115)).

### SMD: Many Active Experiment Configs Are Single-Entity

Several active SMD experiment configs point to data configs that explicitly lock `entity_ids` to one machine. For example:

- `configs/data/smd_rtx3090_machine_2_1_20.yaml` contains only `machine-2-1` ([configs/data/smd_rtx3090_machine_2_1_20.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/data/smd_rtx3090_machine_2_1_20.yaml:1)).
- The RedLamp experiment config `smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr__w20__seed68__default.yaml` resolves through that single-entity data config ([configs/experiment/scale/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr__w20__seed68__default.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/scale/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr__w20__seed68__default.yaml:1)).

For those runs, `selected_entity_ids` has length one, so `evaluation_records` will also have length one.

### AnomalyArchive: Active Runtime Is Single-Entity

The active AnomalyArchive config points to exactly one `file_path`, not a list of entities ([configs/data/anomaly_archive_staffiii.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/data/anomaly_archive_staffiii.yaml:1)).

The parser builds one train sequence, one val sequence, and one test sequence for that one series only, all sharing the same `entity_id` derived from the file name metadata ([src/data/datasets/anomaly_archive.py:141](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:141), [src/data/datasets/anomaly_archive.py:162](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:162)).

So the active AnomalyArchive runtime path is single-entity.

## Code References

- `src/core/config.py:1405` - experiment config resolves referenced data config
- `src/core/config.py:945` - `entity_ids` is optional unless provided
- `src/data/datasets/smd.py:87` - SMD selects all entities when `entity_ids is None`
- `src/data/datasets/smd.py:113` - SMD appends one sequence per selected entity
- `src/data/datasets/anomaly_archive.py:141` - AnomalyArchive builds a single train sequence
- `src/data/datasets/anomaly_archive.py:162` - AnomalyArchive builds a single test sequence
- `src/engine/evaluator.py:115` - evaluator emits one record per grouped entity id

## Open Questions

- The code clearly allows multi-entity `evaluation_records` for SMD. The remaining practical question for any specific claim is therefore: which exact experiment config was resolved for that run?

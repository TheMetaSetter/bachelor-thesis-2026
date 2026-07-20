# Evaluation rerun provenance status

Date: 2026-07-20

This note summarizes the remote evaluation reruns that were confirmed on the shared GPU server at `159.48.242.1:20718`.

## Confirmed completed reruns

The rerun log currently shows 8 completed evaluation-only jobs:

- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6`

One rerun from the same batch is still pending:

- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8`

## Can these runs support the two requested report tables?

### 1. Offline benchmark metrics table

Yes. The completed runs contain `two_stage/stage_b_fusion_finetuning/evaluation_metrics.json`, and that file includes the required fields:

- `vus_pr`
- `affiliation_f1`
- `vus_roc`

That is enough to aggregate per combination and then average across the available seeds.

### 2. UQ summary table

No, not yet.

The completed runs also contain `metrics/uq_summary.json`, but the uncertainty summary fields are still null for the test split in the sampled completed runs. The trace audit also shows no retained MC sample history in the completed artifacts.

In practice, this means:

- the file exists,
- the schema is there,
- but the requested uncertainty statistics are not yet populated with meaningful values.

So the runs are sufficient for the benchmark metrics table, but not sufficient for the uncertainty report table.

## Provenance note

The completed reruns were evaluation-only runs. They did not rerun Stage A or Stage B training.

For the sampled completed run, the report still shows checkpoint provenance fields as present, but the uncertainty summary remains empty:

- `benchmark_status = evaluation_only`
- `checkpoint_path` points to the Stage B checkpoint
- `uq_summary.json` exists
- `uncertainty_summary.* = null`

## Practical conclusion

- Use these 8 completed runs for the metric table.
- Do not use them yet for the uncertainty table.
- Finish the remaining `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8` rerun before deciding whether to backfill or rerun anything else.

## Updated operational note

Before rerunning the remaining evaluation jobs, it is acceptable to prune only the heavy evaluation outputs from earlier runs if and only if the summary artifacts are preserved.

The intended sequence is:

1. Prune heavy files from completed evaluation runs, but keep summary artifacts intact.
2. Re-run evaluation so that all 17 target runs regenerate `uq_summary.json`.
3. Recount the completed runs after the rerun finishes.
4. Re-check whether the resulting `uq_summary.json` files and the remaining `.json` artifacts are sufficient for both tables:
   - the metric table with `VUS-PR`, `affiliation F1`, and `VUS-ROC`
   - the uncertainty table

This note is about the next operational step only. It does not claim the remaining runs have already been fixed.

## Remote check on 2026-07-20

I re-checked the current remote artifacts for the 9 rerun targets:

- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8`

Result:

- `evaluation_metrics.json` is present and complete for the metric table in all 9 runs.
- `uq_summary.json` is present in all 9 runs.
- For the checked test split, the uncertainty summary fields are still all `null` in all 9 runs.

So at the moment:

- **none** of the 9 rerun targets are sufficient for **both** report tables.
- the metric table is ready from the completed reruns,
- the uncertainty table still needs a successful re-evaluation path that yields non-null UQ summary values.

## Safe prune candidates

For the completed evaluation reruns, the following files are the main heavy artifacts that can be pruned safely **if the goal is only to keep the two summary tables**:

- `two_stage/stage_b_fusion_finetuning/traces/*.json`
- `two_stage/stage_b_fusion_finetuning/retention/**/traces.json`
- `traces/*.json`
- `retention/**/traces.json`
- `scores/*.npz`
- `retention/**/scores/*.npz`
- `two_stage/stage_b_fusion_finetuning/evaluation_curves.json`
- `two_stage/stage_b_fusion_finetuning/evaluation_records.json`

Do **not** prune these if you still want the report tables:

- `two_stage/stage_b_fusion_finetuning/evaluation_metrics.json`
- `metrics/uq_summary.json`
- `benchmark/thesis_offline_benchmark_report.json`

The largest files observed in a completed run were the test traces and their retention mirrors, followed by validation traces and W&B logs.

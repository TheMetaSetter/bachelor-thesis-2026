# Benchmark metrics and data directories

Date: 2026-08-07

## Scope

This note lists the directories that contain metrics, reports, checkpoints,
retention data, and input data for the offline and online benchmark.

The remote inventory was checked at revision `820c8886` on
`root@159.48.242.1:20720`.

The paths below use the remote repository root:

```text
/root/bachelor-thesis-2026
```

## Reporting staging directory

The complete reporting bundle is stored under:

```text
/root/bachelor-thesis-2026/outputs/reporting/
```

The offline metric files are staged at:

```text
/root/bachelor-thesis-2026/outputs/reporting/offline_phase_tables/offline_metrics_54/
```

This directory contains 54 metric files:

- 18 THESIS Stage B `evaluation_metrics.json` files.
- 27 traditional ML `offline_metrics.json` files.
- 9 RedLamp encoder `evaluation_metrics.json` files.

The compact offline report data is stored at:

```text
/root/bachelor-thesis-2026/outputs/reporting/offline_phase_tables/offline_report_data.json
/root/bachelor-thesis-2026/outputs/reporting/offline_phase_tables/offline_report_data.md
```

The online metric files are staged at:

```text
/root/bachelor-thesis-2026/outputs/reporting/online_phase_tables/online_metrics_99/
```

This directory contains 99 metric files:

- 54 THESIS `online_metrics.json` files.
- 9 M2N2 `main` `online_metrics.json` files.
- 9 CANDI `main` `online_metrics.json` files.
- 27 traditional ML `main` `online_metrics.json` files.
- `online_metrics_99_manifest.csv`.
- `online_metrics_99_manifest.json`.

The manifest records the method, variant, entity, seed, source path, staged
path, file size, and SHA-256 hash.

The complete file inventory is stored at:

```text
/root/bachelor-thesis-2026/outputs/reporting/benchmark_metrics_manifest.csv
/root/bachelor-thesis-2026/outputs/reporting/benchmark_metrics_manifest.json
```

The manifest covers 153 staged files: 54 offline and 99 online. M2N2 and
CANDI do not have separate offline detection metrics in the remote output
tree. They use the RedLamp encoder checkpoint for their online runs, so the
manifest records their offline metric status as `not_available`.

## Offline phase directories

### THESIS offline benchmark

Run-level directories:

```text
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/
└── {O0,O1}/
    └── {machine_1_6,machine_3_4,machine_3_9}/
        └── seed{36,6,8}/
```

Important subdirectories are:

```text
metrics/
benchmark/
retention/
two_stage/stage_a_multitask_pretraining/
two_stage/stage_b_fusion_finetuning/
```

The main offline metric files are stored in:

```text
metrics/offline_metrics.json
metrics/uq_summary.json
benchmark/thesis_offline_benchmark_report.json
two_stage/stage_b_fusion_finetuning/evaluation_metrics.json
two_stage/stage_b_fusion_finetuning/metrics/offline_metrics.json
two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json
```

### Traditional ML offline benchmark

```text
/root/bachelor-thesis-2026/outputs/benchmark/smd/offline_benchmark/
└── {iforest,kmeans_ad,stumpy_channel_ab}/
    └── {machine_1_6,machine_3_4,machine_3_9}/
        └── seed{36,6,8}/
            ├── metrics/
            └── benchmark/
```

The main files are:

```text
metrics/offline_metrics.json
benchmark/offline_benchmark_report.json
```

### RedLamp encoder evaluation

```text
/root/bachelor-thesis-2026/outputs/benchmark/smd/redlamp_baseline/
└── {machine_1_6,machine_3_4,machine_3_9}/seed{36,6,8}/
```

The evaluation metric file is:

```text
evaluation_metrics.json
```

### Evaluation-only outputs

```text
/root/bachelor-thesis-2026/outputs/eval18/
└── {o0,o1}_m{1_6,3_4,3_9}_s{36,6,8}/metrics/
```

These outputs are separate from the main benchmark tree.

## Online phase directories

### THESIS online benchmark

```text
/root/bachelor-thesis-2026/outputs/benchmark/online/smd/thesis/
└── {O0_A0,O0_A1,O0_A2,O1_A0,O1_A1,O1_A2}/
    └── {machine_1_6,machine_3_4,machine_3_9}/
        └── seed{36,6,8}/
```

Important subdirectories and files are:

```text
online_metrics.json
online_records.json
metrics.jsonl
resolved_experiment_config.json
online_artifact_manifest.json
thresholds/
checkpoints/
benchmark/
retention/
wandb/
```

The benchmark report is stored in:

```text
benchmark/thesis_online_A{0,1,2}_benchmark_report.json
```

The retention data is stored in:

```text
retention/{machine-1-6,machine-3-4,machine-3-9}/A{0,1,2}/
```

### Online baselines

```text
/root/bachelor-thesis-2026/outputs/benchmark/online_streaming/smd/
└── {candi,m2n2,iforest,kmeans_ad,stumpy}/
    └── {main,A0,A1,A2}/
        └── {machine_1_6,machine_3_4,machine_3_9}/
            └── seed{36,6,8}/
```

For the current 99-run matrix, use only the `main` directories. The `A0`,
`A1`, and `A2` directories of the traditional baselines are older outputs
outside the current matrix.

Important files are:

```text
online_metrics.json
online_records.json
metrics.jsonl
resolved_experiment_config.json
thresholds/online_thresholds.json
benchmark/online_streaming_benchmark_report.json
wandb/
```

## Input data directory

The SMD input data is stored at:

```text
/root/bachelor-thesis-2026/data/SMD/
```

The data configuration files are stored at:

```text
/root/bachelor-thesis-2026/configs/data/
```

## Reporting rule

Use the following files as the first source for the comparison table:

1. Offline phase: `offline_metrics.json` or `evaluation_metrics.json`,
   depending on the method.
2. Online phase: `online_metrics.json`.
3. Use the benchmark report and manifest files to verify run identity,
   provenance, and completion status.
4. Use `metrics.jsonl` and `wandb/` for step-level diagnostics, not as the
   primary table source.

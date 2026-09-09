# CLI: remaining-SMD benchmark

The cloud wrapper is the command for the rented server with four V100 GPUs and 44 CPU cores.

The eight selected machines are `machine-3-1`, `machine-3-5`, `machine-3-2`, `machine-3-11`, `machine-3-10`, `machine-1-3`, `machine-1-1`, and `machine-2-8`.

The three completed machines `machine-1-6`, `machine-3-4`, and `machine-3-9` remain excluded by the generator.

The default dataset root is `data/ServerMachineDataset`.

The default output roots are `outputs/benchmark_smoke/smd_remaining` for smoke mode and `outputs/benchmark/smd_remaining` for wet mode.

## Cloud preflight

Run this read-only check before creating configs or tmux sessions.

```bash
bash scripts/benchmarks/run_remaining_smd_cloud_tmux.sh \
  --mode smoke --gpu-count 4 --preflight \
  --entity-id machine-3-1 --entity-id machine-3-5 \
  --entity-id machine-3-2 --entity-id machine-3-11 \
  --entity-id machine-3-10 --entity-id machine-1-3 \
  --entity-id machine-1-1 --entity-id machine-2-8
```

The preflight checks four visible CUDA devices, at least 44 CPU cores, `tmux`, `taskset`, dataset files, and target GPU occupancy.

## Cloud smoke dry-run

```bash
bash scripts/benchmarks/run_remaining_smd_cloud_tmux.sh \
  --mode smoke --gpu-count 4 --dry-run \
  --entity-id machine-3-1 --entity-id machine-3-5 \
  --entity-id machine-3-2 --entity-id machine-3-11 \
  --entity-id machine-3-10 --entity-id machine-1-3 \
  --entity-id machine-1-1 --entity-id machine-2-8
```

This command writes no configs and prints the eight GPU queues, four CPU queues, CPU masks, and metric contract.

## Cloud smoke execution

```bash
bash scripts/benchmarks/run_remaining_smd_cloud_tmux.sh \
  --mode smoke --gpu-count 4 --skip-completed \
  --entity-id machine-3-1 --entity-id machine-3-5 \
  --entity-id machine-3-2 --entity-id machine-3-11 \
  --entity-id machine-3-10 --entity-id machine-1-3 \
  --entity-id machine-1-1 --entity-id machine-2-8
```

The controller creates offline queues first, validates Stage B and RedLamp dependencies, then creates online queues.

The controller session is `smd-smoke`.

Attach with `tmux attach -t smd-smoke`.

GPU sessions use devices `0` through `3` and CPU masks `0-7`, `8-15`, `16-23`, and `24-31`.

CPU baseline sessions use masks `32-37` and `38-43`.

## Cloud wet execution

Run the same preflight and smoke flow with `--mode wet` before this command.

```bash
bash scripts/benchmarks/run_remaining_smd_cloud_tmux.sh \
  --mode wet --gpu-count 4 --skip-completed \
  --entity-id machine-3-1 --entity-id machine-3-5 \
  --entity-id machine-3-2 --entity-id machine-3-11 \
  --entity-id machine-3-10 --entity-id machine-1-3 \
  --entity-id machine-1-1 --entity-id machine-2-8
```

The wet controller session is `smd-wet`.

## Local two-GPU compatibility launcher

The legacy launcher keeps its default of two GPUs for historical workflows.

```bash
bash scripts/benchmarks/run_remaining_smd_smoke.sh --dry-run --no-tmux
bash scripts/benchmarks/run_remaining_smd_wet.sh --dry-run --no-tmux
```

Use repeated `--entity-id ENTITY_ID` arguments to select a subset with either launcher.

Use `--skip-completed` to skip only a run whose expected final report already exists.

Use `--no-tmux` to keep the legacy coordinator in the current shell.

## Metric collection

The cloud controller collects metrics automatically after all online workers finish.

To collect again, run:

```bash
./.venv/bin/python -m scripts.benchmarks.collect_remaining_smd_metrics \
  --manifest outputs/benchmark_smoke/smd_remaining/remaining_smd_manifest.json
```

Replace `benchmark_smoke` with `benchmark` for the wet report.

The compact reports are `remaining_smd_metrics.json` and `remaining_smd_metrics.md`.

The reports retain only `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR`.

The three `VUS-PR@FPR-budget` budgets are `0.1%`, `0.5%`, and `1%`.

Incomplete online metric payloads are reported as `incomplete`, not as valid numeric rows.

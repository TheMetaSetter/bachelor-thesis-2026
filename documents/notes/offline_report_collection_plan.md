# Offline report collection plan

Date: 2026-07-20

## Goal

Collect all completed run results into one compact data bundle so the paper can
be rebuilt into two tables without re-running Stage A or Stage B:

1. Offline phase comparison against baselines.
2. Offline variance report comparing validation and evaluation splits.

## Source roots

Use these trees as raw inputs:

- `outputs/eval18`
- `outputs/benchmark/smd/offline_benchmark`
- `outputs/benchmark/smd/redlamp_baseline`

Important split:
- RedLamp is collected on the local machine from `outputs/benchmark/smd/redlamp_baseline`.
- Thesis and the other methods are collected on the remote machine from the matching `outputs/...` trees there.

## Collector script

Use:

`scripts/ops/collect_offline_report_data.py`

It scans every `evaluation_metrics.json` under the source roots, then pairs each
run with the closest `metrics/uq_summary.json` if it exists.

## What gets preserved

The collector keeps row-level data for every run:

- run identity
- metric values
- UQ summary per split
- trace-audit flags

This means the same raw bundle can later be reshaped into:

- one wide table with methods as rows and entities as columns
- three separate entity tables
- a validation-vs-evaluation variance table

## Output

By default the collector writes:

- JSON bundle: `outputs/reporting/offline_phase_tables/offline_report_data.json`
- small index markdown: `outputs/reporting/offline_phase_tables/offline_report_data.md`

## Practical order

1. Collect RedLamp raw data locally.
2. Collect thesis and other baseline raw data on the remote machine.
2. Verify the metric table group means.
3. Verify the validation/evaluation variance rows.
4. Only then decide whether any artifact pruning is safe.

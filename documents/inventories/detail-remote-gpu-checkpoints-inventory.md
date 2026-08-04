# Remote GPU Checkpoints Inventory

Source: read-only SSH scan on `2026-07-14` from `/root/bachelor-thesis-2026` on the remote GPU server.

## Revalidation on 2026-08-04

A second read-only SSH scan found exactly 18 files matching
`outputs/benchmark/smd/thesis/*/machine_*/seed*/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`.
They match every O0/O1, entity, and seed pair listed below. The remote checkout
was at revision `2200db1ed49024e06b5a85d09634f93e19febc71`, which matched the
local checkout at the time of the scan. No source, config, or output was
written during this check.

The scan only proves that the Stage-B checkpoint files exist. It does not prove
that each corresponding threshold artifact satisfies the new THESIS V4
contract. For example, the checked O1 / `machine_1_6` / seed6 artifact had
schema version 3, no checkpoint hash, and only `offline_point` plus
`online_ewma_point`. It cannot start the new V4 runtime. Re-check the exact
artifact for any checkpoint before an online run.

The remote tree currently contains the same 5 checkpoint files for each THESIS two-stage run:
- `initializations/stage_b_init.pt`
- `stage_a_multitask_pretraining/checkpoints/best.pt`
- `stage_a_multitask_pretraining/checkpoints/final.pt`
- `stage_b_fusion_finetuning/checkpoints/best.pt`
- `stage_b_fusion_finetuning/checkpoints/final.pt`

## O0

### machine_1_6 / seed6
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_1_6 / seed8
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_1_6 / seed36
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_4 / seed6
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_4 / seed8
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_4 / seed36
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_4/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_9 / seed6
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_9 / seed8
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_9 / seed36
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O0/machine_3_9/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

## O1

### machine_1_6 / seed6
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_1_6 / seed8
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_1_6 / seed36
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_1_6/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_4 / seed6
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_4 / seed8
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_4 / seed36
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_4/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_9 / seed6
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_9 / seed8
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

### machine_3_9 / seed36
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/initializations/stage_b_init.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/stage_a_multitask_pretraining/checkpoints/final.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36/two_stage/stage_b_fusion_finetuning/checkpoints/final.pt`

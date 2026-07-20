# Evaluation UQ Backfill Commands

Date: 2026-07-20

Mục tiêu:
- Chạy lại **evaluation-only** cho 1 run mẫu sau bản vá.
- Xác nhận `evaluation_traces.json` còn `uncertainty_history`, nhưng không còn raw MC payload nặng.
- Backfill lại `uq_summary.json` cho toàn bộ run cũ.
- Prune trace nặng sau khi summary đã có số thật.

## 1. Tạo một config evaluation-only tạm

Chọn 1 run mẫu đã có Stage B checkpoint.
Ví dụ dùng run `O0 / machine_3_9 / seed8`.

```bash
cd /root/bachelor-thesis-2026
mkdir -p /tmp/thesis_evalonly_o0_m3_9_seed8
cp \
  outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/generated_configs/02_stage_b_fusion_finetuning.yaml \
  /tmp/thesis_evalonly_o0_m3_9_seed8/eval_only.yaml
```

Sau đó sửa 2 dòng trong `/tmp/thesis_evalonly_o0_m3_9_seed8/eval_only.yaml`:
- `output_dir`
- `checkpoint_dir`

Đặt chúng sang thư mục mới, ví dụ:
- `output_dir: /tmp/thesis_evalonly_o0_m3_9_seed8/output`
- `checkpoint_dir: /tmp/thesis_evalonly_o0_m3_9_seed8/output/checkpoints`

## 2. Chạy evaluation-only cho run mẫu

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python -m scripts.run_thesis_offline_benchmark \
  --experiment-config /tmp/thesis_evalonly_o0_m3_9_seed8/eval_only.yaml \
  --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
  --evaluation-only \
  --checkpoint-path /root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O0/machine_3_9/seed8/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt \
  2>&1 | tee outputs/tmux_logs/evalonly_o0_m3_9_seed8.txt
```

Kiểm tra sau khi chạy:
- `.../output/evaluation_traces.json`
- `.../output/metrics/uq_summary.json`
- `.../output/evaluation_metrics.json`

Mục tiêu là:
- `uncertainty_history` còn tồn tại trong trace
- các field variance trong `uq_summary.json` không còn `null`
- raw sample payload đã được compact

## 3. Backfill lại tất cả `uq_summary.json`

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python scripts/ops/backfill_all_uq_summaries.py \
  --root-dir outputs/benchmark \
  2>&1 | tee outputs/tmux_logs/backfill_all_uq_summaries.txt
```

Nếu muốn backfill cả smoke outputs:

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python scripts/ops/backfill_all_uq_summaries.py \
  --root-dir outputs/benchmark_smoke \
  2>&1 | tee outputs/tmux_logs/backfill_all_uq_summaries_smoke.txt
```

## 4. Prune trace nặng sau khi summary đã ổn

Dry-run trước:

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python scripts/ops/prune_raw_trace_artifacts.py \
  --root-dir outputs/benchmark \
  2>&1 | tee outputs/tmux_logs/prune_raw_trace_artifacts_dry_run.txt
```

Áp dụng thật:

```bash
cd /root/bachelor-thesis-2026
THESIS_CONSOLE_QUIET=1 \
.venv/bin/python scripts/ops/prune_raw_trace_artifacts.py \
  --root-dir outputs/benchmark \
  --apply \
  2>&1 | tee outputs/tmux_logs/prune_raw_trace_artifacts_apply.txt
```

## 5. Thứ tự an toàn

1. Chạy evaluation-only cho 1 run mẫu.
2. Kiểm tra trace compact và `uq_summary.json`.
3. Backfill toàn bộ run cũ.
4. Dry-run prune.
5. Apply prune nếu dry-run đúng.


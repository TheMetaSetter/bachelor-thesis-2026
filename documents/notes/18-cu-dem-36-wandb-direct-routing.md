# 18 cú đêm, 36 W&B, direct routing

## Mục tiêu hiện tại

Chạy đủ 18 offline combinations:

```text
O0/O1 × machine-1-6/machine-3-4/machine-3-9 × seed 6/8/36
```

Mỗi combination chạy Stage A, tạo Stage-B initialization, Stage B và
evaluation.

## Contract của matrix mới

- Reconstruction loss dùng MSE trên normalized input-output.
- `lambda_recon: 0.75`.
- `lambda_cls: 0.25`.
- O0 tắt point-score loss.
- O1 bật Balanced Point-Score Loss ở Stage A.
- Cả Stage A và Stage B dùng `fusion_mode: direct_branch_routing`.
- Mọi data loader dùng `num_workers: 12`.
- Protocol dùng clean validation, q99, window 20, raw-input identity score.

## W&B count

Main configs bật W&B. Mỗi combination tạo hai W&B runs:

```text
18 combinations × 2 training stages = 36 W&B runs
```

Evaluation không tạo W&B run riêng. Dry-run không tạo W&B run.

## Files

- Generator: `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py`
- Runner: `scripts/benchmarks/run_full_direct_recon075_cls025_offline_matrix.sh`
- Generated configs: `outputs/benchmark_full_direct_recon075_cls025/generated_configs/`
- Results: `outputs/benchmark_full_direct_recon075_cls025/smd/thesis/`

## tmux command

Chạy sau khi đăng nhập vào cloud GPU:

```bash
tmux new -s offline18 'bash scripts/benchmarks/run_full_direct_recon075_cls025_offline_matrix.sh'
```

Script chạy preflight cho cả 18 config trước, sau đó chạy tuần tự 18 full
benchmarks. Trước khi mở toàn bộ matrix, vẫn cần một full GPU run đầu tiên
thành công theo quy định của repository.

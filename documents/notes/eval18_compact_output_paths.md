# Compact output paths for thesis evaluation reruns

Date: 2026-07-20

Mục tiêu:
- Giữ path ngắn, dễ kiểm tra.
- Không đụng cây benchmark cũ.
- Chạy evaluation-only vào một root compact riêng.

## Quy ước path

Mỗi combination được map sang alias ngắn:

- `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6`
  -> `outputs/eval18/o0_m1_6_s6/`
- `outputs/benchmark/smd/thesis/O1/machine_3_9/seed36`
  -> `outputs/eval18/o1_m3_9_s36/`

## Script dùng

- `scripts/ops/run_thesis_eval_only_compact.py`

Script này:
- copy Stage B config sang config compact
- đổi `output_dir` và `checkpoint_dir` sang root ngắn
- chạy `scripts.run_thesis_offline_benchmark --evaluation-only`
- ghi manifest tổng tại `outputs/eval18/manifest.json`

## Chỗ kiểm tra nhanh

- config compact:
  `outputs/eval18/<alias>/generated_configs/stage_b_eval_only.yaml`
- artifact compact:
  `outputs/eval18/<alias>/`
- log:
  `outputs/tmux_logs/eval18_<alias>.txt`
- manifest:
  `outputs/eval18/manifest.json`

## Ghi nhớ

- Source checkpoint vẫn đọc từ tree benchmark cũ.
- Tree compact chỉ dùng cho rerun evaluation và audit.
- Nếu muốn pilot, chỉ truyền 1 `--run-root` trước.

# TMux Benchmark Runbook

Ngày dùng:
- 2026-07-06

Mục tiêu:
- Chạy 3 nhánh benchmark riêng trên 3 GPU khác nhau.
- Mỗi nhánh là một `tmux` session độc lập để không gián đoạn khi SSH rớt.
- Không ép GPU phải là RTX 3090.

## 1. Ba nhánh benchmark

### 1.1 Baseline

- Session: `smd-benchmark-baseline`
- GPU: `0`
- Report dir: `outputs/comparative_smd_reports/benchmark-baseline`

Config files:
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed8__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed36__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed6__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed8__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed36__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed8__main.yaml`
- `configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed36__main.yaml`

### 1.2 Thesis base two-stage

- Session: `smd-benchmark-thesis-base`
- GPU: `1`
- Report dir: `outputs/comparative_smd_reports/benchmark-thesis-base`

Config files:
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed8__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed36__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed6__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed8__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed36__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed6__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed8__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed36__main.yaml`

### 1.3 Thesis point-score two-stage

- Session: `smd-benchmark-thesis-point-score`
- GPU: `2`
- Report dir: `outputs/comparative_smd_reports/benchmark-thesis-point-score`

Config files:
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed6__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed8__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed36__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed6__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed8__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed36__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed6__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed8__main.yaml`
- `configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed36__main.yaml`

## 2. Lệnh tmux

### 2.1 Baseline

```bash
tmux new-session -d -s smd-benchmark-baseline bash -lc 'cd "$(git rev-parse --show-toplevel)" && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/preflight_comparative_smd_server.py \
  --report-dir outputs/comparative_smd_reports/benchmark-baseline \
  --gpu-index 0 \
  --required-gpu-name-substring "" \
  --config-paths \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed8__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed36__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed6__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed8__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed36__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed8__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed36__main.yaml \
  --print-json \
  --require-launch-ready && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_comparative_smd_experiments.py \
  --report-dir outputs/comparative_smd_reports/benchmark-baseline \
  --skip-completed \
  --config-paths \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed8__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_1_6__w20__seed36__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed6__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed8__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_4__w20__seed36__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed6__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed8__main.yaml \
  configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-machine_3_9__w20__seed36__main.yaml'
```

### 2.2 Thesis base

```bash
tmux new-session -d -s smd-benchmark-thesis-base bash -lc 'cd "$(git rev-parse --show-toplevel)" && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/preflight_comparative_smd_server.py \
  --report-dir outputs/comparative_smd_reports/benchmark-thesis-base \
  --gpu-index 0 \
  --required-gpu-name-substring "" \
  --config-paths \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed36__main.yaml \
  --print-json \
  --require-launch-ready && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_comparative_smd_experiments.py \
  --report-dir outputs/comparative_smd_reports/benchmark-thesis-base \
  --skip-completed \
  --config-paths \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_4__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-machine_3_9__w20__seed36__main.yaml'
```

### 2.3 Thesis point-score

```bash
tmux new-session -d -s smd-benchmark-thesis-point-score bash -lc 'cd "$(git rev-parse --show-toplevel)" && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/preflight_comparative_smd_server.py \
  --report-dir outputs/comparative_smd_reports/benchmark-thesis-point-score \
  --gpu-index 0 \
  --required-gpu-name-substring "" \
  --config-paths \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed36__main.yaml \
  --print-json \
  --require-launch-ready && \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_comparative_smd_experiments.py \
  --report-dir outputs/comparative_smd_reports/benchmark-thesis-point-score \
  --skip-completed \
  --config-paths \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_1_6__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_4__w20__seed36__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed6__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed8__main.yaml \
  configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-machine_3_9__w20__seed36__main.yaml'
```

## 3. Thao tác trong màn hình `tmux`

1. Sau khi bắn lệnh, kiểm tra session:

```bash
tmux ls
```

2. Vào session muốn xem:

```bash
tmux attach -t smd-benchmark-baseline
tmux attach -t smd-benchmark-thesis-base
tmux attach -t smd-benchmark-thesis-point-score
```

3. Khi đang ở trong `tmux`, thoát ra mà không dừng job:

```text
Ctrl-b rồi nhấn d
```

4. Xem log nếu cần:

```bash
tail -f outputs/tmux_logs/smd-benchmark-baseline.log
tail -f outputs/tmux_logs/smd-benchmark-thesis-base.log
tail -f outputs/tmux_logs/smd-benchmark-thesis-point-score.log
```

5. Nếu muốn dừng một session:

```bash
tmux kill-session -t smd-benchmark-baseline
tmux kill-session -t smd-benchmark-thesis-base
tmux kill-session -t smd-benchmark-thesis-point-score
```

## 4. Ghi chú an toàn

- Trước khi chạy thật, nên kiểm tra `tmux ls` để chắc không đè session cũ.
- Nếu server đã có session trùng tên, thêm `--replace-session` vào launcher hoặc kill session cũ trước.
- Nếu bạn đổi mapping GPU, nhớ giữ `--gpu-index` khớp với `CUDA_VISIBLE_DEVICES`.
- Nếu `CUDA_VISIBLE_DEVICES` đã khóa còn đúng 1 GPU trong tmux session, thì `--gpu-index 0` là đúng vì tiến trình chỉ thấy một CUDA device cục bộ.
- Mỗi nhánh đã tách report dir riêng nên artifacts sẽ không đè nhau.

## 5. Artifact sẽ sinh ra

- `outputs/comparative_smd_reports/benchmark-baseline/`
- `outputs/comparative_smd_reports/benchmark-thesis-base/`
- `outputs/comparative_smd_reports/benchmark-thesis-point-score/`
- `outputs/tmux_logs/smd-benchmark-baseline.log`
- `outputs/tmux_logs/smd-benchmark-thesis-base.log`
- `outputs/tmux_logs/smd-benchmark-thesis-point-score.log`

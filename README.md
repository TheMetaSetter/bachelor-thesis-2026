# Phương pháp THESIS: Các toán tử truy vấn ngẫu nhiên theo kiến trúc hai nhánh cho bài toán phát hiện bất thường chuỗi thời gian

## Giới thiệu dự án nghiên cứu


### CLI Run Convention

Luôn chạy các entry point của dự án dưới dạng **Python module** từ thư mục gốc (repository root). Cách này giúp Python thiết lập đúng module search path và tránh các lỗi import như:

```

ModuleNotFoundError: No module named 'scripts'

````

Ví dụ:

```bash
cd /root/bachelor-thesis-2026
.venv/bin/python -m scripts.run_thesis_offline_benchmark \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__main.yaml \
  --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
  --skip-completed
````

Không nên chạy trực tiếp file:

```bash
python scripts/run_thesis_offline_benchmark.py
```

vì khi đó Python có thể không nhận diện đúng package `scripts`, dẫn đến lỗi import.

Khi muốn **chỉ evaluation trên checkpoint đã có**, dùng:

```bash
cd /root/bachelor-thesis-2026
.venv/bin/python -m scripts.run_thesis_offline_benchmark \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__main.yaml \
  --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
  --evaluation-only \
  --checkpoint-path outputs/benchmark/smd/thesis/O1/machine_1_6/seed6/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt
```

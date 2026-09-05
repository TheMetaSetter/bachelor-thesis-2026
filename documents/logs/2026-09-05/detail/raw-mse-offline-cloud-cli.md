# Chạy Stage A và Stage B với raw-input MSE trên cloud GPU

Anh chạy các lệnh trong Bash, từ thư mục gốc repo trên cloud. Em chỉ chuẩn bị
code và kiểm tra local; em chưa đăng nhập hay chạy training trên cloud.

Reconstruction loss dùng MSE trong đơn vị sensor gốc. O1 giữ balanced BCE,
với reconstruction error đầu vào là raw MSE. Anomaly score dùng identity.
Stage A chạy 25 epoch; Stage B chạy 5 epoch; mỗi stage chọn `best.pt` theo
`val_synth_vus_pr` lớn nhất. Lệnh hai stage tự khởi tạo memory và đánh giá
Stage B `best.pt` bằng raw MSE, với threshold q99 từ clean validation.

## Phase 1 — Chuẩn bị repo và môi trường

Stage đồng bộ — công cụ: SSH, rsync. `cloud-gpu.txt` hiện ghi endpoint dưới
đây; kiểm tra lại file nếu chạy vào ngày khác. File không ghi đường dẫn repo
trên cloud: thay `/ABSOLUTE/REMOTE/REPO` bằng đường dẫn thật.

Trên máy local, tại repo hiện tại:

```bash
export RAW_MSE_REMOTE_REPO=/ABSOLUTE/REMOTE/REPO
rsync -avR --exclude='__pycache__/' --exclude='generated/' -e 'ssh -p 20007' src/ scripts/ configs/ tests/ root@159.48.242.13:"$RAW_MSE_REMOTE_REPO/"
ssh -p 20007 root@159.48.242.13
```

Lệnh rsync đồng bộ code/config/test vào repo đã có, không có `--delete`.
Không đồng bộ `.venv`, credential, dataset hoặc output từ máy local.

Stage kiểm tra — công cụ: Bash, Python/PyTorch, W&B. Các lệnh sau chạy trên cloud:

```bash
cd /ABSOLUTE/REMOTE/REPO
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
test -f data/ServerMachineDataset/train/machine-1-6.txt
test -f data/ServerMachineDataset/train/machine-3-4.txt
test -f data/ServerMachineDataset/train/machine-3-9.txt
.venv/bin/wandb login
.venv/bin/python -m pytest tests/models/test_raw_reconstruction_loss.py tests/runtime/test_raw_mse_training.py tests/evaluation/test_raw_input_mse_scores.py tests/benchmarks/test_raw_mse_rerun_configs.py -q
```

Điều kiện tiếp tục: Python trong `.venv` có CUDA và dependency của repo; dataset
có đủ train/test/test_label cho ba entity; test thành công; W&B đăng nhập được.
Môi trường cloud chưa được em xác minh trực tiếp.

## Phase 2 — Tạo config và chạy một tổ hợp đầy đủ

Stage tạo config — công cụ: Python, YAML. Chạy một lần:

```bash
.venv/bin/python -m scripts.ops.prepare_raw_mse_offline_rerun --run-id raw_mse_20260905
.venv/bin/python -m scripts.experiments.run_two_stage_offline_pretraining --experiment-config configs/generated/raw_mse_20260905/O0__machine_1_6__seed6.yaml --dry-run
```

Generator tạo 18 config O0/O1 × ba entity × seed 6/8/36. Nó không train và từ
chối ghi đè config đã tồn tại. Nếu cần một đợt chạy mới, đổi `--run-id` và dùng
tên mới ở các lệnh tiếp theo. Dry-run chỉ tạo manifest/config stage.

Stage preflight — công cụ: tmux, Python. Chạy O0, machine-1-6, seed 6 với đủ 25+5 epoch:

```bash
tmux new -s raw_mse_20260905
export RAW_MSE_CELL=outputs/benchmark/smd/machine_1_6/seed6/thesis_O0_raw_mse_20260905/offline
mkdir -p "$RAW_MSE_CELL"
.venv/bin/python -m scripts.experiments.run_two_stage_offline_pretraining --experiment-config configs/generated/raw_mse_20260905/O0__machine_1_6__seed6.yaml > "$RAW_MSE_CELL/runner.log" 2>&1
```

Nhấn `Ctrl-b`, rồi `d` để rời tmux. Mở terminal khác để đọc tiến độ:

```bash
tail -f outputs/benchmark/smd/machine_1_6/seed6/thesis_O0_raw_mse_20260905/offline/runner.log
```

Vào lại phiên:

```bash
tmux attach -t raw_mse_20260905
```

Stage kiểm tra artifact — công cụ: Python, SHA256. Sau khi lệnh train trả về thành công:

```bash
.venv/bin/python -m json.tool "$RAW_MSE_CELL/stage_b_fusion_finetuning/evaluation_metrics.json"
sha256sum "$RAW_MSE_CELL/stage_a_multitask_pretraining/initial.pt" "$RAW_MSE_CELL/stage_a_multitask_pretraining/best.pt"
sha256sum "$RAW_MSE_CELL/stage_b_fusion_finetuning/stage_b_init.pt" "$RAW_MSE_CELL/stage_b_fusion_finetuning/best.pt"
```

Kiểm tra tự động các điều kiện chính trước khi mở rộng matrix:

```bash
.venv/bin/python - <<'PY'
import json, math, os
from pathlib import Path
import torch

root = Path(os.environ['RAW_MSE_CELL'])
stage_b = root / 'stage_b_fusion_finetuning'
metrics = json.loads((stage_b / 'evaluation_metrics.json').read_text())
assert metrics['score_space'] == 'raw_input'
assert metrics['point_score_transform'] == 'identity'
assert metrics['threshold_source'] == 'clean_validation_quantile'
assert metrics['is_truncated_evaluation'] == 0
assert metrics['label_regime'] == 'mixed'
for name in ('vus_pr', 'affiliation_f1', 'vus_roc'):
    assert math.isfinite(metrics[name]), (name, metrics[name])
    print(name, metrics[name])
best = torch.load(stage_b / 'best.pt', map_location='cpu', weights_only=False)
initial = torch.load(stage_b / 'initial.pt', map_location='cpu', weights_only=False)
assert best['config']['reconstruction_loss_space'] == 'raw_input'
assert best['config']['model']['training_phase'] == 'stage_b_fusion_finetuning'
assert best['extra_state']['memory_initialized']
assert best['extra_state']['evaluation_threshold_source'] == 'clean_validation_quantile'
for name, value in initial['model_state_dict'].items():
    if name.startswith('encoder.') or name in ('continuous_prototype_bank', 'discrete_codebook'):
        assert torch.equal(value, best['model_state_dict'][name]), name
print('Preflight artifact checks passed; best epoch:', best['epoch'])
PY
```

Metric Stage A và Stage B theo epoch nằm trong `metrics.jsonl` của từng stage.
`val_synth_vus_pr` và `val_synth_affiliation_f1_pointwise` dùng synthetic
validation. `evaluation_metrics.json` ở Stage B chứa metric trên test thật.
Đây là hai tập đánh giá khác nhau; không gộp các số đó thành một cột.

## Phase 3 — Chạy 17 tổ hợp còn lại

Stage matrix — công cụ: Bash, Python, tmux. Chỉ chạy sau khi preflight ở trên qua.
Vòng lặp chạy tuần tự và dừng ngay nếu một tổ hợp lỗi:

```bash
for entity in machine_1_6 machine_3_4 machine_3_9; do
  for variant in O0 O1; do
    for seed in 6 8 36; do
      if [[ "$entity/$variant/$seed" == machine_1_6/O0/6 ]]; then
        continue
      fi
      cell="outputs/benchmark/smd/$entity/seed$seed/thesis_${variant}_raw_mse_20260905/offline"
      mkdir -p "$cell"
      .venv/bin/python -m scripts.experiments.run_two_stage_offline_pretraining \
        --experiment-config "configs/generated/raw_mse_20260905/${variant}__${entity}__seed${seed}.yaml" \
        > "$cell/runner.log" 2>&1 || break 3
    done
  done
done
```

Stage đọc kết quả — công cụ: Python/JSON. Chỉ đọc các artifact của đợt chạy này:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

print('entity,variant,seed,VUS-PR,Affiliation-F1,VUS-ROC')
for entity in ('machine_1_6', 'machine_3_4', 'machine_3_9'):
    for variant in ('O0', 'O1'):
        for seed in (6, 8, 36):
            path = Path('outputs/benchmark/smd') / entity / f'seed{seed}' / f'thesis_{variant}_raw_mse_20260905/offline/stage_b_fusion_finetuning/evaluation_metrics.json'
            if not path.exists():
                print(f'{entity},{variant},{seed},MISSING,MISSING,MISSING')
                continue
            metric = json.loads(path.read_text())
            print(entity, variant, seed, metric['vus_pr'], metric['affiliation_f1'], metric['vus_roc'], sep=',')
PY
```

Các override có chủ đích: `reconstruction_loss_space: raw_input`, raw/identity
evaluation, output riêng theo run ID và bật reconstruction diagnostics.
Giá trị kế thừa từ main config: CUDA, AdamW lr 0.001, gradient clip 0.5,
25+5 epoch, window 20, MC 10, VUS buffer 20, 200 thresholds, W&B online.
CLI không có `--loss-space`, `--epochs` hoặc `--device`; các giá trị nằm trong YAML.

Không dùng `--skip-completed` cho một đợt rerun mới: tùy chọn này chỉ bỏ qua
toàn bộ flow đã hoàn tất, không resume optimizer giữa chừng.

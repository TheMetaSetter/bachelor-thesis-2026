# Online absolute range và pretrained encoder contract

**Ngày chốt:** 2026-08-05  
**Phạm vi:** online test-time adaptation trên SMD cho THESIS, M2N2, CANDI và traditional ML baselines  
**Trạng thái:** quyết định chính thức cho benchmark hiện tại

## Kết luận chính

Benchmark online sẽ stream cùng một đoạn ngắn của từng test series cho tất cả
phương pháp. Đoạn này dùng chỉ số tuyệt đối của entity và có dạng nửa kín
`[absolute_start_index, absolute_end_index)`. Cách này giữ nguyên vị trí thật
của anomaly trong báo cáo.

THESIS dùng encoder trong Stage B `best.pt` của chính THESIS. M2N2 và CANDI
đều dùng encoder CNN đơn giản đã pretrain trong RedLamp. Latent dimension của
RedLamp là `128`, còn latent dimension của THESIS là `64`. Hai dimension này
không cần bằng nhau. Đây là quyết định có chủ ý để dùng ngay checkpoint đã có
và không train lại từng combination.

## 1. Absolute range contract

### 1.1 Dữ liệu cấu hình

Mỗi online config dùng đúng hai trường sau:

```yaml
task_overrides:
  absolute_start_index: 146
  absolute_end_index: 2200
```

Hai trường là tùy chọn ở cấp runtime:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `absolute_start_index` | integer hoặc `null` | Vị trí bắt đầu trong test series gốc của entity, có tính điểm bắt đầu |
| `absolute_end_index` | integer hoặc `null` | Vị trí kết thúc, không tính điểm kết thúc |

Nếu cả hai trường là `null`, runner dùng toàn bộ test series. Nếu một trường
là `null` và trường còn lại không phải `null`, config bị từ chối.

Range phải thỏa:

```text
0 <= absolute_start_index < absolute_end_index <= source_sequence_length
```

Runner cắt range trước khi tạo sliding windows. Với `window_size = 20` và
`stride = 1`, số causal windows của range có độ dài `R` là:

```text
max(0, R - 20 + 1)
```

`max_online_steps` chỉ là giới hạn cho smoke test. Nó được áp dụng sau khi
chọn absolute range và không thay thế hai trường absolute range. Main run phải
để `max_online_steps: null`.

### 1.2 Metadata phải được giữ lại

Sau khi cắt, sequence phải giữ các trường dữ liệu tương ứng:

```text
x
point_labels
mask
timestamps
```

Metadata của sequence phải ghi:

```text
source_sequence_length
sequence_length
absolute_start_index
absolute_end_index
```

Các record online phải dùng lại offset này. Vì vậy, `point_index`,
`window_start_index` và `window_end_index` vẫn là chỉ số toàn cục của entity,
không phải chỉ số bắt đầu từ 0 của đoạn stream.

### 1.3 Range chính thức cho ba entity

Các range dưới đây chứa ba anomaly span đầu tiên, thêm prefix 100 time-step
trước span đầu tiên và suffix 100 time-step sau span cuối cùng.

| Entity | Ba anomaly span được chọn | Absolute range | Độ dài range | Số windows với `w=20`, `stride=1` |
|---|---|---:|---:|---:|
| `machine-1-6` | `[246,252)`, `[653,658)`, `[2092,2100)` | `[146,2200)` | 2,054 | 2,035 |
| `machine-3-4` | `[2734,3520)`, `[4474,4550)`, `[6013,6016)` | `[2634,6116)` | 3,482 | 3,463 |
| `machine-3-9` | `[1199,1230)`, `[5361,5487)`, `[10662,10707)` | `[1099,10807)` | 9,708 | 9,689 |

Các range này áp dụng giống nhau cho THESIS, M2N2, CANDI, Stumpy, KMeansAD
và Isolation Forest. Do đó, khác biệt về performance không đến từ việc mỗi
phương pháp nhìn một đoạn test khác nhau.

## 2. Pretrained encoder contract

### 2.1 Quy tắc theo phương pháp

| Phương pháp | Nguồn tham số encoder | Latent dimension | Cách dùng trong online phase |
|---|---|---:|---|
| THESIS | Stage B `best.pt` của chính combination THESIS | 64 | Load checkpoint; giữ reference encoder và memory theo checkpoint; online chỉ cập nhật projector theo config |
| M2N2 | RedLamp `best.pt` cùng entity và seed | 128 | Chỉ dùng phần encoder làm backbone pretrained; không load RedLamp classification head |
| CANDI | RedLamp `best.pt` cùng entity và seed | 128 | Chỉ dùng phần encoder làm backbone pretrained; không load RedLamp classification head |
| Stumpy, KMeansAD, Isolation Forest | Không có neural encoder | Không áp dụng | Dùng feature/window representation riêng của từng traditional baseline |

### 2.2 Tham số kiến trúc encoder dùng cho deep-learning methods

| Tham số | THESIS | M2N2 | CANDI |
|---|---:|---:|---:|
| `input_dim` | 38 | 38 | 38 |
| `window_size` | 20 | 20 | 20 |
| `encoder_family` | `cnn_simple` | `cnn_simple` | `cnn_simple` |
| Số lớp CNN | 3 | 3 | 3 |
| `kernel_size` | 3 | 3 | 3 |
| `hidden_channels` | 64 | 64 | 64 |
| `dropout` | 0.1 | 0.1 | 0.1 |
| Output latent | 64 | 128 | 128 |
| Tham số online được cập nhật | Projector theo THESIS config | Phần adaptation của M2N2 | Phần adaptation của CANDI |

THESIS và hai baseline dùng cùng input dimension, window size và cấu trúc
CNN cơ bản. Latent dimension được phép khác nhau vì mỗi phương pháp có head và
mục tiêu online riêng. Không thêm projection chỉ để ép RedLamp `128` về
THESIS `64`.

### 2.3 Checkpoint THESIS

THESIS có 18 offline combinations:

```text
2 offline variants (O0, O1)
× 3 entities
× 3 seeds (6, 8, 36)
= 18 Stage B best checkpoints
```

Mỗi online THESIS run phải dùng Stage B checkpoint cùng `offline_variant`,
`entity` và `seed`:

```text
outputs/benchmark/smd/thesis/<O0|O1>/<entity>/seed<seed>/two_stage/
  stage_b_fusion_finetuning/checkpoints/best.pt
```

Danh sách đầy đủ 18 path nằm trong
[stage_b_best_checkpoints.md](../notes/stage_b_best_checkpoints.md) và bản
inventory remote nằm trong
[detail-remote-gpu-checkpoints-inventory.md](../inventories/detail-remote-gpu-checkpoints-inventory.md).

Checkpoint Stage B đại diện cho pipeline THESIS đã train tổng cộng 30 epoch:

```text
Stage A: 25 epoch multi-task learning
Stage B:  5 epoch fusion fine-tuning
Tổng:    30 epoch
```

### 2.4 Checkpoint RedLamp cho M2N2 và CANDI

M2N2 và CANDI dùng checkpoint RedLamp cùng `entity` và `seed`. Không có
`O0/O1` cho hai baseline này. Path canonical trên remote là:

```text
outputs/benchmark/smd/redlamp_baseline/<entity>/seed<seed>/checkpoints/best.pt
```

Có 9 RedLamp checkpoints tương ứng với 3 entity và 3 seed. Cả M2N2 và CANDI
cùng đọc checkpoint tương ứng; không nhân đôi checkpoint thành một bộ tham số
khác nhau cho từng baseline.

Thông số đã chốt của RedLamp encoder:

| Tham số | Giá trị |
|---|---:|
| `input_dim` | 38 |
| `window_size` | 20 |
| `encoder_family` | `cnn_simple` |
| `latent_dim` | 128 |
| `cnn_num_layers` | 3 |
| `cnn_kernel_size` | 3 |
| `cnn_hidden_channels` | 64 |
| `cnn_dropout` | 0.1 |
| Số epoch ghi trong checkpoint metadata | 100 |
| File được chọn | `best.pt` |

Số epoch 100 được chấp nhận. Đây là metadata của checkpoint RedLamp đang có
trên remote, không phải yêu cầu train lại. Checkpoint được chọn theo metric
`val_synth_vus_pr` của quá trình RedLamp. Khi load cho M2N2 hoặc CANDI, runtime
chỉ lấy tensor của encoder và bỏ qua classification head, reconstruction head
và các state không thuộc backbone.

Config cuối cùng của M2N2/CANDI phải biểu diễn rõ các trường tương đương sau:

```yaml
baseline_kwargs:
  input_dim: 38
  window_size: 20
  encoder_family: cnn_simple
  encoder_dim: 128
  cnn_num_layers: 3
  cnn_kernel_size: 3
  cnn_hidden_channels: 64
  cnn_dropout: 0.1
  pretrained_encoder_checkpoint: outputs/benchmark/smd/redlamp_baseline/<entity>/seed<seed>/checkpoints/best.pt
```

`pretrained_encoder_checkpoint` là một phần của contract. Runtime không được
âm thầm train một backbone mới nếu trường này đã được cung cấp.

## 3. Tiêu chí công bằng của benchmark

Benchmark này định nghĩa công bằng ở các điểm sau:

1. Tất cả phương pháp nhìn cùng absolute range của cùng entity.
2. Deep-learning methods dùng cùng input dimension, window size và cấu trúc
   simple 1D-CNN cơ bản.
3. M2N2 và CANDI dùng cùng nguồn RedLamp encoder và cùng quy tắc ghép
   entity/seed.
4. THESIS dùng checkpoint Stage B tương ứng với chính nó, không thay bằng
   RedLamp encoder.
5. Khác biệt latent dimension `64` và `128` được ghi công khai trong bảng
   kết quả. Không gọi hai dimension này là “bằng nhau”.
6. Traditional ML baselines không bị ép phải có encoder vì chúng không phải
   deep-learning methods.

Việc THESIS dùng pipeline 30 epoch còn M2N2/CANDI dùng RedLamp checkpoint có
metadata 100 epoch là một giới hạn thực nghiệm đã được chấp nhận. Mục tiêu hiện
tại là so sánh online adaptation trên cùng stream, với nguồn encoder được ghi
rõ và tái lập được; không train lại encoder cho từng combination.

## 4. Provenance bắt buộc

Mỗi online result phải lưu tối thiểu:

| Nhóm | Trường cần lưu |
|---|---|
| Stream | `entity_id`, `absolute_start_index`, `absolute_end_index`, `source_sequence_length`, `sequence_length` |
| Model | `method`, `seed`, `online_variant`, `encoder_family`, latent dimension |
| Checkpoint | checkpoint path, checkpoint role, SHA-256, nguồn `THESIS Stage B` hoặc `RedLamp` |
| Protocol | `window_size`, `stride`, `max_online_steps`, threshold config |
| Status | `main` hoặc `smoke`, partial-test coverage và mọi giới hạn runtime |

SHA-256 phải được tính trên đúng file checkpoint được load. Không dùng metric
hoặc tên file thay cho checkpoint hash.

## 5. Trạng thái implementation và điều kiện nghiệm thu

Absolute range contract đã có trong
[`src/protocols/online_stream_range.py`](../../src/protocols/online_stream_range.py)
và được runner dùng trước windowization. Các test range cũng kiểm tra offset
entity-global.

Tại thời điểm chốt tài liệu, code baseline online vẫn còn đường chạy tự tạo
`SimpleWindowCnnAutoencoder` và train backbone ngắn trong `adaptive.py`. Đường
chạy đó chưa đáp ứng đầy đủ quyết định RedLamp checkpoint ở mục 2.4. Vì vậy,
trước benchmark chính thức cần kiểm tra các điều kiện sau:

- M2N2 và CANDI đọc đúng `pretrained_encoder_checkpoint`.
- Loader xác nhận shape encoder là `38 -> 64 -> 64 -> 128` theo các Conv1d.
- Loader không load nhầm classification head của RedLamp.
- Runtime không gọi lại backbone training khi đã có checkpoint.
- Một smoke run ghi đúng checkpoint path, SHA-256, latent dimension và
  absolute range.
- Smoke tests cho THESIS, M2N2 và CANDI đều đi qua cùng range contract.

Chỉ sau khi một combination chạy end-to-end và các điều kiện trên pass mới
chạy toàn bộ matrix.

## 6. Terminology changes

Spec này không đổi tên runtime object đã có. Mapping chính thức là:

| Tên dùng trong spec | Runtime object | Trạng thái |
|---|---|---|
| absolute range contract | `absolute_start_index` + `absolute_end_index` và `select_online_stream_sequence` | unchanged |
| Stage B best checkpoint | `stage_b_fusion_finetuning/checkpoints/best.pt` | unchanged |
| RedLamp pretrained encoder | encoder tensors lấy từ RedLamp `best.pt` | new benchmark contract |
| traditional ML baseline encoder | Không có | not applicable, không phải alias |

`absolute time range`, `absolute range` và `online stream range` trong các trao
đổi trước đều chỉ cùng một contract ở trên. Tên canonical trong config vẫn là
`absolute_start_index` và `absolute_end_index`.

## Tài liệu và source liên quan

- [SMD anomaly spans và partial stream](../notes/online_tta_partial_stream_anomaly_spans.md)
- [Stage B best checkpoints](../notes/stage_b_best_checkpoints.md)
- [Online benchmark matrix](../inventories/online-benchmark-combinations-and-smoke-checklist.md)
- [`online_stream_range.py`](../../src/protocols/online_stream_range.py)
- [`adaptive.py`](../../src/baselines/online/adaptive.py)
- [`neural_blocks.py`](../../src/models/neural_blocks.py)
- [`generate_online_streaming_benchmark_configs.py`](../../scripts/benchmarks/generate_online_streaming_benchmark_configs.py)


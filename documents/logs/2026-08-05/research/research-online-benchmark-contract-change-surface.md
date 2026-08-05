---
date: 2026-08-05T17:21:05+07:00
researcher: OpenAI Codex
topic: "Xác định code cần điều chỉnh để đáp ứng online_benchmark_contract.md"
status: complete
revision: 5f3ac2654c3a4d9b5a0d25cc5b08672c9b0fe70c
branch: dev
---

# Research: Xác định code cần điều chỉnh để đáp ứng online benchmark contract

## Summary

Contract absolute range đã được triển khai trong runtime baseline và runtime
THESIS, nhưng generator của THESIS chưa đưa range vào config. Vì vậy, các file
config THESIS hiện tại vẫn có thể stream toàn bộ test series.

M2N2 và CANDI chưa đáp ứng contract pretrained encoder. Runtime hiện tạo và
train một `SimpleWindowCnnAutoencoder` mới với `encoder_dim=64`. Runtime chưa
đọc checkpoint RedLamp, chưa kiểm tra latent `128`, và chưa ghi provenance của
checkpoint vào kết quả.

Generator baseline cũng chưa đáp ứng ma trận combination đã chốt. Source
generator vẫn tạo `A0`, `A1`, `A2` cho M2N2 và CANDI. Nhiều file generated cũ
của cả traditional ML cũng còn các nhãn `A0`, `A1`, `A2`, dù contract yêu cầu
traditional ML dùng một variant `main` và chạy frozen.

Traditional ML scoring flow hiện đã gần khớp contract: online runner dùng các
class trong `src/baselines/online/frozen.py`, dùng window size 20, stride 1,
clean-validation threshold, EWMA và không update test stream. Phần cần điều
chỉnh chính của traditional ML nằm ở variant/config/provenance, không nằm ở
thuật toán score cốt lõi.

## Research question

Sử dụng `prompts/1_research_prompt.md` để trace codebase hiện tại và xác định
những file, symbol, config và test cần điều chỉnh để thỏa
`documents/spec/online_benchmark_contract.md`.

Phạm vi gồm entry point online benchmark, absolute range, protocol chung,
checkpoint encoder, deep-learning baselines M2N2/CANDI, traditional ML
baselines, artifact provenance và test coverage.

## System context

Codebase có hai online entry point khác nhau:

1. `scripts/benchmarks/run_online_streaming_benchmark.py` chạy M2N2, CANDI,
   Stumpy, KMeansAD và Isolation Forest. `BASELINE_BUILDERS` ánh xạ tên
   baseline sang các class trong `src/baselines/online/`.
2. `scripts/benchmarks/run_thesis_online_benchmark.py` gọi runtime THESIS
   trong `src/engine/online_tta/`.

Hai entry point dùng chung ý tưởng absolute range nhưng dùng hai đường config
khác nhau. Vì vậy, chỉnh baseline runner không tự động chỉnh được THESIS
runner.

## Execution path

### Baseline online runner

```text
--benchmark-config
  -> load benchmark config và protocol config
  -> validate protocol
  -> build SMD dataset
  -> lấy train, clean validation và test sequence
  -> instantiate baseline từ BASELINE_BUILDERS
  -> fit/initialize baseline
  -> calibrate threshold trên clean validation
  -> select absolute test range
  -> áp dụng max_online_steps nếu là smoke
  -> run_sequence trên test stream
  -> ghi threshold, metrics, records và benchmark report
```

`run_online_streaming_benchmark()` thực hiện flow này ở
`scripts/benchmarks/run_online_streaming_benchmark.py:239-413`. Runner cắt
absolute range tại dòng 339-343, trước khi gọi `run_sequence()` ở dòng
373-378.

### THESIS online runner

```text
--experiment-config
  -> resolve Stage B checkpoint và threshold artifact
  -> build THESIS online execution context
  -> lấy test sequence
  -> select absolute range
  -> run A0/A1/A2 online TTA
  -> save records, report và final online checkpoint
```

THESIS chọn range tại
`src/engine/online_tta/online_engine_run.py:517-526`. Tuy nhiên, generator
THESIS hiện không ghi hai key absolute range vào `task_overrides`, nên đoạn
runtime này thường nhận `None` và dùng toàn bộ test series.

## Detailed findings

### 1. Absolute range

#### Implemented

`src/protocols/online_stream_range.py:8-43` đã triển khai contract nửa kín
`[absolute_start_index, absolute_end_index)`. Code:

- bắt buộc hai key xuất hiện cùng nhau;
- kiểm tra `0 <= start < end <= source_length`;
- cắt `x`, `point_labels`, `mask` và `timestamps`;
- giữ `source_sequence_length`, `sequence_length`, `absolute_start_index`
  và `absolute_end_index` trong metadata.

`src/core/config_model_validation.py:214-235` cho phép hai key trong
`online_adaptation` task. Các điều kiện cặp và thứ tự index được kiểm tra tại
`src/core/config_model_validation.py:504-523`. Runtime range selector mới là
nơi kiểm tra upper bound theo chiều dài sequence thật.

Baseline runner gọi selector đúng trước windowization tại
`scripts/benchmarks/run_online_streaming_benchmark.py:337-348`. THESIS runner
cũng gọi selector trước khi chạy stream tại
`src/engine/online_tta/online_engine_run.py:517-526`.

#### Config chưa đáp ứng

`scripts/benchmarks/generate_online_streaming_benchmark_configs.py:90-95`
đã thêm range cho baseline configs. Ví dụ config Isolation Forest hiện có
`[146,2200)` tại
`configs/experiment/online_benchmark/iforest/smd__iforest__online_main__machine_1_6__w20__seed6__main.yaml:17-20`.

Ngược lại, `_task_overrides()` của generator THESIS tại
`scripts/benchmarks/generate_online_benchmark_configs.py:108-137` không có
`absolute_start_index` hoặc `absolute_end_index`. Config THESIS hiện tại cũng
không có hai key này tại
`configs/experiment/online_benchmark/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__main.yaml:30-44`.

#### Code/config cần điều chỉnh

1. `scripts/benchmarks/generate_online_benchmark_configs.py` cần đưa bảng
   range chính thức cho ba entity vào THESIS `task_overrides`.
2. Bộ YAML THESIS dưới `configs/experiment/online_benchmark/thesis/` cần được
   regenerate để mỗi main và smoke config chứa đúng range.
3. Test generator THESIS tại
   `tests/online/test_online_benchmark_config_generation.py` cần kiểm tra hai
   key range và giá trị entity tương ứng.
4. Test runtime THESIS cần giữ một case xác nhận records vẫn mang global
   indices sau khi cắt range. Baseline đã có test tương tự tại
   `tests/online/test_online_stream_range.py:35-54`.

Phần selector hiện có bằng chứng test tốt. Không có bằng chứng cần sửa
`src/protocols/online_stream_range.py` cho contract hiện tại.

### 2. Protocol chung

`configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:1-16` đã chọn đúng:

| Trường | Giá trị hiện tại |
|---|---:|
| `window_size` | `20` |
| `offline_window_stride` | `20` |
| `online_window_stride` | `1` |
| Threshold split | `clean_validation` |
| Threshold quantile | `0.99` |
| EWMA current/previous | `0.9` / `0.1` |
| Test label usage | `metrics_only` |
| Point adjustment | `false` |

Baseline runner lấy `window_size` và threshold quantile từ protocol config tại
`scripts/benchmarks/run_online_streaming_benchmark.py:285-297`. Các baseline
online calibrate EWMA trên clean validation tại
`src/baselines/online/adaptive.py:290-373` và
`src/baselines/online/frozen.py:104-182`.

#### Code/config cần điều chỉnh

Protocol chung đã có implementation tương ứng. Cần bổ sung test kiểm tra
main configs của THESIS, M2N2, CANDI và traditional ML cùng trỏ tới protocol
file này. Hiện test generator baseline kiểm tra protocol path, nhưng test chỉ
đại diện cho baseline CANDI tại
`tests/online/test_online_streaming_benchmark_config_generation.py:30-43`.

### 3. M2N2 và CANDI pretrained RedLamp encoder

#### Implemented hiện tại

`AdaptiveStreamingBaselineBase.__init__()` nhận `encoder_dim`, CNN settings
và backbone training settings tại
`src/baselines/online/adaptive.py:56-117`.

`fit()` luôn gọi `_fit_backbone()` tại dòng 119-125. `_fit_backbone()` luôn:

- tạo `SimpleWindowCnnAutoencoder` mới tại dòng 138-146;
- tạo optimizer Adam tại dòng 147-149;
- tạo normalized train windows tại dòng 150-160;
- train backbone theo `backbone_epochs` tại dòng 161-176.

Đây là mismatch trực tiếp với contract. Contract yêu cầu runtime load
checkpoint RedLamp và không train backbone mới khi checkpoint đã được cung
cấp.

Model hiện tại có thể dùng simple 1D-CNN giống RedLamp. `SimpleWindowCnnEncoder`
tạo chuỗi Conv1d với dimensions
`[input_dim] + [hidden_channels] * (num_layers - 1) + [output_dim]` tại
`src/models/neural_blocks.py:109-133`. RedLamp cũng dùng class này tại
`src/models/baseline_impl/redlamp_baseline.py:162-170`.

RedLamp model config đã ghi đúng `input_dim=38`, `latent_dim=128`,
`encoder_family=cnn_simple`, ba CNN layers, kernel size 3, hidden channels 64
và dropout 0.1 tại `configs/model/redlamp_baseline_comparative_smd.yaml:1-10`.
RedLamp có reconstruction decoder và classification head riêng tại
`src/models/baseline_impl/redlamp_baseline.py:173-188`; online contract chỉ
cho phép lấy encoder tensors, không lấy hai head này.

#### Config hiện tại chưa đáp ứng

Generator baseline đang đặt `encoder_dim=64` cho CANDI tại
`scripts/benchmarks/generate_online_streaming_benchmark_configs.py:98-116` và
cho M2N2 tại dòng 117-134. Generator không ghi
`pretrained_encoder_checkpoint` và vẫn ghi `backbone_epochs`,
`backbone_learning_rate` cùng các tham số dùng để train backbone mới.

Representative CANDI YAML xác nhận trạng thái này tại
`configs/experiment/online_benchmark/candi/smd__candi__online_A0__machine_1_6__w20__seed6__main.yaml:11-26`.

#### Code cần điều chỉnh

1. `src/baselines/online/adaptive.py`, đặc biệt `__init__()`, `fit()` và
   `_fit_backbone()`, cần có đường load RedLamp encoder khi config cung cấp
   checkpoint.
2. `src/baselines/online/m2n2.py:15-64` và
   `src/baselines/online/candi.py:15-64` cần truyền và lưu
   `pretrained_encoder_checkpoint` cùng latent `128` ở public constructor.
3. Loader cần đọc đúng payload checkpoint, lấy `model_state_dict` của
   `encoder`, bỏ qua RedLamp decoder/classification head, kiểm tra shape
   `38 -> 64 -> 64 -> 128`, và báo lỗi nếu shape không khớp.
4. Loader cần ghi path và SHA-256 checkpoint vào method metadata hoặc benchmark
   provenance. Repository đã có hàm SHA-256 dùng được tại
   `src/core/artifact_integrity.py:11-20`.
5. Khi checkpoint tồn tại, runtime không được gọi `_fit_backbone()` training
   path. Các tham số `backbone_epochs`, `backbone_batch_size` và
   `backbone_learning_rate` không được tiếp tục điều khiển main run.

Chưa có checkpoint RedLamp `best.pt` trong local source tree để xác nhận trực
tiếp key names của `model_state_dict`. Vì vậy, key mapping cụ thể của loader
là **unknown** cho tới khi đọc một checkpoint thật trên remote hoặc trong
`outputs/`.

### 4. M2N2/CANDI online adaptation behavior

Sau khi score, `AdaptiveStreamingBaselineBase.run_sequence()` tính triage và
gọi `_should_update()` tại `src/baselines/online/adaptive.py:375-426`.
Nếu update được phép, code chỉ cập nhật `reference_mean_` và
`reference_std_` qua `_update_reference()` tại dòng 245-256. Code không cập
nhật neural backbone parameters trong test stream.

M2N2 quyết định update khi raw và EWMA score đều không vượt threshold tại
`src/baselines/online/m2n2.py:53-68`. CANDI quyết định update khi triage là
`gray_zone` hoặc `pnn_candidate` tại `src/baselines/online/candi.py:53-68`.

`online_variant` chỉ được lưu vào metadata; hai `_should_update()` không dùng
giá trị A0/A1/A2 để thay đổi policy. Đây là evidence cho thấy các config
M2N2/CANDI mang A0/A1/A2 hiện không tạo ra ba thuật toán khác nhau.

#### Code/config cần điều chỉnh

1. Generator cần tạo một baseline config `main` cho mỗi entity/seed của M2N2
   và CANDI, không tạo ba config chỉ khác tên variant.
2. Default `online_variant="A0"` trong
   `src/baselines/online/adaptive.py:56-75`, `m2n2.py` và `candi.py` cần được
   thống nhất với baseline label `main`, nếu runner vẫn yêu cầu field này.
3. `scripts/benchmarks/run_online_streaming_benchmark.py:285-297` đang mặc
   định `online_variant` là `A0`; baseline runner cần dùng `main`.
4. Exact online adaptation objective của M2N2/CANDI không được contract mới
   định nghĩa chi tiết. Vì vậy, research này không kết luận rằng phải thay
   `_should_update()` bằng một thuật toán khác. Contract evidence chỉ đủ để
   kết luận phải bỏ variant giả A0/A1/A2 và làm cho checkpoint/provenance đúng.

### 5. THESIS Stage B checkpoint và range

THESIS checkpoint resolution đã có đường riêng. `resolve_stage_b_checkpoint()`
tìm checkpoint theo `offline_variant`, entity, seed, benchmark mode và stage
name tại `src/engine/online_tta/checkpoint_resolution.py:35-113`.

THESIS runner ghi reference checkpoint vào experiment config tại
`scripts/benchmarks/run_thesis_online_benchmark.py:245-250` rồi chạy online
engine. Đây là phần đã đáp ứng quan hệ checkpoint Stage B theo metadata.

Range là phần còn thiếu trong generator THESIS như đã nêu ở mục 1. Threshold
artifact THESIS cũng có contract riêng: schema V4 yêu cầu checkpoint SHA-256 và
các triage thresholds tại `src/protocols/threshold_artifact.py:270-282`.

#### Code/config cần điều chỉnh

1. Thêm range vào THESIS generator và regenerate configs.
2. Giữ mapping `offline_variant/entity/seed` với Stage B checkpoint.
3. Không dùng RedLamp checkpoint thay cho THESIS checkpoint.
4. Không thay đổi THESIS A0/A1/A2 semantics khi sửa baseline config.

### 6. Traditional ML baselines

#### Active online implementation

Online runner import các class frozen tại
`scripts/benchmarks/run_online_streaming_benchmark.py:26-33` và đăng ký chúng
tại `BASELINE_BUILDERS` ở dòng 46-52. Vì vậy, code active cho online benchmark
là `src/baselines/online/frozen.py`, không phải trực tiếp các class offline
trong `src/baselines/traditional/`.

`_FrozenStreamingBaseline` calibrate trên clean validation, dùng EWMA và
`run_sequence()` không update test stream tại
`src/baselines/online/frozen.py:104-250`.

Các tham số algorithm hiện khớp main contract:

| Method | Evidence hiện tại |
|---|---|
| Stumpy | `window_size=20`, `normalize=True`, `p=2.0` tại `frozen.py:256-305`; channel scores gộp bằng `nanmax` tại dòng 288-297 |
| KMeansAD | `n_clusters=20`, `normalize_windows=True` tại `frozen.py:309-332`; `n_init=10` tại dòng 334-344 |
| Isolation Forest | `n_estimators=100`, `max_samples="auto"`, `max_features=1.0`, `contamination="auto"`, `normalize_windows=True` tại `frozen.py:373-419` |
| Shared online protocol | stride-1 window matrix tại `src/baselines/online/base.py:34-57`; threshold clean validation và EWMA trong `frozen.py:104-182` |

#### Config/matrix mismatch

Generator source hiện đặt traditional methods là `main` tại
`scripts/benchmarks/generate_online_streaming_benchmark_configs.py:27-34`.
Representative main YAML cũng có `online_variant: main` và các range đúng tại
`configs/experiment/online_benchmark/iforest/smd__iforest__online_main__machine_1_6__w20__seed6__main.yaml:1-20`.

Tuy nhiên, generated tree còn nhiều file `A0`, `A1` và `A2` bên cạnh file
`main`. Generator không xóa file cũ khi chạy lại, nên filesystem hiện tại
không phải là matrix contract. Những file stale này không được launch.

#### Code/config cần điều chỉnh

1. Giữ generator source chỉ tạo `main` cho Stumpy, KMeansAD và Isolation
   Forest.
2. Regenerate hoặc loại khỏi launch manifest các YAML traditional cũ mang
   `A0/A1/A2`.
3. Đổi default `online_variant="A0"` trong `_FrozenStreamingBaseline` tại
   `src/baselines/online/frozen.py:62-81` thành `main` để config thiếu field
   không âm thầm tạo nhãn THESIS.
4. Mở rộng `_method_metadata()` của frozen methods để ghi toàn bộ
   hyperparameter main vào result provenance. Hiện Stumpy ghi normalize/p và
   KMeans/Isolation Forest chủ yếu ghi normalization cùng variant tại
   `frozen.py:299-305`, `363-370` và `435-443`.

Không có evidence cần thay thuật toán Stumpy, KMeansAD hoặc Isolation Forest
để đạt protocol hiện tại.

### 7. Result provenance và artifact schema

Baseline runner tạo threshold artifact tại
`scripts/benchmarks/run_online_streaming_benchmark.py:320-332`, sau đó ghi
report, metrics và records tại dòng 382-413. Runner hiện không đưa
`calibration["method_metadata"]` vào report `online_execution`.

Contract yêu cầu result lưu method, seed, variant, encoder metadata,
checkpoint path/role/SHA-256, stream range và protocol. Baseline report hiện
chỉ lưu một phần: benchmark config, protocol, threshold artifact, stream
selection, metrics và records.

`build_threshold_artifact()` đã có các trường `checkpoint_sha256` và
`resolved_config_sha256` tại `src/protocols/threshold_artifact.py:237-269`,
nhưng baseline runner không truyền checkpoint SHA-256 vào đó. Artifact schema
V3 của baseline cũng không bắt buộc checkpoint hash như THESIS V4.

#### Code cần điều chỉnh

1. `scripts/benchmarks/run_online_streaming_benchmark.py` cần nhận metadata
   từ baseline calibration và ghi vào report/provenance.
2. M2N2/CANDI cần truyền pretrained checkpoint path và SHA-256 vào threshold
   artifact hoặc một provenance manifest tương ứng.
3. Traditional ML cần ghi rõ `checkpoint_role: not_applicable` thay vì tạo
   checkpoint giả; method metadata vẫn phải ghi đầy đủ hyperparameter.
4. Cần test rằng report main có stream range, method metadata và provenance
   đúng với config đã resolve.

`src/core/artifact_integrity.py:11-20` đã có primitive hash file. Available
evidence không yêu cầu tạo thêm cơ chế hash mới.

## Required code change surface

| Mức | File/symbol | Mismatch cần xử lý | Test hoặc bằng chứng cần cập nhật |
|---|---|---|---|
| Blocker | `scripts/benchmarks/generate_online_benchmark_configs.py:_task_overrides` | Thiếu absolute range cho THESIS | `tests/online/test_online_benchmark_config_generation.py` và một THESIS runtime range test |
| Blocker | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:BENCHMARK_METHOD_VARIANTS` | M2N2/CANDI vẫn tạo A0/A1/A2 | `tests/online/test_online_streaming_benchmark_config_generation.py` |
| Blocker | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:_baseline_kwargs` | Deep baseline latent 64, thiếu RedLamp checkpoint | Generator config assertions |
| Blocker | `src/baselines/online/adaptive.py:fit/_fit_backbone` | Luôn train backbone mới | Checkpoint load/no-training test |
| Blocker | `src/baselines/online/m2n2.py`, `candi.py` | Public constructor chưa nhận checkpoint và latent128 contract | M2N2/CANDI checkpoint fixture tests |
| High | `scripts/benchmarks/run_online_streaming_benchmark.py` | Default A0 và thiếu method/checkpoint provenance | Wrapper report assertions |
| High | `src/baselines/online/frozen.py` | Default frozen variant A0; metadata thiếu đủ hyperparameters | Frozen baseline metadata tests |
| High | Generated YAML dưới `configs/experiment/online_benchmark/` | Stale baseline A0/A1/A2; THESIS thiếu range | Regeneration/matrix count check |
| Medium | `src/protocols/threshold_artifact.py` integration | Baseline checkpoint hash chưa được truyền | Artifact provenance test |
| No change evidenced | `src/protocols/online_stream_range.py` | Range slicing và global offset đã đúng | Existing range tests pass |
| No change evidenced | `src/baselines/online/frozen.py` scoring algorithms | Main traditional hyperparameters và frozen flow đã đúng | Existing baseline flow tests pass |

## Tests and validation

Đã chạy narrow read-only runtime checks bằng:

```text
.venv/bin/python -m pytest -q \
  tests/online/test_online_streaming_baseline_contracts.py \
  tests/online/test_online_stream_range.py \
  tests/online/test_online_streaming_benchmark_wrapper.py
```

Kết quả: `6 passed`, một cảnh báo `joblib` về không đọc được số physical
cores. Cảnh báo này không làm test fail.

Các test pass xác nhận implementation hiện tại có thể calibrate/run các
baseline, giữ global indices, áp dụng absolute range và giới hạn smoke steps.
Chúng chưa xác nhận contract mới vì:

- baseline test vẫn kỳ vọng M2N2/CANDI `encoder_dim == 64` tại
  `tests/online/test_online_streaming_baseline_contracts.py:110-116`;
- config-generation test vẫn kỳ vọng CANDI `online_variant == A0` tại
  `tests/online/test_online_streaming_benchmark_config_generation.py:30-41`;
- không có test load RedLamp checkpoint;
- không có test cấm backbone training khi checkpoint tồn tại;
- không có test THESIS generated config chứa absolute range;
- không có report test cho checkpoint SHA-256 và method metadata của baseline.

## Configuration observed

| Setting | Active value | Evidence | Scope |
|---|---:|---|---|
| `window_size` | `20` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:2` | Shared protocol |
| `online_window_stride` | `1` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:7` | Shared protocol |
| `online_threshold_quantile` | `0.99` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:8-9` | Shared protocol |
| EWMA | `0.9 / 0.1` | `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:10-11` | Shared protocol |
| THESIS latent | `64` | `scripts/benchmarks/generate_online_benchmark_configs.py:93-105` | THESIS online model |
| RedLamp latent | `128` | `configs/model/redlamp_baseline_comparative_smd.yaml:1-10` | Pretrained source |
| Current M2N2/CANDI online latent | `64` | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:98-134` | Current implementation, contract mismatch |
| KMeansAD main clusters | `20` | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:143-150` | Traditional main config |
| Isolation Forest main trees | `100` | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:151-158` | Traditional main config |
| Main `max_online_steps` | `null` | Representative YAMLs under `configs/experiment/online_benchmark/` | Main run |

## Conflicts and uncertainties

1. The normative `online_benchmark_contract.md` says M2N2/CANDI use one
   baseline `main` configuration, while current source generator and many
   generated YAMLs use A0/A1/A2. Active source/config and contract disagree.
2. `documents/inventories/online-benchmark-combinations-and-smoke-checklist.md:13-19`
   agrees with the contract and says A0/A1/A2 belong to THESIS only. The
   generator is therefore the conflicting active source that needs correction.
3. `documents/spec/full-spec-v2.md:14-20` describes A0/A1/A2 as THESIS online
   variants and should not be read as evidence that traditional baselines need
   those labels.
4. The contract says M2N2/CANDI use the RedLamp encoder, but the local
   repository does not contain the real remote RedLamp `best.pt` payload. The
   exact state-dict key names and checkpoint metadata are unknown locally.
5. The contract names M2N2 and CANDI policies but does not define a new
   objective for their online update. Current code updates reference mean/std,
   not neural parameters. This report identifies the fact and the affected
   symbols; it does not invent a replacement adaptation objective.
6. Current baseline threshold artifacts use the generic non-THESIS schema path.
   The available contract requires checkpoint provenance for deep baselines,
   but does not require converting all baselines to THESIS V4 triage fields.

## Open questions

1. What exact keys does the remote RedLamp `best.pt` use for the encoder state
   dictionary, and does its checkpoint include the expected `cnn_simple`
   architecture metadata?
2. Should the online M2N2/CANDI decoder remain baseline-owned after loading only
   the RedLamp encoder, or does an existing external baseline checkpoint define
   a compatible reconstruction head? The current contract only settles that the
   RedLamp classification and reconstruction heads are not loaded.
3. Should `online_variant: main` be mandatory in baseline YAMLs, or should the
   baseline runner omit that field and derive a method-specific label? Current
   contract allows `main` for runner compatibility; active code still defaults
   to `A0`.

## Evidence

- `prompts/1_research_prompt.md:1-372` — research workflow, evidence labels,
  execution-path tracing and report format used here.
- `documents/spec/online_benchmark_contract.md:1-408` — target absolute range,
  encoder, protocol, traditional baseline and provenance contract.
- `scripts/benchmarks/run_online_streaming_benchmark.py:26-52` — active baseline
  imports and builder map.
- `scripts/benchmarks/run_online_streaming_benchmark.py:239-413` — baseline
  runner flow, range selection, calibration and output writing.
- `src/engine/online_tta/online_engine_run.py:517-526` — active THESIS range
  selection call.
- `src/protocols/online_stream_range.py:8-43` — absolute range implementation.
- `src/core/config_model_validation.py:214-235,504-523` — online task keys and
  range validation.
- `scripts/benchmarks/generate_online_benchmark_configs.py:108-137` — THESIS
  task config generation without absolute range.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:27-34,98-158`
  — baseline variants and current baseline hyperparameters.
- `src/baselines/online/adaptive.py:56-176` — current fresh backbone creation
  and training path.
- `src/baselines/online/adaptive.py:375-466` — current deep baseline scoring and
  reference-stat update flow.
- `src/models/neural_blocks.py:61-133` — shared simple 1D-CNN structure.
- `src/models/baseline_impl/redlamp_baseline.py:32-43,162-188` — RedLamp
  encoder and separate heads.
- `src/baselines/online/frozen.py:59-253,256-443` — frozen traditional online
  flow and method-specific scores.
- `tests/online/test_online_streaming_baseline_contracts.py:52-123` — current
  baseline runtime test and its latent-64 expectation.
- `tests/online/test_online_streaming_benchmark_config_generation.py:17-54` —
  current generated-config count and A0 expectation.
- `tests/online/test_online_stream_range.py:35-54` — existing range metadata
  test.
- `src/core/artifact_integrity.py:11-20` — existing SHA-256 file primitive.


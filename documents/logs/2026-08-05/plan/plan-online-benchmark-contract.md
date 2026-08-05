---
date: 2026-08-05 17:39:36 +07:00
planner: OpenAI Codex
topic: "Điều chỉnh codebase theo online benchmark contract"
status: ready
revision: 5f3ac2654c3a4d9b5a0d25cc5b08672c9b0fe70c
branch: dev
related_research: documents/logs/2026-08-05/research/research-online-benchmark-contract-change-surface.md
---

# Implementation Plan: Điều chỉnh codebase theo online benchmark contract

## Summary

Codebase cần được điều chỉnh theo năm nhóm chính:

1. Mọi online config phải chọn đúng absolute range của test series.
2. THESIS vẫn dùng Stage B `best.pt`; M2N2 và CANDI dùng RedLamp encoder
   checkpoint với latent dimension `128`.
3. M2N2 và CANDI chỉ có một baseline config `main` cho mỗi entity và seed.
   Traditional ML cũng dùng `main` và frozen.
4. Kết quả phải ghi đủ stream range, protocol, method metadata, checkpoint
   role và SHA-256.
5. Test, preflight và một smoke run phải chứng minh đúng matrix trước khi chạy
   benchmark chính.

Kế hoạch giữ nguyên score và adaptation policy hiện tại của từng baseline khi
contract chưa định nghĩa objective mới. Kế hoạch không train lại RedLamp
encoder cho từng combination.

## Request

Sử dụng `prompts/2_plan_prompt.md` để lập các phase tuần tự nhằm làm cho
codebase đáp ứng
`documents/spec/online_benchmark_contract.md`.

Phạm vi gồm absolute range, protocol chung, checkpoint encoder, M2N2, CANDI,
traditional ML baselines, generated configs, result provenance, test và
preflight. Chỉ lập kế hoạch; chưa sửa source code, config hay test trong phase
này.

## Current state

Baseline runner và THESIS runner đã có cơ chế cắt range bằng
`select_online_stream_sequence`, nhưng generator THESIS chưa ghi hai trường
absolute range vào config. Vì vậy, config THESIS có thể vẫn stream toàn bộ test
series.

M2N2 và CANDI hiện tạo một `SimpleWindowCnnAutoencoder` mới rồi train ngắn
trong `AdaptiveStreamingBaselineBase._fit_backbone()`. Generator đặt latent
dimension là `64`, chưa ghi `pretrained_encoder_checkpoint`, và chưa có đường
load RedLamp encoder.

Generator baseline hiện tạo `A0`, `A1`, `A2` cho M2N2 và CANDI. Runtime của
hai baseline chỉ dùng `online_variant` như metadata; các nhãn này không tạo ra
ba policy khác nhau. Preflight cũng đang đếm `81` online baseline configs,
trong khi contract mới yêu cầu `45`: 9 M2N2, 9 CANDI và 27 traditional ML.

Traditional ML scoring flow trong
`src/baselines/online/frozen.py` đã gần khớp contract: fit trên train,
calibrate trên clean validation, score stride `1`, EWMA và frozen trong test.
Phần còn lệch chủ yếu là default variant, generated config cũ và provenance.

## Desired end state

Runtime flow chính phải có dạng:

```text
contract ranges/checkpoint inventory
  -> generate one valid config per matrix cell
  -> validate config and checkpoint identity
  -> fit/calibrate without test labels
  -> select [absolute_start_index, absolute_end_index)
  -> run stride-1 stream with global indices
  -> write metrics, records and complete provenance
  -> pass one smoke before the full matrix
```

Ma trận main sau khi hoàn tất:

| Nhóm | Số combination |
| --- | ---: |
| THESIS | 54 |
| M2N2 | 9 |
| CANDI | 9 |
| Stumpy Channel AB | 9 |
| KMeansAD | 9 |
| Isolation Forest | 9 |
| Tổng | 99 |

## Scope

### In scope

- Thêm ba absolute ranges đã chốt vào generator THESIS và xác nhận baseline
  generator giữ đúng range.
- Chuẩn hóa variant baseline thành `main`.
- Load RedLamp `best.pt` cho encoder M2N2/CANDI với latent `128`, không load
  classification head hoặc reconstruction head của RedLamp.
- Bảo đảm checkpoint có shape và provenance rõ ràng.
- Ghi method metadata, checkpoint metadata, SHA-256 và stream metadata vào
  result.
- Cập nhật generated-config tests, runtime-flow tests, preflight và checklist
  hiện hành.
- Chạy một end-to-end smoke rồi mới chuẩn bị full matrix.

### Out of scope

- Không thay đổi Stage B checkpoint của THESIS.
- Không ép RedLamp latent `128` về THESIS latent `64`.
- Không đổi công thức adaptation của M2N2/CANDI khi contract chưa yêu cầu.
- Không ép traditional ML dùng neural encoder.
- Không dùng test labels để calibrate hoặc chọn hyperparameter.
- Không sửa lịch sử của `full-spec-v2.md` thành một spec mới. Nếu tài liệu đó
  còn mô tả matrix cũ, chỉ ghi rõ nó là legacy và dùng contract hiện hành làm
  nguồn chuẩn.
- Không xóa rộng toàn bộ generated output. Các file stale chỉ được xóa sau
  khi xác định đúng danh sách; trước mắt chúng không được đưa vào launch
  manifest hoặc báo cáo.

## Evidence

- `prompts/2_plan_prompt.md:1-372` — quy trình lập plan, yêu cầu trace runtime,
  chia phase, verification và rollback.
- `codebase_preferences.md:30-86` — vị trí log, matrix smoke trước full run,
  artifact provenance và nguyên tắc đơn giản hóa.
- `documents/spec/online_benchmark_contract.md:1-408` — contract normative về
  range, encoder, matrix, traditional ML và provenance.
- `documents/logs/2026-08-05/research/research-online-benchmark-contract-change-surface.md:12-490`
  — research report đã trace entry point, runtime, config, test và change
  surface.
- `src/protocols/online_stream_range.py:8-43` — selector đã cắt range nửa kín
  và giữ global offset metadata.
- `scripts/benchmarks/run_online_streaming_benchmark.py:239-413` — active
  baseline flow, calibration, range selection và report writing.
- `src/engine/online_tta/online_engine_run.py:517-526` — active THESIS range
  selection.
- `scripts/benchmarks/generate_online_benchmark_configs.py:108-137` — THESIS
  generator hiện thiếu range.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:27-158`
  — baseline variants và hyperparameters hiện tại.
- `src/baselines/online/adaptive.py:56-176` — M2N2/CANDI hiện tạo và train
  backbone mới.
- `src/baselines/online/m2n2.py:15-76` và
  `src/baselines/online/candi.py:15-80` — public constructor và policy riêng.
- `src/baselines/online/frozen.py:59-443` — active traditional ML runtime.
- `scripts/ops/preflight_full_benchmark_matrix.py:122-148` — preflight hiện
  đếm baseline neural theo ba variant.
- `tests/online/test_online_streaming_benchmark_config_generation.py:17-54`,
  `tests/online/test_online_streaming_baseline_contracts.py:52-123` và
  `tests/benchmarks/test_full_benchmark_matrix_preflight.py:8-27` — test hiện
  đang kỳ vọng behavior cũ.

## Terminology mapping

| Tên hiện hành hoặc cũ | Tên trong contract | Trạng thái | Quy tắc triển khai |
| --- | --- | --- | --- |
| `absolute_start_index` + `absolute_end_index` | absolute range | unchanged | Giữ nguyên field và selector hiện có. |
| THESIS `A0`, `A1`, `A2` | THESIS online variants | unchanged | Không đổi semantics THESIS. |
| M2N2/CANDI `A0`, `A1`, `A2` | baseline `main` | deprecated config usage, không phải alias | Không sinh thêm combination theo A-variant. |
| `online_variant: main` | baseline configuration label | new standardized label | Dùng cho M2N2, CANDI và traditional ML. |
| fresh online CNN | RedLamp pretrained encoder | replaced for main baseline run | Không âm thầm train encoder mới khi checkpoint có mặt. |
| `pretrained_encoder_checkpoint` | RedLamp checkpoint path | new contract field | Path phải đi cùng entity và seed. |
| Stage B `best.pt` | THESIS reference checkpoint | unchanged | Resolve theo offline variant, entity và seed. |
| traditional ML encoder | not applicable | unchanged | Không tạo checkpoint giả cho traditional ML. |

Các tên `absolute time range`, `absolute range` và `online stream range` trong
trao đổi trước không tạo runtime object mới. Tên canonical trong config vẫn là
`absolute_start_index` và `absolute_end_index`.

## Implementation approach

Thực hiện theo các phase nhỏ, mỗi phase có test riêng và để codebase chạy được.
Trước khi viết loader, kiểm tra một RedLamp `best.pt` thật để biết payload và
key state-dict. Không tự đoán key name.

Tách trách nhiệm theo flow hiện tại:

- generator sở hữu matrix và config;
- baseline runner sở hữu instantiate, calibration, range selection và report;
- adaptive baseline sở hữu encoder initialization và online score/policy;
- frozen baseline sở hữu traditional score;
- artifact/provenance layer sở hữu hash và metadata;
- preflight sở hữu kiểm tra count, naming và path trước main run.

Không thêm một runtime framework mới. Nếu logic load checkpoint làm
`adaptive.py` vượt giới hạn readability của repo, đặt phần đọc payload vào một
helper nhỏ được đề xuất là
`src/baselines/online/redlamp_encoder_checkpoint.py`. Helper này chỉ đọc,
validate và trả encoder state; nó không sở hữu online training loop.

## Stage dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 0 | Checkpoint thật và generated tree hiện tại | Input đã xác nhận cho generator và loader |
| Phase 1 | Matrix target và checkpoint paths từ Phase 0 | Config chính thức và preflight count mới |
| Phase 2 | Config mới từ Phase 1 và payload mapping từ Phase 0 | Runtime M2N2/CANDI dùng encoder đúng contract |
| Phase 3 | Runtime metadata từ Phase 2 và report flow hiện tại | Provenance đầy đủ cho test và smoke |
| Phase 4 | Contract behavior của Phase 1-3 | Regression gate trước end-to-end run |
| Phase 5 | Focused suite pass, checkpoint thật và revision hợp lệ | Quyết định chạy full matrix 99 cells |

Các stage trong cùng một phase chạy theo thứ tự số. Chỉ chuyển sang phase tiếp
theo khi điều kiện hoàn tất của phase hiện tại đã đạt.

## Phase 0: Chốt input checkpoint và matrix trước khi sửa code

### Goal

Biến các đầu vào chưa xác nhận thành bằng chứng cụ thể để implementation không
phải đoán payload RedLamp hoặc danh sách config được phép chạy.

### Stages

#### Stage 0.1: Xác nhận RedLamp checkpoint inventory

- **Kết quả:** Có danh sách 9 checkpoint theo entity và seed, cùng key encoder,
  shape, epoch metadata và SHA-256 đã đọc từ dữ liệu thật.
- **Điều kiện hoàn tất:** Mỗi path được đánh dấu `verified` hoặc `missing`;
  không còn dùng giả định về payload để thiết kế loader.

#### Stage 0.2: Chốt generated-config inventory

- **Kết quả:** Có danh sách config `main`, config `smoke` và các path baseline
  `A0/A1/A2` cũ đang tồn tại.
- **Điều kiện hoàn tất:** Exact stale paths và expected main paths được ghi lại;
  chưa xóa file nào.

#### Stage 0.3: Đóng input gate cho implementation

- **Kết quả:** Checkpoint mapping, matrix target và giới hạn cleanup được coi là
  input chính thức cho các phase sau.
- **Điều kiện hoàn tất:** Nếu checkpoint bắt buộc bị thiếu, phase này kết thúc
  ở trạng thái blocked và không cho phép chạy performance benchmark.

### Changes

#### 1. Kiểm tra RedLamp checkpoint thật

- **File/nguồn:** remote canonical path
  `outputs/benchmark/smd/redlamp_baseline/<entity>/seed<seed>/checkpoints/best.pt`
- **Symbol:** checkpoint payload, state-dict và metadata thực tế.
- **Change:** Đọc read-only ít nhất một checkpoint; nếu có thể, kiểm tra cả 9
  path theo entity/seed. Ghi lại key encoder, shape từng Conv1d, metadata
  `latent_dim=128`, epoch metadata `100`, và SHA-256.
- **Reason:** Local tree chưa có RedLamp `best.pt`; key mapping hiện là
  unknown.
- **Dependencies:** Không chạy loader implementation trước khi có payload mẫu
  hoặc một fixture được tạo từ payload thật.

#### 2. Xác nhận current generated tree

- **File:** `configs/experiment/online_benchmark/` và
  `scripts/ops/preflight_full_benchmark_matrix.py`.
- **Change:** Lập danh sách expected main paths và stale A0/A1/A2 baseline
  paths. Không xóa ở phase này.
- **Reason:** Generator không tự xóa file cũ; chỉ nhìn source generator không
  đủ để chứng minh filesystem đã sạch.

### Verification

#### Automated

- [ ] Một check read-only xác nhận 9 RedLamp path theo entity/seed hoặc báo
  chính xác path còn thiếu.
- [ ] Check đếm file tách `main` khỏi stale `A0/A1/A2`.

#### Manual

- [ ] Xác nhận remote source revision, host key và checkpoint path trước khi
  dùng remote. Không ghi hoặc cleanup remote ở phase này.

### Risks

- **Payload khác dự kiến:** loader phải fail với lỗi về key/shape rõ ràng;
  không dùng `strict=False` để che mismatch.
- **Remote thiếu checkpoint:** dừng ở preflight, không chạy full matrix bằng
  encoder random.

## Phase 1: Chuẩn hóa absolute range và generated matrix

### Goal

Mọi config chính thức mô tả đúng stream và đúng số combination trước khi đụng
vào model runtime.

### Stages

#### Stage 1.1: Gắn absolute range vào mọi generator

- **Kết quả:** THESIS và baseline config đều chứa đúng cặp
  `absolute_start_index` và `absolute_end_index` theo entity.
- **Điều kiện hoàn tất:** Cả main và smoke dùng cùng range contract; chỉ
  `max_online_steps` được phép làm smoke ngắn hơn.

#### Stage 1.2: Chuẩn hóa baseline matrix và checkpoint reference

- **Kết quả:** M2N2 và CANDI chỉ sinh variant `main`, dùng latent `128` và
  trỏ tới RedLamp checkpoint đúng entity/seed; traditional ML vẫn dùng
  `main`.
- **Điều kiện hoàn tất:** Generator không còn tạo baseline neural
  `A0/A1/A2` cho matrix chính.

#### Stage 1.3: Đồng bộ runtime defaults và preflight

- **Kết quả:** Runner, constructors và preflight hiểu `main` là nhãn baseline
  canonical và tính đúng 45 baseline combinations.
- **Điều kiện hoàn tất:** Preflight không nhận stale variant làm một cell hợp
  lệ của matrix mới.

#### Stage 1.4: Tái tạo và kiểm tra config tree

- **Kết quả:** Generated YAML và checklist hiện hành phản ánh 54 THESIS,
  45 baseline và tổng 99 comparison cells.
- **Điều kiện hoàn tất:** Count, path, range, protocol và diff đều được kiểm
  tra trước khi sang runtime model.

### Changes

#### 1. THESIS generator

- **File:** `scripts/benchmarks/generate_online_benchmark_configs.py`
- **Symbol:** `_task_overrides()`.
- **Change:** Thêm bảng range cho `machine_1_6`, `machine_3_4` và
  `machine_3_9` vào `task_overrides`, với hai field xuất hiện cùng nhau cho cả
  main và smoke. Giữ `max_online_steps: null` trong main và giá trị smoke
  hiện có trong smoke.
- **Reason:** Runtime selector đã đúng; config generator mới là missing link.
- **Dependencies:** Không sửa `select_online_stream_sequence` nếu test hiện có
  tiếp tục pass.

#### 2. Baseline generator

- **File:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`
- **Symbols:** `BENCHMARK_METHOD_VARIANTS`, `_baseline_kwargs()` và
  `build_online_streaming_benchmark_config()`.
- **Change:** Đổi M2N2/CANDI thành `("main",)`. Ghi `encoder_dim=128` và
  `pretrained_encoder_checkpoint` theo entity/seed. Bỏ các field training
  backbone khỏi main contract hoặc làm chúng không còn được runtime đọc.
  Giữ traditional ML `main`, smoke size và hyperparameter main đã chốt.
- **Reason:** Contract yêu cầu 9 M2N2 + 9 CANDI, không phải 27 + 27.
- **Dependencies:** Phase 0 cung cấp path mẫu và loader contract.

#### 3. Default label của runner và baseline constructors

- **Files:** `scripts/benchmarks/run_online_streaming_benchmark.py`,
  `src/baselines/online/adaptive.py`, `src/baselines/online/m2n2.py`,
  `src/baselines/online/candi.py` và `src/baselines/online/frozen.py`.
- **Symbols:** `online_variant` defaults và baseline instantiation.
- **Change:** Default baseline label thành `main`. Config explicit `main` vẫn
  được ưu tiên; không map `main` thành THESIS A0.
- **Reason:** Config thiếu field không được âm thầm sinh nhãn THESIS.

#### 4. Preflight matrix

- **File:** `scripts/ops/preflight_full_benchmark_matrix.py`
- **Symbols:** `_validate_wrapper_configs()`, baseline loop và report count.
- **Change:** Đếm CANDI/M2N2 như traditional ML: một pattern `online_main`,
  mỗi method 9 main configs. Đổi report từ `baselines: 81` thành
  `baselines: 45`.
- **Reason:** Preflight phải ngăn full run sai matrix.

#### 5. Generated YAML và current checklist

- **Files:** generated YAML dưới
  `configs/experiment/online_benchmark/` và
  `documents/inventories/online-benchmark-combinations-and-smoke-checklist.md`.
- **Change:** Regenerate file theo generator mới. Cập nhật checklist hiện
  hành từ 72 deep online runs thành 99 comparison cells nếu tài liệu mô tả
  toàn bộ contract; ghi riêng 54 THESIS và 45 baselines. Giữ
  `full-spec-v2.md` là tài liệu lịch sử, không sửa semantics lịch sử.
- **Reason:** Source đúng nhưng inventory/preflight sai vẫn có thể dẫn tới
  launch sai.
- **Dependencies:** Cần xác định exact stale paths ở Phase 0. Xóa stale file
  chỉ sau manifest và diff review; nếu chưa cần, chỉ loại chúng khỏi launch.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py tests/benchmarks/test_full_benchmark_matrix_preflight.py` — test generator và preflight pass với range và count mới.
- [ ] `.venv/bin/python -m scripts.benchmarks.generate_online_benchmark_configs --print-count` — sinh đúng 108 file THESIS gồm main và smoke.
- [ ] `.venv/bin/python -m scripts.benchmarks.generate_online_streaming_benchmark_configs --print-count` — sinh đúng 90 file baseline gồm 45 main và 45 smoke.
- [ ] `git diff --check` — generated config không có whitespace lỗi.

#### Manual

- [ ] Mở một YAML THESIS, một YAML M2N2, một YAML CANDI và mỗi traditional
  baseline; xác nhận range, protocol path, variant, `max_online_steps` và
  output path.

### Risks

- **Generated file cũ còn tồn tại:** không dùng glob không lọc variant để
  launch; preflight phải chỉ nhận pattern chính thức.
- **Smoke bị báo như main:** giữ `benchmark_mode` và `max_online_steps` trong
  report, không dùng metric smoke trong bảng performance.

## Phase 2: Load RedLamp encoder cho M2N2 và CANDI

### Goal

M2N2 và CANDI dùng đúng encoder CNN pretrained của RedLamp, latent `128`, và
không train một encoder mới trong `fit()` khi checkpoint được cung cấp.

### Stages

#### Stage 2.1: Khóa payload mapping và shape contract

- **Kết quả:** Runtime biết chính xác state-dict nào là encoder và xác nhận
  kiến trúc `38 -> 64 -> 64 -> 128` trước khi load.
- **Điều kiện hoàn tất:** Key thiếu, shape sai hoặc payload không đọc được đều
  tạo lỗi rõ ràng; không dùng `strict=False` để che mismatch.

#### Stage 2.2: Load encoder qua adaptive baseline path

- **Kết quả:** M2N2 và CANDI nhận checkpoint path từ config, khởi tạo encoder
  từ RedLamp và truyền latent `128` vào score path hiện tại.
- **Điều kiện hoàn tất:** Hai baseline dùng cùng load contract nhưng vẫn giữ
  policy riêng của từng method.

#### Stage 2.3: Giữ lifecycle của baseline-owned components

- **Kết quả:** Runtime chỉ lấy tensor encoder; RedLamp classification head và
  reconstruction head không đi vào M2N2/CANDI.
- **Điều kiện hoàn tất:** Decoder hoặc score head của baseline được xác định
  rõ là khởi tạo, giữ nguyên hay cập nhật; không tự đổi objective để xử lý
  latent mismatch.

#### Stage 2.4: Chặn fresh-encoder training

- **Kết quả:** Khi checkpoint hợp lệ tồn tại, `fit()` không train encoder mới
  và không âm thầm fallback sang random initialization.
- **Điều kiện hoàn tất:** Test lifecycle chứng minh checkpoint path không gọi
  training loop của fresh backbone.

### Changes

#### 1. Checkpoint loader

- **File:** `src/baselines/online/adaptive.py` hoặc helper đề xuất
  `src/baselines/online/redlamp_encoder_checkpoint.py`.
- **Symbol:** đường khởi tạo backbone trong `fit()` và `_fit_backbone()`.
- **Change:** Thêm một đường initialize/load có thứ tự rõ ràng:

  ```text
  validate input_dim and CNN config
    -> load checkpoint payload
    -> locate encoder state-dict from verified key mapping
    -> validate shape 38 -> 64 -> 64 -> 128
    -> load only encoder tensors
    -> keep baseline-owned score/adaptation components
  ```

  Khi checkpoint không tồn tại, key không đúng hoặc shape không khớp, raise
  lỗi rõ ràng. Không fallback sang train encoder random trong main run.
- **Reason:** Đây là mismatch lớn nhất giữa contract và runtime hiện tại.
- **Dependencies:** Payload thật từ Phase 0.

#### 2. Public constructors

- **Files:** `src/baselines/online/m2n2.py`,
  `src/baselines/online/candi.py` và `src/baselines/online/adaptive.py`.
- **Symbols:** `__init__()` của ba class.
- **Change:** Nhận `pretrained_encoder_checkpoint`; lưu path đã resolve; đặt
  default latent `128` cho M2N2/CANDI; truyền các tham số CNN đã chốt.
- **Reason:** Config phải đi đến runtime mà không qua hidden default.

#### 3. Tách encoder pretrained khỏi RedLamp heads

- **Files:** model object được dùng trong `_score_backbone_windows()` và
  `src/models/baseline_impl/redlamp_baseline.py` nếu cần adapter.
- **Symbols:** encoder state load và score call.
- **Change:** Chỉ lấy tensor của encoder. Không load RedLamp classification
  head, reconstruction head hoặc state không thuộc backbone. Giữ score/adaptation
  interface hiện tại của M2N2/CANDI; không tự tạo objective mới.
- **Reason:** Contract cho phép dùng encoder RedLamp nhưng không cho dùng nhầm
  các head của RedLamp.
- **Dependencies:** Cần ghi rõ baseline-owned component nào được khởi tạo và
  kiểm tra nó trong Phase 0/Phase 2. Nếu checkpoint thật cho thấy score head
  không tương thích, dừng trước khi chọn cách fallback.

#### 4. No-training guard

- **File:** `src/baselines/online/adaptive.py`.
- **Symbol:** `fit()` và `_fit_backbone()`.
- **Change:** Khi checkpoint có mặt, không tạo optimizer để train encoder và
  không gọi training loop hiện tại. Các field `backbone_epochs`,
  `backbone_batch_size`, `backbone_learning_rate` không điều khiển main path.
- **Reason:** Tránh silent fallback và giữ đúng quyết định không train lại.

### Verification

#### Automated

- [ ] Test fixture tạo checkpoint RedLamp-compatible trong `tmp_path`, load vào
  M2N2 và CANDI, rồi so sánh encoder tensors trước/sau load.
- [ ] Test monkeypatch `_fit_backbone()` hoặc optimizer path để chứng minh
  checkpoint path không chạy fresh-backbone training.
- [ ] Test checkpoint thiếu key, sai shape và thiếu file đều fail với lỗi rõ
  ràng.
- [ ] Test metadata có path, role `pretrained_encoder`, latent `128` và
  `encoder_family: cnn_simple`.

#### Manual

- [ ] Với một checkpoint thật, kiểm tra không có classification/reconstruction
  head RedLamp trong state được dùng cho M2N2/CANDI.

### Risks

- **Decoder/score head chưa có contract đầy đủ:** không được tự load RedLamp
  decoder. Nếu score hiện tại cần decoder, phải xác định baseline-owned decoder
  và lifecycle của nó trước khi chạy performance; ghi quyết định đó vào test
  và provenance.
- **Latent mismatch:** giữ `128` cho baseline và `64` cho THESIS; không thêm
  projection chỉ để làm hai số bằng nhau.

## Phase 3: Hoàn thiện provenance và report

### Goal

Mỗi kết quả main cho phép truy ngược method, stream, protocol và đúng checkpoint
đã load.

### Stages

#### Stage 3.1: Chuẩn hóa method metadata

- **Kết quả:** Mỗi method công bố variant, hyperparameter, encoder family,
  latent dimension và vai trò checkpoint trong cùng một metadata contract.
- **Điều kiện hoàn tất:** Traditional ML ghi rõ checkpoint là
  `not_applicable`, thay vì để thiếu field hoặc tạo checkpoint giả.

#### Stage 3.2: Gắn identity và SHA-256 cho artifact

- **Kết quả:** Deep baseline report và threshold artifact trỏ tới đúng file
  checkpoint đã load cùng SHA-256; traditional ML không bị hash giả.
- **Điều kiện hoàn tất:** Hash trong report khớp file được kiểm tra tại runtime.

#### Stage 3.3: Đưa metadata vào report cuối

- **Kết quả:** Report ghi method metadata, absolute range, protocol,
  `max_online_steps`, status và artifact paths cho cả main và smoke.
- **Điều kiện hoàn tất:** Người đọc có thể phân biệt kết quả main với smoke và
  truy ngược được checkpoint mà không cần đọc log runtime.

#### Stage 3.4: Giữ tương thích report cũ

- **Kết quả:** Field mới được thêm tại report boundary mà không đổi tên âm
  thầm các field mà summarizer hiện tại đang dùng.
- **Điều kiện hoàn tất:** Report parser/summarizer hiện có tiếp tục đọc được
  report sau migration.

### Changes

#### 1. Baseline method metadata

- **Files:** `src/baselines/online/adaptive.py`,
  `src/baselines/online/m2n2.py`, `src/baselines/online/candi.py` và
  `src/baselines/online/frozen.py`.
- **Symbols:** `_backbone_metadata()`, `_method_metadata()`.
- **Change:** Ghi đủ hyperparameter đã dùng, variant, encoder family, latent,
  checkpoint path, checkpoint role, checkpoint source và SHA-256. Traditional
  ML ghi `checkpoint_role: not_applicable`.
- **Reason:** Contract yêu cầu audit được nguồn tham số và protocol nội tại.

#### 2. SHA-256 và threshold artifact

- **Files:** `scripts/benchmarks/run_online_streaming_benchmark.py`,
  `src/protocols/threshold_artifact.py` và
  `src/core/artifact_integrity.py`.
- **Symbols:** baseline calibration/report path và `build_threshold_artifact()`.
- **Change:** Dùng primitive `sha256_file` hiện có để hash đúng checkpoint đã
  load. Truyền hash deep baseline vào artifact/provenance; không tạo hash giả
  cho traditional ML. Không đổi baseline thành THESIS V4 nếu contract không
  yêu cầu.
- **Reason:** Hash phải gắn với file thật, không gắn với metric hoặc tên file.

#### 3. Report wiring

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py`.
- **Symbol:** `run_online_streaming_benchmark()` và
  `online_execution` report.
- **Change:** Đưa `calibration["method_metadata"]` vào report cùng stream
  selections, protocol, max steps, status và artifact paths. Main report phải
  cho thấy đúng range request; smoke phải cho thấy giới hạn smoke.
- **Reason:** Metadata hiện được tạo trong calibration nhưng runner chưa ghi
  vào report.

### Verification

#### Automated

- [ ] Wrapper test kiểm tra report có `method_metadata`, stream range, protocol,
  checkpoint role/path/hash và `online_variant` đúng.
- [ ] Artifact test kiểm tra hash của file checkpoint khớp hash trong result.
- [ ] Traditional report test kiểm tra `checkpoint_role` là
  `not_applicable`.

#### Manual

- [ ] Đọc một report của M2N2/CANDI và một report traditional; có thể xác định
  ngay method, entity, seed, range, threshold source và checkpoint status.

### Risks

- **Report schema cũ:** giữ các field cũ đang được summarizer dùng; thêm field
  mới ở boundary report thay vì đổi tên âm thầm.
- **Smoke range bị thay đổi do max steps:** ghi rõ `benchmark_mode=smoke` và
  `max_online_steps`; không dùng smoke report cho main table.

## Phase 4: Khóa runtime flow và test contract

### Goal

Test fail ngay khi code quay lại full stream, latent cũ, variant cũ hoặc fresh
encoder training.

### Stages

#### Stage 4.1: Khóa generator và range contract bằng test

- **Kết quả:** Test kiểm tra range theo entity, baseline variant `main`, latent
  `128`, checkpoint path và số lượng matrix.
- **Điều kiện hoàn tất:** Generator regression làm test fail trước khi tạo
  config để chạy benchmark.

#### Stage 4.2: Khóa runtime lifecycle và global indices

- **Kết quả:** Test chứng minh calibration dùng clean validation, stream giữ
  global indices và M2N2/CANDI load checkpoint mà không train fresh encoder.
- **Điều kiện hoàn tất:** Test bao phủ cả lỗi checkpoint và policy update riêng
  của hai deep baseline.

#### Stage 4.3: Khóa preflight và chạy focused suite

- **Kết quả:** Preflight nhận đúng 54 THESIS và 45 baseline; focused online
  suite pass trên code/config mới.
- **Điều kiện hoàn tất:** Không còn test nào bảo vệ count, variant hoặc latent
  theo contract cũ.

### Changes

#### 1. Config-generation tests

- **Files:** `tests/online/test_online_benchmark_config_generation.py` và
  `tests/online/test_online_streaming_benchmark_config_generation.py`.
- **Change:** THESIS test kiểm tra ba range theo entity cho main/smoke. Baseline
  test kiểm tra CANDI/M2N2 chỉ có `main`, latent `128`, checkpoint path đúng
  entity/seed và traditional chỉ có `main`.
- **Reason:** Generator là nguồn tạo matrix; test phải kiểm tra contract chứ
  không kiểm tra tên file cũ.

#### 2. Runtime range and global indices

- **Files:** `tests/online/test_online_stream_range.py`,
  `tests/online/test_online_streaming_baseline_contracts.py` và một test THESIS
  phù hợp trong `tests/online/`.
- **Change:** Giữ test selector hiện có; bổ sung test records có global
  `point_index`, `window_start_index`, `window_end_index` sau khi cắt range.
- **Reason:** Range đúng nhưng offset sai vẫn làm hỏng so sánh anomaly span.

#### 3. M2N2/CANDI runtime flow

- **Files:** `tests/online/test_online_streaming_baseline_contracts.py` và test
  mới nếu fixture checkpoint cần tách riêng.
- **Change:** Đổi kỳ vọng latent `64` thành `128`; kiểm tra checkpoint load,
  no-training guard, update policy riêng và clean-validation calibration.
- **Reason:** Test hiện tại đang bảo vệ implementation cũ.

#### 4. Full matrix preflight test

- **File:** `tests/benchmarks/test_full_benchmark_matrix_preflight.py`.
- **Change:** Đổi expected online baseline count từ `81` thành `45`; thêm
  assertion rằng preflight chọn `online_main` cho M2N2/CANDI/traditional.
- **Reason:** Preflight là gate trước main run.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_streaming_baseline_contracts.py tests/online/test_online_stream_range.py tests/online/test_online_streaming_benchmark_wrapper.py tests/online/test_online_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py tests/benchmarks/test_full_benchmark_matrix_preflight.py` — tất cả test contract pass.
- [ ] Chạy full online test subset sau khi các test focused pass để phát hiện
  regression ở THESIS runtime.

#### Manual

- [ ] Review một record đầu/cuối: chỉ số vẫn là entity-global và không dùng
  test label trong calibration/update decision.

### Risks

- **Test fixture không phản ánh payload thật:** giữ một test với payload key
  mapping đã xác nhận từ Phase 0; fixture tổng hợp chỉ kiểm tra shape và lifecycle.

## Phase 5: Smoke, preflight và bàn giao full matrix

### Goal

Chứng minh một combination chạy end-to-end trên code/config mới trước khi chạy
99 main combinations.

### Stages

#### Stage 5.1: Kiểm tra local config và entry points

- **Kết quả:** Một config THESIS, M2N2, CANDI và traditional được resolve
  thành công với đúng range, protocol, checkpoint và output path.
- **Điều kiện hoàn tất:** Preflight và dry-run pass mà không tạo artifact
  training ngoài contract.

#### Stage 5.2: Chạy hai end-to-end smoke đại diện

- **Kết quả:** Một THESIS smoke và một M2N2 smoke chạy cùng entity, seed và
  absolute range; mỗi run tạo report, metrics, records và provenance.
- **Điều kiện hoàn tất:** Cả hai report cho thấy đúng main/smoke status,
  checkpoint identity và global index.

#### Stage 5.3: Review smoke và quyết định release gate

- **Kết quả:** Các lỗi về runtime flow, checkpoint hoặc artifact được xử lý
  trước khi mở rộng phạm vi.
- **Điều kiện hoàn tất:** Preflight báo `54 + 45`, smoke pass và không còn
  mismatch giữa revision local với revision được phép chạy.

#### Stage 5.4: Bàn giao full matrix

- **Kết quả:** Codebase và launch manifest sẵn sàng cho 99 main combinations;
  traditional ML chạy frozen theo cùng contract.
- **Điều kiện hoàn tất:** Chỉ launch sau khi tất cả gate tự động và manual của
  các phase trước đều pass.

### Changes

#### 1. Local dry-run và preflight

- **Files:** `scripts/benchmarks/run_online_streaming_benchmark.py`,
  `scripts/benchmarks/run_thesis_online_benchmark.py` và preflight script.
- **Change:** Chạy dry-run cho một THESIS, một M2N2, một CANDI và nếu thời gian
  cho phép một traditional method; xác nhận resolved config, checkpoint path,
  range và output directory.
- **Reason:** Mỗi entry point có flow riêng; baseline smoke không chứng minh
  THESIS smoke.

#### 2. One concrete end-to-end smoke

- **Selection:** THESIS `O1/A2/machine_1_6/seed6` theo checklist hiện hành,
  sau đó một baseline `M2N2/main/machine_1_6/seed6` dùng cùng range.
- **Change:** Dùng đúng protocol `smd_window20_cleanval_q99_ewma09.yaml`,
  range contract `[146,2200)` cho benchmark smoke, `max_online_steps` chỉ để
  giới hạn smoke nếu cần. Không dùng metric smoke cho performance table.
- **Reason:** Đây là kiểm tra end-to-end tối thiểu trước full matrix.
- **Dependencies:** Phases 0-4 pass; RedLamp checkpoint có thể đọc; Stage B
  và threshold artifact THESIS cùng identity.

#### 3. Full matrix gate

- **Change:** Chỉ phát lệnh full matrix sau khi preflight báo
  `54 THESIS + 45 baseline` và smoke report có provenance đầy đủ. Traditional
  ML vẫn chạy cùng range và protocol nhưng frozen.
- **Reason:** Tránh tốn GPU/time cho matrix sai config hoặc dùng encoder random.

### Verification

#### Automated

- [ ] `.venv/bin/python -m scripts.ops.preflight_full_benchmark_matrix --json` — report ready, THESIS 54, baseline 45.
- [ ] Dry-run wrapper cho config smoke pass.
- [ ] One end-to-end smoke tạo report, metrics, records và provenance; không
  tạo encoder training artifact ngoài contract.
- [ ] `git diff --check` và full focused pytest pass trước launch.

#### Manual

- [ ] Kiểm tra report smoke có đúng absolute range, latent, checkpoint source,
  SHA-256 và `benchmark_mode=smoke`.
- [ ] Trước remote main run, kiểm tra revision, GPU, disk, active jobs và
  checkpoint paths read-only. Chỉ sau đó mới chạy đúng command đã duyệt.

### Risks

- **Smoke pass nhưng không đi qua mọi policy branch:** ghi branch coverage
  thực tế; không tuyên bố mọi adaptation branch đã được kiểm tra nếu stream
  không chạm branch đó.
- **Remote source khác local revision:** dừng, không upload/reset/ghi đè remote
  ngoài phạm vi được duyệt.
- **Disk tăng do generated/output artifacts:** giữ report, metrics, records,
  checkpoint cần thiết và provenance; không giữ raw forward-pass output mặc
  định.

## Testing strategy

Testing chia thành bốn lớp:

1. **Pure contract:** range nửa kín, validation, shape, key mapping và SHA-256.
2. **Generator:** file count, variant, range, protocol và checkpoint path.
3. **Runtime flow:** calibration clean validation, global index, baseline policy,
   no fresh encoder training và report metadata.
4. **Integration:** preflight, dry-run và một end-to-end smoke cho mỗi entry
   point chính.

Test không dùng anomaly labels để chọn threshold. Test fixture chỉ dùng dữ
liệu nhỏ và checkpoint tạm; performance không được suy ra từ fixture.

## Migration and rollback

Generated YAML là artifact có thể tái tạo từ generator. Migration thực hiện
theo thứ tự:

1. sửa generator và test;
2. chạy generator;
3. kiểm tra exact path/count/diff;
4. cập nhật preflight và launch allow-list;
5. chỉ sau đó mới quyết định xóa stale YAML cụ thể.

Rollback code dùng revision trước khi sửa. Rollback config chỉ dùng các file
đã generate từ cùng revision với code. Không trộn code loader mới với config
M2N2/CANDI latent `64` hoặc checkpoint path thiếu.

Nếu RedLamp checkpoint không đọc được, rollback an toàn là không chạy main
baseline; không fallback sang fresh encoder vì kết quả sẽ vi phạm contract.

## Documentation

- Cập nhật `documents/inventories/online-benchmark-combinations-and-smoke-checklist.md`
  theo matrix 99 và một baseline config `main`.
- Giữ `documents/spec/online_benchmark_contract.md` là nguồn chuẩn.
- Ghi trạng thái triển khai và các giới hạn checkpoint payload vào research
  report hoặc log implementation tương ứng.
- Không sửa lịch sử `documents/spec/full-spec-v2.md` để che khác biệt matrix;
  nếu cần, trỏ người đọc về contract hiện hành.

## Final verification

- [ ] THESIS generated config chứa range đúng cho cả ba entity.
- [ ] M2N2/CANDI mỗi method có đúng 9 main config và dùng RedLamp path cùng
  entity/seed.
- [ ] Traditional ML mỗi method có đúng 9 main config, variant `main`, frozen.
- [ ] Preflight báo `54` THESIS và `45` baseline online configs.
- [ ] M2N2/CANDI load encoder latent `128`, không load RedLamp heads và không
  train fresh encoder khi checkpoint tồn tại.
- [ ] Main report ghi entity, seed, range, protocol, method metadata,
  checkpoint role/path/SHA-256 và smoke/main status.
- [ ] Một THESIS smoke và một baseline smoke chạy end-to-end trước full matrix.
- [ ] Full matrix chỉ được launch sau khi tất cả gate trên pass.

## Assumptions and non-blocking uncertainties

- **Checkpoint key names:** chưa có RedLamp `best.pt` trong local tree. Phase 0
  phải xác nhận bằng payload thật; đây là input kỹ thuật bắt buộc, không phải
  lý do để đoán.
- **Baseline-owned decoder/score head:** contract chỉ chốt load RedLamp
  encoder và bỏ RedLamp heads. Trước implementation cần xác định lifecycle
  của component score hiện tại; không tự load RedLamp decoder và không tự đổi
  objective.
- **Stale generated YAML:** plan mặc định không xóa rộng; launch/preflight chỉ
  nhận danh sách `main` chính thức. Việc xóa exact stale paths là cleanup tùy
  chọn sau khi review manifest.
- **Historical spec counts:** `full-spec-v2.md` có semantics matrix cũ. Contract
  mới và checklist hiện hành là nguồn dùng cho benchmark này; mapping ở trên
  ghi rõ đây không phải rename runtime object.

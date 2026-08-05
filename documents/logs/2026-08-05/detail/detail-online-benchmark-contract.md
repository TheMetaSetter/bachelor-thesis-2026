---
date: 2026-08-05 17:57:42 +07:00
topic: "Chi tiết triển khai online benchmark contract"
status: ready
revision: 5f3ac2654c3a4d9b5a0d25cc5b08672c9b0fe70c
source_structure: documents/logs/2026-08-05/plan/plan-online-benchmark-contract.md
related_documents:
  - documents/spec/online_benchmark_contract.md
  - documents/logs/2026-08-05/research/research-online-benchmark-contract-change-surface.md
  - prompts/4_detail_prompt.md
---

# Detailed Implementation: Điều chỉnh codebase theo online benchmark contract

## Summary

Tài liệu này chia 6 phase của plan thành 22 stage. Mỗi stage có các atomic
step theo đúng thứ tự thực hiện. Mỗi step nêu file, symbol, input, output, lỗi,
khả năng tương thích và cách kiểm tra.

Tài liệu này chỉ mô tả thay đổi. Chưa sửa source code, config, test hoặc
generated YAML.

## Source structure

Đọc plan tại
`documents/logs/2026-08-05/plan/plan-online-benchmark-contract.md`. Plan có
6 phase:

| Phase | Kết quả chính | Số stage |
| --- | --- | ---: |
| 0 | Xác nhận checkpoint và matrix đầu vào | 3 |
| 1 | Chuẩn hóa range và generated matrix | 4 |
| 2 | Load RedLamp encoder cho M2N2/CANDI | 4 |
| 3 | Ghi provenance và report | 4 |
| 4 | Khóa behavior bằng test | 3 |
| 5 | Smoke, preflight và gate full matrix | 4 |

Các stage trong cùng một phase chạy theo thứ tự số. Không chạy phase sau khi
phase trước chưa đạt điều kiện hoàn tất.

## Current state

- `src/protocols/online_stream_range.py:8-43` đã cắt stream theo interval nửa
  kín `[start, end)`, nhưng generator THESIS chưa ghi range vào config.
- `scripts/benchmarks/generate_online_benchmark_configs.py:108-137` chưa ghi
  `absolute_start_index` và `absolute_end_index`.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:28-34`
  đang sinh `A0/A1/A2` cho M2N2 và CANDI.
- `src/baselines/online/adaptive.py:119-176` luôn tạo và train một
  `SimpleWindowCnnAutoencoder` mới.
- `src/baselines/online/m2n2.py:15-54` và
  `src/baselines/online/candi.py:21-60` chưa nhận RedLamp checkpoint.
- `src/baselines/online/adaptive.py:268-288` đã có method metadata, nhưng
  metadata hiện chưa có checkpoint path, role và SHA-256.
- `scripts/benchmarks/run_online_streaming_benchmark.py:309-412` nhận
  `method_metadata` trong calibration nhưng chưa ghi field này vào report.
- `scripts/ops/preflight_full_benchmark_matrix.py:134-148` đang đếm 81 online
  baseline configs thay vì 45.
- RedLamp dùng `SimpleWindowCnnEncoder` tại
  `src/models/baseline_impl/redlamp_baseline.py:162-170`. Config RedLamp chốt
  input 38, latent 128, 3 lớp CNN, kernel 3, hidden 64 và dropout 0.1 tại
  `configs/model/redlamp_baseline_comparative_smd.yaml:1-10`.

## Desired end state

1. Generator ghi đúng absolute range cho cả THESIS và baseline.
2. THESIS có 54 main configs. M2N2, CANDI và ba traditional ML baselines có
   tổng cộng 45 main configs.
3. M2N2 và CANDI cùng đọc RedLamp `best.pt` theo entity và seed.
4. Runtime chỉ load tensor của RedLamp encoder. Runtime không load RedLamp
   classification head hoặc reconstruction head.
5. Runtime không train fresh encoder khi main config có checkpoint.
6. Report ghi stream, protocol, method metadata, checkpoint role, checkpoint
   path, SHA-256 và trạng thái main/smoke.
7. Focused tests, preflight và một THESIS smoke cùng một baseline smoke pass
   trước full matrix.

## Scope

### In scope

- Generator, runtime baseline, provenance, preflight, tests và checklist hiện
  hành.
- RedLamp encoder loader cho M2N2 và CANDI.
- Absolute ranges `[146,2200)`, `[2634,6116)` và `[1099,10807)`.
- Baseline variant `main`.

### Out of scope

- Train lại RedLamp encoder.
- Ép latent 128 của baseline thành latent 64 của THESIS.
- Đổi policy update của M2N2 hoặc CANDI.
- Load RedLamp decoder hoặc classification head.
- Đổi lịch sử `documents/spec/full-spec-v2.md`.
- Xóa rộng generated YAML hoặc output artifact.

## Evidence

- `documents/spec/online_benchmark_contract.md:132-157` — quy định encoder,
  latent dimension và kiến trúc CNN.
- `documents/spec/online_benchmark_contract.md:191-241` — quy định RedLamp
  checkpoint, path, epoch metadata và no-fallback rule.
- `documents/spec/online_benchmark_contract.md:248-255` — quy định score,
  policy update và hyperparameter của các method.
- `documents/spec/online_benchmark_contract.md:345-353` — quy định provenance.
- `src/protocols/online_stream_range.py:8-43` — runtime range selector hiện có.
- `src/core/config_model_validation.py:214-236,504-523` — schema đã cho phép
  hai trường absolute range và kiểm tra chúng xuất hiện cùng nhau.
- `scripts/benchmarks/generate_online_benchmark_configs.py:108-137` — THESIS
  generator còn thiếu range.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:22-194`
  — baseline range đã có, nhưng neural variant và encoder config còn cũ.
- `src/baselines/online/adaptive.py:56-176` — constructor và fresh-backbone
  training path hiện tại.
- `src/baselines/online/adaptive.py:178-218,375-466` — score path và online
  update path hiện tại.
- `src/baselines/online/m2n2.py:12-76` — M2N2 policy và metadata hiện tại.
- `src/baselines/online/candi.py:18-80` — CANDI policy và metadata hiện tại.
- `src/baselines/online/frozen.py:59-443` — traditional ML frozen runtime.
- `scripts/benchmarks/run_online_streaming_benchmark.py:239-413` — baseline
  entry point, calibration, range selection và report.
- `src/core/artifact_integrity.py:11-20` — hàm `sha256_file` hiện có.
- `src/protocols/threshold_artifact.py:237-373` — threshold artifact hỗ trợ
  checkpoint SHA-256.

## Atomic step conventions

Trong mỗi stage:

- `File / Symbol` chỉ vị trí mà implementer cần mở.
- `Input` ghi dữ liệu hoặc config mà step nhận.
- `Action` ghi hành động cụ thể.
- `Output` ghi trạng thái observable sau step.
- `Error / compatibility` ghi điều kiện dừng và behavior phải giữ.
- `Check` ghi kiểm tra ngay sau step.

## Phase 0: Chốt input checkpoint và matrix trước khi sửa code

### Goal

Phase này tạo bằng chứng về 9 RedLamp checkpoints và generated-config tree.
Các phase sau không được đoán payload hoặc path.

### Dependencies

- Không cần source change từ phase trước.
- Cần quyền đọc remote hoặc một bản checkpoint thật trong `outputs/`.

### Detailed steps

#### Stage 0.1: Xác nhận RedLamp checkpoint inventory

1. **Liệt kê 9 checkpoint canonical.**
   - **File / Symbol:** remote path
     `outputs/benchmark/smd/redlamp_baseline/<entity>/seed<seed>/checkpoints/best.pt`.
   - **Input:** 3 entity `machine_1_6`, `machine_3_4`, `machine_3_9` và 3 seed
     `6`, `8`, `36`.
   - **Action:** Đọc từng path bằng remote read-only workflow. Không tạo, sửa
     hoặc xóa file trên remote.
   - **Output:** Danh sách path tồn tại và path bị thiếu.
   - **Error / compatibility:** Nếu path thiếu, đánh dấu `missing` và dừng
     việc chuẩn bị full matrix. Không thay bằng random encoder.
   - **Check:** Đối chiếu danh sách với 9 entity-seed cells của contract.

2. **Đọc payload của một checkpoint thật.**
   - **File / Symbol:** payload của `best.pt`; chưa chọn tên key trước khi đọc.
   - **Input:** Một path tồn tại từ bước 1.
   - **Action:** Đọc top-level keys, nested mapping, state-dict keys và metadata
     epoch. Không in hoặc lưu tensor đầy đủ.
   - **Output:** Mapping đã xác nhận từ payload tới encoder state-dict.
   - **Error / compatibility:** Nếu payload không phải mapping hoặc không có
     encoder state-dict, ghi lỗi chính xác và dừng Phase 2 design.
   - **Check:** Xác nhận có thể phân biệt encoder với decoder và
     classification head.

3. **Kiểm tra shape của encoder.**
   - **File / Symbol:** các Conv1d trong RedLamp encoder.
   - **Input:** State-dict mapping đã đọc ở bước 2.
   - **Action:** Đối chiếu tensor shape với `input_dim=38`, hidden channels 64,
     3 lớp, kernel 3 và output latent 128.
   - **Output:** Shape contract `38 -> 64 -> 64 -> 128` được xác nhận.
   - **Error / compatibility:** Nếu shape khác contract, không thêm projection
     hoặc đổi latent để che mismatch. Ghi checkpoint vào danh sách không hợp lệ.
   - **Check:** So sánh key và shape của ít nhất một checkpoint khác nếu 9
     checkpoint đều đọc được.

4. **Ghi identity của từng checkpoint.**
   - **File / Symbol:** checkpoint inventory của Phase 0.
   - **Input:** Path, metadata epoch, encoder mapping và file bytes.
   - **Action:** Tính SHA-256 trên đúng file `best.pt`. Ghi entity, seed, epoch
     metadata, path và digest.
   - **Output:** Bảng identity dùng cho config, loader test và provenance.
   - **Error / compatibility:** Nếu file đổi giữa lúc đọc payload và tính hash,
     đọc lại file và ghi nhận mismatch.
   - **Check:** Tính lại hash một lần và xác nhận hai digest giống nhau.

#### Stage 0.2: Chốt generated-config inventory

1. **Liệt kê generated YAML hiện tại.**
   - **File / Symbol:** `configs/experiment/online_benchmark/`.
   - **Input:** Các thư mục `thesis`, `candi`, `m2n2`, `stumpy`, `kmeans_ad`,
     `iforest`.
   - **Action:** Liệt kê file bằng path pattern hiện có. Tách `main`, `smoke` và
     tên baseline `A0/A1/A2`.
   - **Output:** Ba danh sách expected main, expected smoke và stale files.
   - **Error / compatibility:** Không xóa file trong bước inventory.
   - **Check:** So sánh count hiện tại với generator và preflight hiện tại.

2. **Kiểm tra config representative.**
   - **File / Symbol:** một YAML THESIS, CANDI, M2N2 và mỗi traditional method.
   - **Input:** Generated files từ bước 1.
   - **Action:** Đọc `entity_id`, `seed`, variant, range, protocol path, latent
     và output path.
   - **Output:** Danh sách field đang lệch contract.
   - **Error / compatibility:** Nếu YAML không parse được, giữ nguyên file và
     ghi path lỗi.
   - **Check:** Đối chiếu range với ba giá trị trong contract.

3. **Lập stale-file manifest.**
   - **File / Symbol:** generated tree và launch allow-list của preflight.
   - **Input:** Danh sách stale từ bước 1.
   - **Action:** Ghi exact path của baseline `A0/A1/A2` không còn thuộc matrix
     chính thức. Không dùng glob xóa rộng.
   - **Output:** Manifest để review trước cleanup hoặc loại khỏi launch.
   - **Error / compatibility:** Nếu không phân biệt được main và stale path,
     không xóa file và không mở rộng preflight.
   - **Check:** Review tên file với `BENCHMARK_METHOD_VARIANTS` hiện tại.

#### Stage 0.3: Đóng input gate cho implementation

1. **Xác định trạng thái từng checkpoint.**
   - **File / Symbol:** checkpoint inventory và contract matrix.
   - **Input:** Kết quả Stage 0.1.
   - **Action:** Gán trạng thái `verified`, `missing` hoặc `invalid` cho từng
     entity-seed cell.
   - **Output:** Bảng input chính thức cho generator và loader.
   - **Error / compatibility:** Chỉ `verified` mới được dùng cho main run.
   - **Check:** Không còn checkpoint nào ở trạng thái không rõ.

2. **Khóa mapping của loader.**
   - **File / Symbol:** mapping encoder state-dict được xác nhận ở Stage 0.1.
   - **Input:** Top-level keys và tensor shapes thật.
   - **Action:** Ghi mapping vào detail implementation khi implementer bắt đầu
     Phase 2. Không tự đổi mapping trong fixture.
   - **Output:** Phase 2 có input cụ thể để viết loader.
   - **Error / compatibility:** Nếu payload giữa các checkpoint không đồng nhất,
     dừng trước khi tạo helper dùng chung.
   - **Check:** Loader design chỉ tham chiếu key đã đọc từ payload thật.

3. **Chốt launch boundary.**
   - **File / Symbol:** generated-config inventory và preflight allow-list.
   - **Input:** Expected main paths và stale manifest.
   - **Action:** Chọn main paths được phép chạy. Đưa stale paths ra khỏi launch
     selection nhưng chưa xóa.
   - **Output:** Một matrix target rõ ràng cho Phase 1.
   - **Error / compatibility:** Nếu có ít hơn 9 RedLamp path hợp lệ, đánh dấu
     benchmark baseline là blocked.
   - **Check:** Matrix target có 54 THESIS và 45 baseline main cells.

### Tests

- **Checkpoint inventory check**
  - **Location:** remote read-only inspection và implementation log.
  - **Level:** Manual integration check.
  - **Setup:** Có remote host và canonical checkpoint paths.
  - **Action:** Đọc path, payload key, shape và SHA-256.
  - **Expected result:** Có 9 identity records hoặc có danh sách thiếu chính
    xác.

### Verification

#### Automated

- [ ] Check file inventory tách được `main`, `smoke` và stale variants.
- [ ] Check hash đọc lại trả cùng SHA-256 cho mỗi file đã xác nhận.

#### Manual

- [ ] Xác nhận remote revision, host key và path trước khi đọc checkpoint.
- [ ] Xác nhận Phase 0 không ghi hoặc cleanup remote.

### Risks and recovery

- **Risk:** Remote thiếu checkpoint hoặc payload khác nhau.
  **Mitigation:** Dừng main baseline và ghi path lỗi. **Verification:** Đối
  chiếu bảng 9 cells. **Recovery:** Bổ sung đúng checkpoint đã được phê duyệt;
  không fallback sang encoder random.
- **Risk:** Cleanup nhầm generated YAML.
  **Mitigation:** Chỉ lập exact stale manifest. **Verification:** Review diff
  path trước cleanup. **Recovery:** Không xóa ở Phase 0.

### Complete when

- 9 checkpoint paths có trạng thái rõ ràng.
- Encoder key mapping và shape đã được xác nhận từ payload thật.
- Main allow-list và stale manifest đã tách riêng.

## Phase 1: Chuẩn hóa absolute range và generated matrix

### Goal

Phase này tạo config đúng contract trước khi runtime load model.

### Dependencies

- Stage 0.3 đã xác nhận checkpoint paths và matrix target.
- Giữ nguyên `select_online_stream_sequence` vì runtime selector đã có test.

### Detailed steps

#### Stage 1.1: Gắn absolute range vào mọi generator

1. **Thêm range cho THESIS generator.**
   - **File / Symbol:** `scripts/benchmarks/generate_online_benchmark_configs.py`,
     `_task_overrides()`.
   - **Input:** `entity_id` và contract range.
   - **Action:** Ghi `absolute_start_index` và `absolute_end_index` cho ba entity:
     `machine-1-6=(146,2200)`, `machine-3-4=(2634,6116)` và
     `machine-3-9=(1099,10807)`.
   - **Output:** Main và smoke THESIS config có cùng range request.
   - **Error / compatibility:** Unknown entity phải fail rõ ràng. Không đổi
     selector hoặc đổi index sang zero-based khác.
   - **Check:** Build một config cho mỗi entity và đọc hai field sau merge.

2. **Giữ smoke limit riêng với range.**
   - **File / Symbol:** `_task_overrides()` và `build_online_benchmark_config()`.
   - **Input:** `smoke` boolean.
   - **Action:** Giữ `max_online_steps=None` cho main và giữ giới hạn hiện có
     cho smoke. Không thay absolute end index bằng smoke end index.
   - **Output:** Smoke bắt đầu tại đúng absolute start nhưng có thể ngắn hơn.
   - **Error / compatibility:** Không ghi metric smoke vào main table.
   - **Check:** Assert main có `None`, smoke có giá trị dương.

3. **Kiểm tra validation schema.**
   - **File / Symbol:** `src/core/config_model_validation.py`, các check tại
     `online_adaptation`.
   - **Input:** Config sau generator.
   - **Action:** Chạy `validate_experiment_config()` trên config THESIS mới.
   - **Output:** Hai field được chấp nhận và xuất hiện cùng nhau.
   - **Error / compatibility:** Nếu validation fail vì field không nằm trong
     schema, sửa schema hiện có trước khi sửa runtime.
   - **Check:** Test cả trường hợp thiếu một trong hai field để giữ lỗi hiện có.

#### Stage 1.2: Chuẩn hóa baseline matrix và checkpoint reference

1. **Chỉ sinh variant `main` cho M2N2 và CANDI.**
   - **File / Symbol:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`,
     `BENCHMARK_METHOD_VARIANTS`.
   - **Input:** Method names `candi`, `m2n2`.
   - **Action:** Đổi mỗi tuple neural thành `("main",)`.
   - **Output:** Mỗi neural method có 9 main và 9 smoke configs.
   - **Error / compatibility:** Không map `main` thành THESIS `A0`. THESIS
     variants vẫn giữ nguyên ở generator riêng.
   - **Check:** Count generator trả 90 file gồm 45 main và 45 smoke.

2. **Đưa entity vào baseline kwargs builder.**
   - **File / Symbol:** `_baseline_kwargs()` và
     `build_online_streaming_benchmark_config()`.
   - **Input:** `method`, `entity_id`, `seed`, `smoke`.
   - **Action:** Truyền `entity_id` vào `_baseline_kwargs()` để builder tạo đúng
     path RedLamp theo entity và seed.
   - **Output:** Baseline kwargs có checkpoint path không phụ thuộc hidden
     default.
   - **Error / compatibility:** Unknown entity phải fail trước khi ghi YAML.
   - **Check:** Đọc path của cả ba entity và ba seed.

3. **Ghi RedLamp checkpoint reference.**
   - **File / Symbol:** `_baseline_kwargs()`.
   - **Input:** Verified checkpoint inventory từ Phase 0.
   - **Action:** Ghi
     `outputs/benchmark/smd/redlamp_baseline/<entity>/seed<seed>/checkpoints/best.pt`
     vào `pretrained_encoder_checkpoint` cho M2N2 và CANDI.
   - **Output:** Hai method cùng trỏ tới cùng checkpoint entity-seed.
   - **Error / compatibility:** Không tạo path theo method hoặc theo
     `O0/O1`. Không sinh YAML nếu path mapping không tồn tại trong inventory.
   - **Check:** So sánh CANDI và M2N2 path của cùng entity-seed.

4. **Đặt latent và CNN contract cho neural baseline.**
   - **File / Symbol:** `_baseline_kwargs()`.
   - **Input:** RedLamp architecture contract.
   - **Action:** Đặt `input_dim=38`, `encoder_dim=128`, `encoder_family=cnn_simple`,
     `cnn_num_layers=3`, `cnn_kernel_size=3`, `cnn_hidden_channels=64` và
     `cnn_dropout=0.1`.
   - **Output:** YAML biểu diễn rõ kiến trúc encoder.
   - **Error / compatibility:** Không dùng `encoder_dim=64` cho M2N2/CANDI.
   - **Check:** Assert toàn bộ field trong generated config.

5. **Loại training fields khỏi main baseline config.**
   - **File / Symbol:** `_baseline_kwargs()`.
   - **Input:** Các field `backbone_epochs`, `backbone_batch_size` và
     `backbone_learning_rate` hiện có.
   - **Action:** Không ghi các field này vào config main vì main dùng checkpoint.
     Nếu smoke cần field để tương thích parser, runtime phải bỏ qua chúng khi
     checkpoint có mặt.
   - **Output:** Config không bật fresh-backbone training.
   - **Error / compatibility:** Không xóa field khỏi parser nếu caller khác còn
     dùng; chỉ ngăn main path đọc chúng.
   - **Check:** Tìm các field trong neural generated YAML và preflight.

#### Stage 1.3: Đồng bộ runtime defaults và preflight

1. **Đổi default variant của runner.**
   - **File / Symbol:** `scripts/benchmarks/run_online_streaming_benchmark.py`,
     `run_online_streaming_benchmark()`.
   - **Input:** Config có hoặc không có `online_variant`.
   - **Action:** Đổi fallback từ `A0` thành `main`.
   - **Output:** Baseline config thiếu field không tự nhận THESIS label.
   - **Error / compatibility:** Config explicit variant vẫn được giữ để test
     legacy; preflight main không nhận legacy variant.
   - **Check:** Wrapper test đọc report với config explicit `main`.

2. **Đổi default của adaptive baseline constructors.**
   - **File / Symbol:** `AdaptiveStreamingBaselineBase.__init__()`,
     `M2N2StreamingBaseline.__init__()` và `CANDIStreamingBaseline.__init__()`.
   - **Input:** Constructor kwargs từ generated config.
   - **Action:** Đặt `online_variant="main"` và latent default `128` cho neural
     baseline.
   - **Output:** Constructor không còn mặc định A0 hoặc latent 64.
   - **Error / compatibility:** Cập nhật mọi direct caller trong test để truyền
     fixture checkpoint hợp lệ.
   - **Check:** Test metadata trả variant `main` và encoder dim `128`.

3. **Đổi default của frozen baseline constructors.**
   - **File / Symbol:** `_FrozenStreamingBaseline` và ba subclass trong
     `src/baselines/online/frozen.py`.
   - **Input:** Traditional config.
   - **Action:** Đặt `online_variant="main"`.
   - **Output:** Traditional report dùng một label canonical.
   - **Error / compatibility:** Không thêm neural checkpoint field cho
     traditional ML.
   - **Check:** Direct constructor test và generated config test đều trả main.

4. **Cập nhật preflight baseline count.**
   - **File / Symbol:** `build_preflight_report()` và vòng lặp baseline tại
     `scripts/ops/preflight_full_benchmark_matrix.py:134-148`.
   - **Input:** Main YAML tree.
   - **Action:** Đếm CANDI/M2N2 như một variant `main`; kiểm tra mỗi method có 9
     file và `online_variant == "main"`.
   - **Output:** Report trả `baselines: 45`.
   - **Error / compatibility:** Nếu còn A0/A1/A2 trong allow-list, preflight
     fail thay vì đếm chúng vào matrix.
   - **Check:** Test report expected `{"thesis": 54, "baselines": 45}`.

#### Stage 1.4: Tái tạo và kiểm tra config tree

1. **Chạy THESIS generator.**
   - **File / Symbol:** `scripts/benchmarks/generate_online_benchmark_configs.py`,
     `generate_thesis_online_benchmark_configs()`.
   - **Input:** Generator đã có range.
   - **Action:** Chạy module generator với `--print-count` và sau đó generate
     YAML.
   - **Output:** 108 file THESIS gồm 54 main và 54 smoke.
   - **Error / compatibility:** Nếu count khác 108, không chuyển Phase 2.
   - **Check:** Đọc một YAML cho mỗi entity và kiểm tra range.

2. **Chạy baseline generator.**
   - **File / Symbol:** `scripts/benchmarks/generate_online_streaming_benchmark_configs.py`,
     `generate_online_streaming_benchmark_configs()`.
   - **Input:** Method variants và checkpoint paths mới.
   - **Action:** Chạy module generator với `--print-count` và generate YAML.
   - **Output:** 90 file baseline gồm 45 main và 45 smoke.
   - **Error / compatibility:** Nếu file stale còn tồn tại, giữ chúng ngoài
     launch allow-list; không xóa bằng glob.
   - **Check:** Count từng method là 18 file gồm main và smoke.

3. **Kiểm tra YAML bằng parser hiện có.**
   - **File / Symbol:** `src/core/config.py`, `load_experiment_config()` và
     YAML loader của baseline runner.
   - **Input:** Generated YAML.
   - **Action:** Load từng main config trong allow-list. Kiểm tra protocol,
     range, variant, seed, entity và output path.
   - **Output:** Mọi main config parse được và chứa đủ contract fields.
   - **Error / compatibility:** Config thiếu field hoặc path sai phải fail trước
     runtime.
   - **Check:** Chạy focused generator tests.

4. **Cập nhật inventory hiện hành.**
   - **File / Symbol:** `documents/inventories/online-benchmark-combinations-and-smoke-checklist.md`.
   - **Input:** Count và allow-list mới.
   - **Action:** Ghi 54 THESIS, 45 baseline và tổng 99 comparison cells. Ghi
     M2N2/CANDI/traditional dùng `main`.
   - **Output:** Checklist mô tả đúng matrix đang chạy.
   - **Error / compatibility:** Không sửa semantics lịch sử trong
     `documents/spec/full-spec-v2.md`.
   - **Check:** Đối chiếu checklist với preflight JSON.

5. **Review generated diff.**
   - **File / Symbol:** generated YAML tree.
   - **Input:** Diff sau generator.
   - **Action:** Kiểm tra exact paths, stale paths và output directories. Chỉ
     xóa exact stale file sau khi review riêng.
   - **Output:** Generated tree có launch set xác định.
   - **Error / compatibility:** Nếu diff chứa file ngoài benchmark scope, dừng
     và kiểm tra generator output root.
   - **Check:** Chạy `git diff --check` và kiểm tra `git status`.

### Tests

- **Generator contract test**
  - **Location:** `tests/online/test_online_benchmark_config_generation.py` và
    `tests/online/test_online_streaming_benchmark_config_generation.py`.
  - **Level:** Unit/integration.
  - **Setup:** Generate configs trong test fixture hoặc config tree được project
    dùng hiện tại.
  - **Action:** Đọc range, variant, latent, checkpoint path và count.
  - **Expected result:** THESIS có 54 main; baseline có 45 main; smoke giữ
    `max_online_steps`.
  - **Edge cases:** Unknown entity, thiếu cặp range, stale variant.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py tests/benchmarks/test_full_benchmark_matrix_preflight.py` pass.
- [ ] THESIS generator in `108` file.
- [ ] Baseline generator in `90` file.
- [ ] `git diff --check` pass.

#### Manual

- [ ] Mở một main YAML của từng method group và kiểm tra range, protocol,
  variant, checkpoint path và output path.

### Risks and recovery

- **Risk:** Generator ghi path đúng tên nhưng checkpoint không tồn tại.
  **Mitigation:** Đối chiếu Phase 0 inventory. **Verification:** Preflight đọc
  path. **Recovery:** Dừng main baseline và sửa path generator.
- **Risk:** Stale YAML bị đếm vào matrix. **Mitigation:** Preflight dùng pattern
  main chính thức. **Verification:** Count từng method. **Recovery:** Loại path
  khỏi allow-list; chỉ cleanup exact file sau review.

### Complete when

- Generated config có đúng range và checkpoint reference.
- Main matrix có 54 THESIS và 45 baseline.
- Preflight nhận đúng variant `main` cho baseline.

## Phase 2: Load RedLamp encoder cho M2N2 và CANDI

### Goal

Phase này thay fresh-backbone training bằng việc load encoder RedLamp. Runtime
giữ decoder và policy thuộc baseline hiện tại, nhưng không load RedLamp heads.

### Dependencies

- Stage 0.3 đã xác nhận payload key mapping.
- Stage 1 đã ghi `pretrained_encoder_checkpoint` và latent 128 vào config.

### Detailed steps

#### Stage 2.1: Khóa payload mapping và shape contract

1. **Tạo loader boundary nhỏ.**
   - **File / Symbol:** **Proposed new file:**
     `src/baselines/online/redlamp_encoder_checkpoint.py`.
   - **Input:** Path checkpoint, expected input dim, latent dim và CNN settings.
   - **Action:** Tách logic đọc payload, lấy encoder state-dict và kiểm tra
     shape khỏi online adaptation loop.
   - **Output:** Loader trả state-dict chỉ thuộc encoder hoặc báo lỗi.
   - **Error / compatibility:** Không dùng `strict=False` để bỏ qua key mismatch.
     Không để helper train hoặc cập nhật online parameters.
   - **Check:** Import helper không khởi tạo dataset hoặc optimizer.

2. **Đọc file và kiểm tra mapping.**
   - **File / Symbol:** proposed loader function trong helper mới.
   - **Input:** `pretrained_encoder_checkpoint` đã resolve.
   - **Action:** Kiểm tra file tồn tại, đọc payload bằng cơ chế checkpoint hiện
     có, rồi lấy đúng encoder mapping đã xác nhận ở Phase 0.
   - **Output:** Mapping tensor encoder và metadata checkpoint.
   - **Error / compatibility:** File thiếu, payload sai kiểu hoặc key thiếu phải
     raise lỗi có path và key liên quan.
   - **Check:** Test checkpoint thiếu file, thiếu key và payload sai kiểu.

3. **Kiểm tra kiến trúc tensor.**
   - **File / Symbol:** proposed loader function.
   - **Input:** Encoder state-dict và expected architecture.
   - **Action:** Kiểm tra input 38, hidden 64, 3 lớp, kernel 3 và latent 128.
   - **Output:** State-dict hợp lệ để load vào `SimpleWindowCnnEncoder`.
   - **Error / compatibility:** Sai shape phải raise `ValueError` nêu expected
     và observed shape.
   - **Check:** Test checkpoint sai latent, sai input channel và sai kernel.

4. **Tính checkpoint identity một lần.**
   - **File / Symbol:** loader và `src/core/artifact_integrity.py:11-20`,
     `sha256_file()`.
   - **Input:** Path file đã load.
   - **Action:** Tính SHA-256 sau khi kiểm tra file. Trả digest cùng path và role
     `pretrained_encoder`.
   - **Output:** Runtime có identity dùng cho metadata và threshold artifact.
   - **Error / compatibility:** Hash lỗi phải dừng load; không ghi digest giả.
   - **Check:** So sánh digest với Phase 0 inventory.

5. **Giữ mapping đã xác nhận trong test fixture.**
   - **File / Symbol:** `tests/online/test_online_streaming_baseline_contracts.py`
     hoặc test mới trong `tests/online/`.
   - **Input:** Payload fixture có key mapping giống checkpoint thật.
   - **Action:** Tạo checkpoint nhỏ trong `tmp_path` bằng kiến trúc CNN hiện có.
     Ghi metadata epoch 100 và encoder state-dict theo mapping thật.
   - **Output:** Fixture kiểm tra loader mà không cần remote.
   - **Error / compatibility:** Fixture không được kiểm tra key giả khác payload
     thật.
   - **Check:** Test fixture load thành công và trả đúng tensor count.

#### Stage 2.2: Load encoder qua adaptive baseline path

1. **Thêm checkpoint input vào base constructor.**
   - **File / Symbol:** `AdaptiveStreamingBaselineBase.__init__()`.
   - **Input:** `pretrained_encoder_checkpoint`, input dim, encoder dim và CNN
     settings từ config.
   - **Action:** Nhận path, resolve path tương đối theo repository root và lưu
     path đã resolve.
   - **Output:** Base class biết checkpoint nào sẽ load.
   - **Error / compatibility:** Main path thiếu path phải fail trước fit. Không
     tự tạo path theo method.
   - **Check:** Constructor test kiểm tra path lưu đúng entity-seed file.

2. **Khởi tạo container baseline.**
   - **File / Symbol:** `src/models/simple_window_cnn_autoencoder.py`,
     `_fit_backbone()` trong `adaptive.py`.
   - **Input:** Train feature dimension và encoder contract.
   - **Action:** Khởi tạo `SimpleWindowCnnAutoencoder` với latent 128 để giữ
     interface hiện tại gồm reconstruction output và latent output.
   - **Output:** Baseline có encoder và decoder riêng của baseline.
   - **Error / compatibility:** Input train dimension khác 38 phải fail.
   - **Check:** Assert encoder output dimension 128.

3. **Load chỉ encoder tensors.**
   - **File / Symbol:** `_fit_backbone()` hoặc helper load được gọi từ đó.
   - **Input:** Container baseline và encoder state-dict từ Stage 2.1.
   - **Action:** Load state-dict vào `self.backbone_.encoder`. Không load
     `self.backbone_.decoder` từ RedLamp.
   - **Output:** Encoder tensors khớp checkpoint; decoder vẫn thuộc baseline.
   - **Error / compatibility:** Load mismatch phải fail ngay. Không dùng
     `strict=False` nếu nó che key của encoder.
   - **Check:** So sánh từng encoder tensor giữa fixture checkpoint và model.

4. **Truyền input qua M2N2 và CANDI.**
   - **File / Symbol:** `M2N2StreamingBaseline.__init__()` và
     `CANDIStreamingBaseline.__init__()`.
   - **Input:** Baseline kwargs từ generator.
   - **Action:** Truyền `pretrained_encoder_checkpoint` và latent/CNN kwargs vào
     base constructor.
   - **Output:** Hai class dùng cùng loader nhưng giữ `adaptation_momentum`
     lần lượt 0.01 và 0.02.
   - **Error / compatibility:** Không đổi `_should_update()` của từng class.
   - **Check:** Test metadata và update policy của hai class riêng biệt.

5. **Kiểm tra score path.**
   - **File / Symbol:** `_score_backbone_windows()`.
   - **Input:** Window đã chuẩn hóa theo reference mean/std.
   - **Action:** Giữ reconstruction score và latent score hiện có. Dùng encoder
     mới load để tạo latent.
   - **Output:** Score path nhận latent 128 mà không thêm projection về 64.
   - **Error / compatibility:** Không đổi threshold hoặc EWMA formula trong
     stage này.
   - **Check:** Test một forward pass và kiểm tra score có finite value.

#### Stage 2.3: Giữ lifecycle của baseline-owned components

1. **Xác định decoder thuộc baseline.**
   - **File / Symbol:** `SimpleWindowCnnAutoencoder.decoder` và
     `_score_backbone_windows()`.
   - **Input:** Code hiện tại tạo decoder tại
     `src/models/simple_window_cnn_autoencoder.py:33-35`.
   - **Action:** Giữ decoder của container baseline. Không lấy RedLamp
     reconstruction head.
   - **Output:** M2N2/CANDI chỉ dùng RedLamp encoder và score head hiện tại.
   - **Error / compatibility:** Không thêm objective hoặc decoder training mới.
   - **Check:** State-dict loaded từ RedLamp không chứa decoder/classification
     head trong target load.

2. **Giữ adaptation reference update.**
   - **File / Symbol:** `_update_reference()` và `run_sequence()`.
   - **Input:** Windows được policy cho phép update.
   - **Action:** Giữ update mean/std hiện tại. Không cập nhật encoder parameters
     trên test stream.
   - **Output:** M2N2/CANDI vẫn khác nhau ở `_should_update()` nhưng dùng cùng
     encoder đã load.
   - **Error / compatibility:** Không dùng test label cho update decision.
   - **Check:** Record ghi `did_update` đúng policy và encoder parameters không
     đổi trong test stream.

3. **Giữ M2N2 policy.**
   - **File / Symbol:** `M2N2StreamingBaseline._should_update()`.
   - **Input:** Raw score, EWMA score và threshold.
   - **Action:** Giữ điều kiện cả raw và EWMA không vượt threshold.
   - **Output:** M2N2 chỉ update cửa sổ không bị xem là anomaly.
   - **Error / compatibility:** Không dùng `triage_decision` mới để thay policy.
   - **Check:** Test một case score dưới và một case score trên threshold.

4. **Giữ CANDI policy.**
   - **File / Symbol:** `CANDIStreamingBaseline._should_update()`.
   - **Input:** `triage_decision` hiện tại.
   - **Action:** Giữ update cho `gray_zone` hoặc `pnn_candidate`.
   - **Output:** CANDI giữ policy đã chốt trong contract.
   - **Error / compatibility:** Không đổi tên triage label trong stage loader.
   - **Check:** Test hai label update và một label không update.

#### Stage 2.4: Chặn fresh-encoder training

1. **Bỏ optimizer khỏi main load path.**
   - **File / Symbol:** `_fit_backbone()` trong `adaptive.py`.
   - **Input:** Checkpoint path hợp lệ.
   - **Action:** Khi path tồn tại, chỉ khởi tạo model container và load encoder.
     Không tạo `torch.optim.Adam` và không chạy reconstruction loop.
   - **Output:** Main run không train encoder.
   - **Error / compatibility:** Nếu load fail, raise lỗi; không gọi nhánh train
     cũ để cứu run.
   - **Check:** Monkeypatch optimizer hoặc training loop và assert không được gọi.

2. **Xử lý các backbone training fields.**
   - **File / Symbol:** base constructor và `_backbone_metadata()`.
   - **Input:** Config cũ có `backbone_epochs`, `backbone_batch_size`,
     `backbone_learning_rate`.
   - **Action:** Không dùng các field này để điều khiển main load path. Có thể
     ghi chúng là deprecated metadata nếu parser còn nhận.
   - **Output:** Config cũ không làm main train lại encoder.
   - **Error / compatibility:** Không âm thầm coi `backbone_epochs=0` là load
     thành công nếu checkpoint thiếu.
   - **Check:** Test checkpoint thiếu và field training còn sót đều fail rõ ràng.

3. **Đặt model ở evaluation mode sau load.**
   - **File / Symbol:** `_fit_backbone()` và `_score_backbone_windows()`.
   - **Input:** Encoder/decoder container đã load.
   - **Action:** Gọi `eval()` trước score. Không dùng dropout stochastic trong
     baseline score nếu contract không yêu cầu.
   - **Output:** Forward score ổn định cho cùng input và seed.
   - **Error / compatibility:** Không thay đổi `cnn_dropout` trong config.
   - **Check:** Hai forward pass cùng input trả cùng score trong eval mode.

4. **Ghi no-training evidence.**
   - **File / Symbol:** method metadata và runtime test.
   - **Input:** Checkpoint-loaded baseline.
   - **Action:** Ghi `encoder_initialization: pretrained_checkpoint` và role
     `pretrained_encoder`.
   - **Output:** Report phân biệt load checkpoint với train fresh.
   - **Error / compatibility:** Không ghi `trained_from_scratch` cho main path.
   - **Check:** Wrapper report có field no-training evidence.

### Tests

- **Loader round-trip test**
  - **Location:** `tests/online/test_online_streaming_baseline_contracts.py`
    hoặc test loader riêng.
  - **Level:** Unit.
  - **Setup:** Tạo checkpoint RedLamp-compatible trong `tmp_path`.
  - **Action:** Khởi tạo M2N2 và CANDI với cùng checkpoint.
  - **Expected result:** Encoder tensors khớp checkpoint; latent shape là 128.
  - **Edge cases:** File thiếu, key thiếu, shape sai, checkpoint có head dư.

- **No-training lifecycle test**
  - **Location:** tests online.
  - **Level:** Unit/integration.
  - **Setup:** Monkeypatch optimizer hoặc training loop.
  - **Action:** Gọi constructor/fit với checkpoint hợp lệ.
  - **Expected result:** Optimizer và fresh training loop không chạy.
  - **Edge cases:** Checkpoint path None hoặc path không tồn tại phải fail.

### Verification

#### Automated

- [ ] Loader fixture load đúng encoder state-dict.
- [ ] Loader reject file thiếu, key thiếu và shape sai.
- [ ] M2N2/CANDI dùng latent 128.
- [ ] No-training guard pass.

#### Manual

- [ ] Đọc một payload thật và xác nhận target load không chứa RedLamp decoder
  hoặc classification head.

### Risks and recovery

- **Risk:** Payload thật dùng key khác fixture. **Mitigation:** Phase 0 ghi key
  mapping trước khi viết loader. **Verification:** Test bằng payload mapping thật.
  **Recovery:** Dừng implementation Phase 2 và cập nhật mapping; không dùng
  `strict=False`.
- **Risk:** Decoder baseline chưa được pretrain. **Mitigation:** Giữ decoder
  baseline hiện tại và không đổi objective trong task này. **Verification:**
  Report ghi rõ chỉ encoder dùng checkpoint. **Recovery:** Không chạy
  performance nếu contract mới yêu cầu decoder pretrained.

### Complete when

- M2N2 và CANDI load đúng encoder 38-to-128 từ RedLamp.
- Runtime không train fresh encoder khi checkpoint hợp lệ.
- Policy update và score interface hiện tại vẫn chạy.

## Phase 3: Hoàn thiện provenance và report

### Goal

Phase này làm cho mỗi result có thể truy ngược method, stream, protocol và
checkpoint đã load.

### Dependencies

- Phase 1 đã tạo config contract.
- Phase 2 đã trả method metadata và checkpoint identity.

### Detailed steps

#### Stage 3.1: Chuẩn hóa method metadata

1. **Mở rộng adaptive metadata.**
   - **File / Symbol:** `AdaptiveStreamingBaselineBase._backbone_metadata()` và
     `_method_metadata()`.
   - **Input:** Encoder config, checkpoint identity và adaptation momentum.
   - **Action:** Ghi method, variant, encoder family, input dim, latent dim,
     CNN settings, checkpoint path, role, source và SHA-256.
   - **Output:** M2N2/CANDI calibration trả metadata đầy đủ.
   - **Error / compatibility:** Nếu deep baseline thiếu checkpoint identity,
     calibration fail thay vì ghi metadata rỗng.
   - **Check:** Assert từng field trong calibration output.

2. **Giữ policy metadata của M2N2.**
   - **File / Symbol:** `M2N2StreamingBaseline._method_metadata()`.
   - **Input:** Existing policy string và adaptation momentum.
   - **Action:** Giữ policy `update_on_non_anomalous_windows`, thêm checkpoint
     metadata từ base.
   - **Output:** Report phân biệt M2N2 với CANDI.
   - **Error / compatibility:** Không đổi tên method `m2n2` hoặc policy semantics.
   - **Check:** Test metadata policy exact match.

3. **Giữ policy metadata của CANDI.**
   - **File / Symbol:** `CANDIStreamingBaseline._method_metadata()`.
   - **Input:** Existing policy string và adaptation momentum.
   - **Action:** Giữ policy `update_on_gray_zone_and_pnn_candidate`, thêm
     checkpoint metadata từ base.
   - **Output:** Report phân biệt CANDI với M2N2.
   - **Error / compatibility:** Không đổi triage policy.
   - **Check:** Test metadata policy exact match.

4. **Bổ sung traditional hyperparameters.**
   - **File / Symbol:** `_method_metadata()` trong `src/baselines/online/frozen.py`.
   - **Input:** Model fields của Stumpy, KMeansAD và IForest.
   - **Action:** Ghi window size, normalization, p, n_clusters, n_init,
     n_estimators và các tham số frozen tương ứng. Ghi
     `checkpoint_role: not_applicable`.
   - **Output:** Traditional result có metadata đủ để tái lập score.
   - **Error / compatibility:** Không tạo checkpoint hoặc encoder metadata giả.
   - **Check:** Mỗi traditional class trả đúng fields của chính nó.

#### Stage 3.2: Gắn identity và SHA-256 cho artifact

1. **Truyền checkpoint hash vào calibration artifact.**
   - **File / Symbol:** `AdaptiveStreamingBaselineBase.calibrate()` và
     `build_threshold_artifact()`.
   - **Input:** SHA-256 từ Phase 2 và protocol values.
   - **Action:** Truyền `checkpoint_sha256` cho deep baseline artifact. Giữ
     schema baseline hiện tại nếu `method_name` không phải THESIS.
   - **Output:** Artifact có hash thật của RedLamp checkpoint.
   - **Error / compatibility:** Traditional ML không truyền hash giả. Không
     ép baseline dùng THESIS V4 triage fields.
   - **Check:** Validate threshold artifact và so sánh hash với file.

2. **Giữ threshold artifact backward-compatible.**
   - **File / Symbol:** `src/protocols/threshold_artifact.py`,
     `validate_threshold_artifact()`.
   - **Input:** Baseline artifact schema hiện tại.
   - **Action:** Chỉ yêu cầu checkpoint SHA-256 bắt buộc cho deep baseline có
     checkpoint. Không thay đổi rule schema THESIS V4.
   - **Output:** THESIS artifact cũ và baseline artifact mới vẫn parse đúng.
   - **Error / compatibility:** Artifact thiếu field bắt buộc phải fail theo
     schema của method.
   - **Check:** Chạy artifact tests cho THESIS, deep baseline và traditional.

3. **Giữ path resolved và source.**
   - **File / Symbol:** method metadata trong adaptive baseline.
   - **Input:** Config path tương đối và repository root.
   - **Action:** Ghi cả path config nhận vào và path file đã resolve nếu report
     cần audit local/remote.
   - **Output:** Người đọc biết checkpoint source là RedLamp và file nào được
     load.
   - **Error / compatibility:** Không thay path canonical trong config bằng
     absolute local path nếu điều đó làm mất khả năng tái lập remote.
   - **Check:** Report giữ canonical path và hash.

4. **Kiểm tra hash sau write.**
   - **File / Symbol:** `write_threshold_artifact()` và artifact output path.
   - **Input:** Artifact đã build.
   - **Action:** Validate trước write và đọc lại artifact sau write.
   - **Output:** Artifact trên disk giữ đúng checkpoint SHA-256.
   - **Error / compatibility:** Nếu read-back hash khác, dừng run.
   - **Check:** Test write/read round-trip.

#### Stage 3.3: Đưa metadata vào report cuối

1. **Đưa method metadata vào report.**
   - **File / Symbol:** `run_online_streaming_benchmark()`, report dict tại
     `scripts/benchmarks/run_online_streaming_benchmark.py:392-406`.
   - **Input:** `calibration["method_metadata"]`.
   - **Action:** Thêm metadata vào `online_execution` và giữ field cũ.
   - **Output:** Report có method, policy, encoder và checkpoint metadata.
   - **Error / compatibility:** Nếu calibration không trả metadata, fail contract
     test thay vì tự tạo metadata rỗng.
   - **Check:** Wrapper test tìm field trong JSON report.

2. **Ghi stream contract vào report.**
   - **File / Symbol:** `online_execution.stream_selections`.
   - **Input:** Metadata sau `select_online_stream_sequence()` và smoke truncation.
   - **Action:** Ghi entity, source length, absolute start, absolute end và
     selected sequence length.
   - **Output:** Report phân biệt range contract với `max_online_steps`.
   - **Error / compatibility:** Không thay global index trong records.
   - **Check:** Assert report range `[146,2200)` trước smoke truncation.

3. **Ghi protocol và status.**
   - **File / Symbol:** top-level report và `online_execution`.
   - **Input:** Protocol config, `benchmark_mode`, `max_online_steps`.
   - **Action:** Ghi window size, offline/online stride, threshold split,
     quantile, EWMA weights, label usage, mode và max steps.
   - **Output:** Report cho biết run dùng main hay smoke và protocol nào.
   - **Error / compatibility:** Không dùng test labels trong calibration.
   - **Check:** Report test kiểm tra `clean_validation` và `metrics_only`.

4. **Ghi artifact paths.**
   - **File / Symbol:** `report["artifact_paths"]`.
   - **Input:** Threshold, metrics, records và report paths.
   - **Action:** Giữ các path hiện có và thêm path cần cho method metadata nếu
     runtime tạo artifact mới.
   - **Output:** Report mở được artifact tương ứng.
   - **Error / compatibility:** Không lưu raw forward-pass output ngoài contract.
   - **Check:** Kiểm tra từng path tồn tại sau smoke.

#### Stage 3.4: Giữ tương thích report cũ

1. **Đọc caller của report fields.**
   - **File / Symbol:** report writer, summarizer và tests hiện có.
   - **Input:** Search toàn project cho `online_execution`, `artifact_paths`,
     `metric_history` và `records`.
   - **Action:** Liệt kê field mà caller đang đọc trước khi đổi schema.
   - **Output:** Danh sách field không được đổi tên.
   - **Error / compatibility:** Nếu caller phụ thuộc list shape hiện tại, chỉ
     thêm metadata ở key mới.
   - **Check:** Chạy wrapper test trước và sau thay đổi.

2. **Thêm field thay vì đổi field cũ.**
   - **File / Symbol:** `online_execution` report mapping.
   - **Input:** Metadata mới.
   - **Action:** Thêm key mới như `method_metadata` và giữ threshold, records,
     metrics paths hiện có.
   - **Output:** Parser cũ vẫn đọc được report.
   - **Error / compatibility:** Không đổi `online_variant` thành tên mới trong
     record.
   - **Check:** Parse report bằng test hiện có và test mới.

3. **Kiểm tra smoke/main separation.**
   - **File / Symbol:** report status fields.
   - **Input:** Main và smoke reports.
   - **Action:** Kiểm tra mode, max steps và sequence length khác nhau đúng
     nguyên nhân.
   - **Output:** Summarizer không gộp smoke vào performance table.
   - **Error / compatibility:** Nếu report thiếu mode, không dùng metric đó.
   - **Check:** Manual review hai report representative.

### Tests

- **Report provenance test**
  - **Location:** `tests/online/test_online_streaming_benchmark_wrapper.py`.
  - **Level:** Integration.
  - **Setup:** Fake baseline trả `method_metadata` và fake checkpoint fixture.
  - **Action:** Chạy wrapper đến report write.
  - **Expected result:** Report có method metadata, stream selection, protocol,
    status, artifact paths và checkpoint hash.
  - **Edge cases:** Traditional `not_applicable`, missing metadata và smoke cap.

- **Artifact hash test**
  - **Location:** test artifact/protocol hiện có hoặc test online mới.
  - **Level:** Unit/integration.
  - **Setup:** File checkpoint tạm.
  - **Action:** Build, write, load và validate threshold artifact.
  - **Expected result:** SHA-256 trong artifact bằng `sha256_file(path)`.
  - **Edge cases:** File bị thay sau hash phải fail read-back check.

### Verification

#### Automated

- [ ] Wrapper report test pass với deep baseline metadata.
- [ ] Artifact hash round-trip pass.
- [ ] Traditional report có `checkpoint_role: not_applicable`.

#### Manual

- [ ] Đọc một report M2N2/CANDI và một report traditional. Có thể xác định
  method, entity, seed, range, protocol và checkpoint status.

### Risks and recovery

- **Risk:** Report schema cũ bị phá. **Mitigation:** Chỉ thêm key mới ở boundary.
  **Verification:** Chạy parser/wrapper tests. **Recovery:** Khôi phục tên field
  cũ và giữ metadata mới ở key riêng.
- **Risk:** Hash không phải file runtime đã load. **Mitigation:** Tính hash từ
  path loader trả về. **Verification:** Hash round-trip. **Recovery:** Đánh dấu
  report invalid và không dùng kết quả.

### Complete when

- Main report có đầy đủ provenance.
- Hash deep baseline khớp file đã load.
- Traditional report không chứa checkpoint giả.

## Phase 4: Khóa runtime flow và test contract

### Goal

Phase này làm test fail ngay khi code quay lại full stream, latent cũ, variant
cũ hoặc fresh encoder training.

### Dependencies

- Phase 1-3 đã hoàn tất implementation path.
- Có fixture checkpoint theo mapping thật.

### Detailed steps

#### Stage 4.1: Khóa generator và range contract bằng test

1. **Cập nhật THESIS generator test.**
   - **File / Symbol:** `tests/online/test_online_benchmark_config_generation.py`,
     test tạo toàn bộ config.
   - **Input:** Generated THESIS configs.
   - **Action:** Assert range theo từng entity cho main và smoke.
   - **Output:** Test fail nếu generator bỏ range.
   - **Error / compatibility:** Giữ count 54 main và 54 smoke trong test.
   - **Check:** Chạy test riêng sau khi generate.

2. **Cập nhật baseline generator test.**
   - **File / Symbol:** `tests/online/test_online_streaming_benchmark_config_generation.py`.
   - **Input:** CANDI, M2N2 và traditional generated YAML.
   - **Action:** Assert neural variant `main`, latent 128, checkpoint path,
     protocol path và range.
   - **Output:** Test bảo vệ đúng 45 baseline main cells.
   - **Error / compatibility:** Test không tham chiếu sample A0 cũ.
   - **Check:** Chạy test sample cho từng method.

3. **Khóa protocol field.**
   - **File / Symbol:** generator tests và protocol config.
   - **Input:** `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`.
   - **Action:** Assert window 20, offline stride 20, online stride 1, clean
     validation, q99, EWMA 0.9/0.1 và `metrics_only`.
   - **Output:** Main methods dùng cùng protocol.
   - **Error / compatibility:** Không đổi protocol để làm test pass.
   - **Check:** Test config path và loaded protocol values.

4. **Khóa preflight count.**
   - **File / Symbol:** `tests/benchmarks/test_full_benchmark_matrix_preflight.py`.
   - **Input:** Preflight report.
   - **Action:** Đổi expected online count từ 81 thành 45 và kiểm tra method
     variant selection.
   - **Output:** Test fail nếu preflight nhận legacy neural variants.
   - **Error / compatibility:** Giữ offline count và threshold safety assertions.
   - **Check:** Chạy test preflight riêng.

#### Stage 4.2: Khóa runtime lifecycle và global indices

1. **Cập nhật deep baseline fixture dimension.**
   - **File / Symbol:** `tests/online/test_online_streaming_baseline_contracts.py`.
   - **Input:** Test sequence hiện có 3 channels và checkpoint fixture mới.
   - **Action:** Dùng 38 channels cho M2N2/CANDI hoặc tách fixture deep khỏi
     traditional fixture.
   - **Output:** Test phản ánh contract input dim 38.
   - **Error / compatibility:** Traditional tests có thể giữ fixture nhỏ nếu
     không kiểm tra neural encoder.
   - **Check:** CANDI/M2N2 forward pass không fail vì input dimension.

2. **Kiểm tra clean-validation calibration.**
   - **File / Symbol:** baseline contract tests.
   - **Input:** Train, clean validation và test sequence có labels chỉ để metric.
   - **Action:** Gọi `calibrate()` với validation không có anomaly decision từ
     test. Assert threshold source là clean validation stride-1 EWMA.
   - **Output:** Calibration không đọc test labels.
   - **Error / compatibility:** Empty validation phải giữ lỗi hiện có.
   - **Check:** Test `threshold_source` và protocol weights.

3. **Kiểm tra global indices sau range selection.**
   - **File / Symbol:** `tests/online/test_online_stream_range.py` và baseline
     contract tests.
   - **Input:** Sequence được cắt `[10,32)` hoặc range contract thật trong
     fixture.
   - **Action:** Chạy baseline và THESIS selector trên sequence đã cắt.
   - **Output:** `point_index`, `window_start_index` và `window_end_index` giữ
     offset entity-global.
   - **Error / compatibility:** Không chấp nhận index bắt đầu lại từ zero.
   - **Check:** Assert record đầu và record cuối.

4. **Kiểm tra M2N2 no-training và policy.**
   - **File / Symbol:** baseline contract tests và `m2n2.py`.
   - **Input:** Checkpoint fixture, raw score, EWMA score và threshold.
   - **Action:** Chạy fit/calibrate/run sequence và kiểm tra optimizer không chạy;
     sau đó kiểm tra điều kiện update.
   - **Output:** Encoder không đổi; policy chỉ update non-anomalous windows.
   - **Error / compatibility:** Checkpoint lỗi phải fail trước run sequence.
   - **Check:** So sánh state-dict trước/sau test stream.

5. **Kiểm tra CANDI no-training và policy.**
   - **File / Symbol:** baseline contract tests và `candi.py`.
   - **Input:** Checkpoint fixture và triage labels.
   - **Action:** Chạy fit/calibrate/run sequence; kiểm tra update cho gray zone,
     pnn candidate và không update cho label khác.
   - **Output:** Encoder không đổi; policy giữ semantics hiện tại.
   - **Error / compatibility:** Không dùng test labels để tạo triage label.
   - **Check:** Assert `did_update` và state-dict.

#### Stage 4.3: Khóa preflight và chạy focused suite

1. **Cập nhật test report shape.**
   - **File / Symbol:** wrapper test.
   - **Input:** Report có method metadata mới.
   - **Action:** Assert field mới nhưng giữ assert field cũ.
   - **Output:** Report contract được khóa ở boundary.
   - **Error / compatibility:** Không assert thứ tự JSON key.
   - **Check:** Load JSON bằng parser hiện có.

2. **Chạy focused online suite.**
   - **File / Symbol:** các test online đã nêu trong plan.
   - **Input:** Source/config/test sau Phase 1-3.
   - **Action:** Chạy focused pytest command.
   - **Output:** Tất cả contract tests pass.
   - **Error / compatibility:** Test fail vì payload mismatch phải quay lại
     Stage 2.1, không sửa expected để che lỗi.
   - **Check:** Lưu output test và failure path.

3. **Chạy full online test subset.**
   - **File / Symbol:** `tests/online/` liên quan online runtime.
   - **Input:** Focused suite đã pass.
   - **Action:** Chạy subset đầy đủ để phát hiện regression THESIS runtime.
   - **Output:** Không có regression ngoài contract change.
   - **Error / compatibility:** Nếu test legacy yêu cầu A0 cho baseline, cập
     nhật test theo mapping contract, không đổi THESIS A0/A1/A2.
   - **Check:** Phân loại failure thành implementation hoặc stale expectation.

4. **Review test record đầu/cuối.**
   - **File / Symbol:** generated records và test assertions.
   - **Input:** Records của baseline và THESIS test.
   - **Action:** Kiểm tra global index, threshold, `did_update` và label usage.
   - **Output:** Test chứng minh calibration/update không dùng anomaly labels.
   - **Error / compatibility:** Record sai range làm Phase 4 fail.
   - **Check:** Manual review một record đầu và cuối.

### Tests

- **Contract suite**
  - **Location:** `tests/online/` và `tests/benchmarks/`.
  - **Level:** Unit/integration.
  - **Setup:** Generated configs và checkpoint fixture.
  - **Action:** Chạy generator, runtime, report và preflight tests.
  - **Expected result:** Range, variant, latent, loader, policy, metadata và
    count đều pass.
  - **Edge cases:** Mismatch key, stale config, missing range và global offset.

### Verification

#### Automated

- [ ] Focused pytest command trong plan pass.
- [ ] Full online test subset pass.
- [ ] Preflight test báo 54 THESIS và 45 baseline.

#### Manual

- [ ] Review record đầu/cuối và xác nhận index là entity-global.
- [ ] Xác nhận test labels chỉ xuất hiện trong metric calculation.

### Risks and recovery

- **Risk:** Fixture không phản ánh checkpoint thật. **Mitigation:** Một fixture
  dùng key mapping đã xác nhận. **Verification:** Loader test và shape test.
  **Recovery:** Dừng smoke nếu payload thật không load được.

### Complete when

- Focused suite và full online subset pass.
- Preflight test khóa đúng count và variant.
- Runtime tests khóa loader, policy, range và global index.

## Phase 5: Smoke, preflight và bàn giao full matrix

### Goal

Phase này chạy một đường end-to-end nhỏ, kiểm tra artifact và chỉ mở full matrix
khi mọi gate pass.

### Dependencies

- Phase 0-4 đã pass.
- Remote checkpoint, THESIS Stage B checkpoint và protocol artifact đều tồn tại.
- Local revision và remote revision đã được đối chiếu.

### Detailed steps

#### Stage 5.1: Kiểm tra local config và entry points

1. **Chọn representative configs.**
   - **File / Symbol:** generated THESIS, M2N2, CANDI và một traditional YAML.
   - **Input:** Main/smoke configs sau Phase 1.
   - **Action:** Chọn cùng entity `machine_1_6`, seed 6 và protocol.
   - **Output:** Bộ config đại diện cho từng entry point.
   - **Error / compatibility:** Không chọn stale A0/A1/A2 baseline YAML.
   - **Check:** Path nằm trong main/smoke allow-list.

2. **Chạy baseline dry-run.**
   - **File / Symbol:** `scripts/benchmarks/run_online_streaming_benchmark.py`,
     `run_online_streaming_benchmark()`.
   - **Input:** `--benchmark-config` và protocol path hiện có.
   - **Action:** Chạy `--dry-run` với M2N2, CANDI và traditional representative.
   - **Output:** Report dry-run ghi config path, protocol path và output path.
   - **Error / compatibility:** Dry-run không được build dataset hoặc train
     encoder.
   - **Check:** Đọc `benchmark_status=dry_run`.

3. **Resolve THESIS config.**
   - **File / Symbol:** `scripts/run_thesis_online_benchmark.py` và THESIS
     online runtime.
   - **Input:** THESIS smoke config.
   - **Action:** Dùng entry point hiện có để resolve Stage B checkpoint,
     threshold artifact và absolute range.
   - **Output:** THESIS config resolve được mà không stream full series.
   - **Error / compatibility:** Nếu parser không hỗ trợ dry-run, chỉ thực hiện
     config resolution đã có; không thêm CLI flag trong task này.
   - **Check:** Kiểm tra resolved Stage B path và range metadata.

4. **Chạy preflight trước smoke.**
   - **File / Symbol:** `scripts/ops/preflight_full_benchmark_matrix.py`,
     `build_preflight_report()`.
   - **Input:** Full generated tree.
   - **Action:** Chạy `--json` và đọc report.
   - **Output:** Report ready với THESIS 54 và baseline 45.
   - **Error / compatibility:** Preflight fail thì không chạy smoke.
   - **Check:** Kiểm tra protocol safety và count.

#### Stage 5.2: Chạy hai end-to-end smoke đại diện

1. **Chọn THESIS smoke.**
   - **File / Symbol:** THESIS config `O1/A2/machine_1_6/seed6` và
     `run_thesis_online_benchmark.py`.
   - **Input:** Stage B checkpoint, threshold artifact, range `[146,2200)` và
     protocol chung.
   - **Action:** Chạy smoke với `max_online_steps` hiện có.
   - **Output:** Report, metrics, records và provenance của THESIS.
   - **Error / compatibility:** Không dùng smoke metric cho performance table.
   - **Check:** Report ghi benchmark mode smoke và absolute range request.

2. **Chọn M2N2 smoke.**
   - **File / Symbol:** M2N2 `main/machine_1_6/seed6` và
     `run_online_streaming_benchmark.py`.
   - **Input:** RedLamp checkpoint tương ứng và cùng range `[146,2200)`.
   - **Action:** Chạy smoke sau khi THESIS smoke pass.
   - **Output:** Report, metrics, records, threshold artifact và method metadata.
   - **Error / compatibility:** Checkpoint load fail thì dừng, không fallback.
   - **Check:** Report ghi latent 128, checkpoint SHA-256 và no-training evidence.

3. **Kiểm tra stream records.**
   - **File / Symbol:** `online_records.json` và report `stream_selections`.
   - **Input:** Hai smoke output.
   - **Action:** Đọc record đầu/cuối và đối chiếu range, window indices, threshold
     và `did_update`.
   - **Output:** Records giữ global index và smoke cap đúng.
   - **Error / compatibility:** Nếu point index bắt đầu từ zero sau range, dừng
     full matrix.
   - **Check:** So sánh `absolute_start_index` với `window_start_index` đầu.

4. **Kiểm tra artifact retention.**
   - **File / Symbol:** smoke output directories.
   - **Input:** Report, metrics, records, thresholds và provenance.
   - **Action:** Xác nhận artifact cần cho audit tồn tại; không giữ raw forward
     outputs ngoài contract.
   - **Output:** Disk usage nằm trong retention policy.
   - **Error / compatibility:** Artifact thiếu làm smoke fail.
   - **Check:** Liệt kê output tree trước khi mở rộng run.

5. **Đối chiếu THESIS và M2N2 smoke.**
   - **File / Symbol:** hai report JSON.
   - **Input:** Same entity, seed, range và protocol.
   - **Action:** So sánh stream metadata, không so sánh để ép latent bằng nhau.
   - **Output:** Chỉ khác biệt method, encoder source, latent và policy.
   - **Error / compatibility:** Range hoặc protocol khác nhau làm smoke fail.
   - **Check:** Manual checklist trước release gate.

#### Stage 5.3: Review smoke và quyết định release gate

1. **Kiểm tra provenance completeness.**
   - **File / Symbol:** smoke reports.
   - **Input:** Report của THESIS và M2N2.
   - **Action:** Kiểm tra method, entity, seed, range, protocol, checkpoint role,
     path, hash, status và artifact paths.
   - **Output:** Hai report đủ thông tin audit.
   - **Error / compatibility:** Thiếu field thì sửa Phase 3 và chạy lại smoke.
   - **Check:** Dùng checklist contract field.

2. **Kiểm tra runtime branch đã chạy.**
   - **File / Symbol:** records và metric history.
   - **Input:** Smoke stream ngắn.
   - **Action:** Ghi branch policy thực sự được chạm. Không tuyên bố coverage
     của branch mà stream không đi qua.
   - **Output:** Smoke result có giới hạn được ghi rõ.
   - **Error / compatibility:** Không dùng smoke để kết luận performance.
   - **Check:** Manual review `did_update` và triage decisions.

3. **Kiểm tra revision và remote state.**
   - **File / Symbol:** local git revision và remote runtime state.
   - **Input:** Revision `5f3ac2654c3a4d9b5a0d25cc5b08672c9b0fe70c` hoặc revision
     đã được phê duyệt mới hơn.
   - **Action:** Đọc revision, GPU, disk, active jobs và checkpoint paths.
     Không upload, reset hoặc cleanup remote.
   - **Output:** Runtime target khớp code và artifact đã kiểm tra.
   - **Error / compatibility:** Revision khác thì dừng full launch.
   - **Check:** Lưu read-only inspection result.

4. **Ra quyết định release gate.**
   - **File / Symbol:** preflight JSON và smoke reports.
   - **Input:** Kết quả các bước 1-3.
   - **Action:** Chỉ chuyển Phase 5.4 khi preflight ready, hai smoke pass và
     không còn mismatch.
   - **Output:** Trạng thái `approved_for_full_matrix` trong run checklist.
   - **Error / compatibility:** Bất kỳ gate nào fail đều giữ trạng thái blocked.
   - **Check:** Người review xác nhận từng gate.

#### Stage 5.4: Bàn giao full matrix

1. **Tạo launch manifest.**
   - **File / Symbol:** preflight allow-list và generated main YAML.
   - **Input:** 54 THESIS và 45 baseline main paths.
   - **Action:** Chọn exact paths từ preflight. Không dùng stale paths hoặc smoke
     paths.
   - **Output:** Manifest có 99 main combinations.
   - **Error / compatibility:** Count khác 99 thì không launch.
   - **Check:** Đếm manifest theo method.

2. **Chốt artifact retention.**
   - **File / Symbol:** output layout và retention policy.
   - **Input:** Report, metrics, records, threshold artifact và provenance.
   - **Action:** Giữ report-ready artifacts và checkpoint cần audit. Không giữ
     raw forward-pass outputs mặc định.
   - **Output:** Disk budget phù hợp full matrix.
   - **Error / compatibility:** Disk không đủ thì dừng trước launch.
   - **Check:** Kiểm tra disk read-only.

3. **Chạy full matrix theo manifest.**
   - **File / Symbol:** online benchmark entry points hiện có.
   - **Input:** Launch manifest và revision đã gate.
   - **Action:** Chạy đúng 54 THESIS và 45 baseline. Traditional ML giữ frozen.
   - **Output:** Main reports có cùng contract và không lẫn smoke.
   - **Error / compatibility:** Job fail phải giữ report/status để resume hoặc
     retry exact cell; không chạy cell khác variant để thay thế.
   - **Check:** Theo dõi count completed/failed theo manifest.

4. **Đối chiếu kết quả sau run.**
   - **File / Symbol:** main reports và result inventory.
   - **Input:** Output của 99 cells.
   - **Action:** Kiểm tra mỗi cell có report, stream metadata, protocol,
     provenance và status hoàn tất.
   - **Output:** Bảng performance chỉ lấy main cells đủ provenance.
   - **Error / compatibility:** Cell thiếu artifact hoặc hash không khớp bị loại
     khỏi bảng và ghi trạng thái lỗi.
   - **Check:** Chạy post-run inventory và review missing cells.

### Tests

- **Preflight integration test**
  - **Location:** `tests/benchmarks/test_full_benchmark_matrix_preflight.py`.
  - **Level:** Integration.
  - **Setup:** Generated matrix và checkpoint resolver fixture.
  - **Action:** Gọi `build_preflight_report()`.
  - **Expected result:** Report ready, THESIS 54, baseline 45.
  - **Edge cases:** Stale variant, missing range, missing checkpoint.

- **End-to-end smoke**
  - **Location:** Existing benchmark entry points and smoke output tree.
  - **Level:** End-to-end.
  - **Setup:** One THESIS and one M2N2 smoke config.
  - **Action:** Resolve, calibrate, select range, run stream và write report.
  - **Expected result:** Report, metrics, records và provenance đầy đủ.
  - **Edge cases:** Remote revision mismatch, missing checkpoint, disk shortage.

### Verification

#### Automated

- [ ] `.venv/bin/python -m scripts.ops.preflight_full_benchmark_matrix --json`
  trả `54` THESIS và `45` baseline.
- [ ] Baseline dry-run pass.
- [ ] Một THESIS smoke và một M2N2 smoke tạo đủ report-ready artifacts.
- [ ] Focused pytest và `git diff --check` pass trước full launch.

#### Manual

- [ ] Kiểm tra report smoke có range, latent, checkpoint source, SHA-256 và
  `benchmark_mode=smoke`.
- [ ] Kiểm tra remote revision, GPU, disk, active jobs và checkpoint paths bằng
  read-only workflow.

### Risks and recovery

- **Risk:** Smoke không chạm mọi policy branch. **Mitigation:** Ghi branch đã
  chạy. **Verification:** Review records. **Recovery:** Chạy thêm diagnostic
  smoke nếu cần; không đổi performance matrix.
- **Risk:** Remote revision khác local. **Mitigation:** Đọc revision trước launch.
  **Verification:** So sánh exact revision. **Recovery:** Dừng launch; không
  upload hoặc reset remote.
- **Risk:** Full run tạo quá nhiều artifact. **Mitigation:** Giữ retention set
  đã chốt. **Verification:** Kiểm tra disk và output inventory. **Recovery:**
  Dừng launch trước cell tiếp theo; không xóa broad output tree.

### Complete when

- Preflight báo `54 + 45`.
- THESIS và M2N2 smoke pass với provenance đầy đủ.
- Launch manifest có đúng 99 main cells.
- Main result inventory phân biệt rõ completed, failed và missing cells.

## Interface and data changes

### Config fields

- THESIS task config nhận `absolute_start_index` và `absolute_end_index`.
- M2N2/CANDI baseline kwargs nhận `pretrained_encoder_checkpoint`.
- M2N2/CANDI dùng `encoder_dim=128`.
- Baseline config dùng `online_variant: main`.

### Runtime metadata

- Deep baseline metadata có checkpoint path, role, source và SHA-256.
- Traditional metadata ghi `checkpoint_role: not_applicable`.
- Report ghi stream range, protocol, mode và max online steps.

### Compatibility

- THESIS `A0/A1/A2` giữ nguyên semantics.
- Existing report keys giữ nguyên; metadata mới được thêm ở key mới.
- `full-spec-v2.md` vẫn là lịch sử và không trở thành nguồn chuẩn cho matrix
  mới.

## Deployment and rollout

1. Đọc checkpoint và generated tree trước khi sửa code.
2. Sửa generator, preflight và constructors.
3. Regenerate YAML và chạy generator tests.
4. Implement loader và chạy runtime tests.
5. Thêm provenance và chạy report tests.
6. Chạy focused suite và preflight.
7. Chạy một THESIS smoke và một M2N2 smoke.
8. Chỉ sau khi smoke pass mới chạy 99 main cells.

Nếu Phase 2 không đọc được RedLamp checkpoint, rollback an toàn là không chạy
main M2N2/CANDI. Không rollback bằng cách bật fresh encoder, vì cách đó vi phạm
contract.

## Documentation changes

- Cập nhật `documents/inventories/online-benchmark-combinations-and-smoke-checklist.md`
  theo matrix 99.
- Giữ `documents/spec/online_benchmark_contract.md` là nguồn chuẩn.
- Ghi payload mapping thật và checkpoint limitations vào implementation log
  sau khi Phase 0 hoàn tất.
- Không sửa semantics của `documents/spec/full-spec-v2.md`.

## Final verification

- [ ] Mỗi entity có đúng absolute range trong THESIS và baseline config.
- [ ] M2N2 và CANDI có đúng 9 main configs cho mỗi method.
- [ ] Ba traditional methods có đúng 9 main configs cho mỗi method.
- [ ] Preflight báo 54 THESIS và 45 baseline.
- [ ] RedLamp encoder load đúng 38-to-128 và không load RedLamp heads.
- [ ] M2N2/CANDI không train fresh encoder khi checkpoint tồn tại.
- [ ] M2N2/CANDI giữ đúng policy riêng.
- [ ] Report có stream, protocol, method metadata, checkpoint role/path/SHA-256
  và main/smoke status.
- [ ] THESIS smoke và M2N2 smoke pass trước full matrix.
- [ ] Chỉ 99 main cells xuất hiện trong performance table.

## Assumptions and non-blocking uncertainties

- Checkpoint key names phải lấy từ payload thật ở Phase 0. Detail này không
  đoán key name.
- `SimpleWindowCnnAutoencoder.decoder` vẫn là decoder thuộc baseline vì
  contract chỉ cho load RedLamp encoder. Nếu contract sau này yêu cầu decoder
  pretrained, cần lập decision mới trước benchmark performance.
- Main benchmark không dùng fallback fresh encoder. Direct unit tests phải dùng
  checkpoint fixture hợp lệ.
- Generated stale YAML có thể còn trên disk. Preflight và launch manifest phải
  loại chúng khỏi matrix trước khi cleanup.

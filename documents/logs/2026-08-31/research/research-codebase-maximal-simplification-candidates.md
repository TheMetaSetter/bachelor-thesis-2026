---
date: 2026-08-31T09:21:06+07:00
researcher: OpenAI Codex
topic: "Xác định nhiều nhất có thể các đoạn code, file hoặc module có thể xoá hoặc chỉnh sửa để codebase tối giản nhất có thể"
status: complete
revision: 6cd32ac94cc20876828f5c62fd2df5c3ac557587
branch: dev
---

# Research: các ứng viên tối giản codebase

## Summary

Ý chính ở đây là codebase có thể giảm rất nhiều file mà không cần đổi thuật toán chính. Phần dư lớn nhất nằm ở hai cây config song song, config cũ do generator không dọn, wrapper tương thích, artifact bị commit nhầm và các nhánh legacy. Phần khó hơn nằm ở model THESIS, config validation, trainer và online engine: chúng không nên xoá nguyên khối, nhưng cần giảm số đường chạy và làm rõ một owner cho mỗi trách nhiệm.

Kiểm kê hiện tại ghi nhận:

- 346 file Python first-party, tổng 59.865 dòng trong `src/`, `scripts/`, `tests/` và `demo/`.
- 9 file trong `src/` vượt giới hạn 500 dòng.
- 92 callable trong `src/` vượt giới hạn 50 dòng.
- 365 config nằm trong cây phụ `scripts/configs/experiment/`, trong khi source và test hiện hành dùng `configs/experiment/`.
- 306 config baseline online cũ không còn thuộc ma trận mà generator hiện tại sinh ra.
- 18 config STUMPY offline nằm trong thư mục tên cũ `stumpy_channel_ab/`.
- 28 file `.DS_Store` và 10 artifact build LaTeX đang được Git theo dõi.
- 1 module tính threshold online không có importer tĩnh, 1 script 542 dòng bị trùng byte-for-byte, 2 cặp preset config trùng byte-for-byte, và nhiều symbol không có caller tĩnh.

Nếu chỉ thực hiện nhóm rủi ro thấp và migration cơ học, codebase có thể bỏ khoảng 700 file config/artifact/script thừa. Con số này không bao gồm các nhánh protocol cần quyết định của con người.

## Research question

Xác định nhiều nhất có thể các đoạn code, file hoặc module có thể xoá hoặc chỉnh sửa để codebase tối giản nhất có thể.

Phạm vi nghiên cứu gồm `src/`, `scripts/`, `configs/`, `tests/`, `demo/`, `presentation/`, `documents/spec/` và `bsc-thesis-ref-codebases/`. Em không sửa implementation trong lần research này.

## System context

Codebase có ba đường chạy chính liên quan trực tiếp đến việc tối giản:

1. Offline THESIS: config experiment đi qua runner benchmark, trainer/evaluator, `ThesisMultitaskModel`, checkpoint và artifact reporting.
2. Online THESIS: wrapper benchmark resolve checkpoint, ghi `reference_checkpoint_path`, rồi gọi online TTA engine cho từng causal window.
3. Baseline offline/online: generator tạo ma trận YAML, runner chọn baseline, calibration chọn threshold, sau đó ghi report.

`documents/spec/full-spec-v3.md` là spec normative hiện hành. Spec giữ một public model THESIS, một causal window online và yêu cầu bypass `view_a/view_b` trong v3 (`documents/spec/full-spec-v3.md:118-128`, `documents/spec/full-spec-v3.md:187-199`). `codebase_preferences.md` yêu cầu public model phải sở hữu lifecycle, helper không được phân tán lifecycle bằng mixin, file không quá 500 dòng và callable không quá 50 dòng (`codebase_preferences.md:71-75`, `codebase_preferences.md:99-104`).

## Method and evidence levels

Em dùng bốn mức kết luận:

- **Xoá ngay, rủi ro thấp:** source, test, config và docs hiện hành không có caller hoặc có bản canonical trùng hoàn toàn.
- **Xoá sau migration cơ học:** có wrapper, path cũ hoặc test/docs còn trỏ tới; cần đổi reference trước.
- **Chỉnh sửa, không xoá nguyên module:** code đang chạy nhưng có nhiều owner, nhiều nhánh hoặc facade khó trace.
- **Cần quyết định:** source, spec và test chưa thống nhất, hoặc experiment có thể vẫn cần cho luận văn.

Một kết quả `rg` không thấy caller chỉ là bằng chứng tĩnh. Python có thể import động hoặc người dùng có thể gọi CLI bằng tay. Vì vậy, các entrypoint độc lập được xếp thấp hơn code thư viện không có caller.

## Detailed findings

### A. Nhóm nên xoá trước

| ID | File hoặc symbol | Bằng chứng hiện tại | Hành động tối giản |
| --- | --- | --- | --- |
| A1 | `src/engine/online_tta/threshold_calibration.py` | Bốn hàm chỉ có definition tại dòng 9, 19, 33, 47; không có import/call trong `src/`, `scripts/`, `tests/` hoặc `demo/`. Runtime calibration hiện hành ở `online_calibration.py`. | Xoá nguyên module và chạy test online/calibration. |
| A2 | `scripts/recalibrate_thesis_threshold_artifacts_v4.py` | Trùng byte-for-byte với `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py`; mỗi file 542 dòng. Test và các tool mới dùng bản trong `scripts/ops/`. | Giữ bản `scripts/ops/`, xoá bản top-level, sửa reference docs cũ nếu có. |
| A3 | `configs/model/redlamp_baseline_redlamp_aligned.yaml` | Trùng byte-for-byte với `configs/model/redlamp_baseline.yaml`. | Giữ tên canonical ngắn hơn; xoá preset có hậu tố. |
| A4 | `configs/task/multitask_tsad_redlamp_multiclass_window20_redlamp_aligned.yaml` | Trùng byte-for-byte với `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml`. | Chọn một tên canonical, đổi reference, xoá bản còn lại. |
| A5 | 28 file `.DS_Store` | Git đang theo dõi 28 file metadata Finder; `.gitignore` chưa chặn `.DS_Store`. | Xoá khỏi Git và thêm `.DS_Store` vào `.gitignore`. |
| A6 | 10 file build LaTeX trong `presentation/thesis_slides_22127208_vi-VN/slides/` | Các extension `aux`, `bbl`, `blg`, `fdb_latexmk`, `fls`, `nav`, `out`, `snm`, `toc`, `xdv` là output build, không phải source. | Xoá khỏi Git; ignore theo extension. Chỉ giữ PDF nếu repository chủ đích lưu bản render. |
| A7 | `src/models/redlamp_mlp_baseline.py` | Chỉ là compatibility shim; không có reference trong source, test, notebook hoặc config hiện hành. Public model canonical là `src/models/redlamp_baseline.py`. | Xoá sau một test import âm tính xác nhận không còn API nội bộ phụ thuộc. |

### B. Symbol không có caller tĩnh

Các symbol sau không có name reference ngoài definition trong phạm vi first-party. Đây là các ứng viên xoá theo từng hàm, không phải lý do xoá toàn bộ file:

| Symbol | Vị trí | Ghi chú |
| --- | --- | --- |
| `NoOpArtifactSink` | `src/engine/artifact_sinks.py:21-32` | Builder trả list rỗng khi không có sink, nên class no-op không cần cho natural control flow. |
| `build_smd_dataloaders` | `src/data/loaders.py:383` | Registry/dataset bundle hiện hành dùng entrypoint khác. |
| `compute_a2_online_contrastive_loss` | `src/engine/online_tta/online_losses.py:69-82` | Online A2 hiện dùng token multi-positive InfoNCE, không gọi loss cũ này. |
| `extract_last_point_label` | `src/baselines/online/base.py:149` | Không có baseline caller. |
| `register_archival_dataset_components` | `src/core/runtime_components.py:30` | Không có entrypoint gọi registration này. |
| `resolve_output_root` | `demo/loaders.py:101` | Demo không gọi helper này. |
| `signature_window_to_dict` | `src/engine/online_tta/signature_verification.py:200-209` | Chỉ có deserialize `signature_window_from_dict` được giữ; không có serializer caller. Cần kiểm tra artifact cũ trước khi xoá. |

`clear_registry`, `Windowizer`, `collect_clean_validation_scores` và `resume_online_runtime` cũng bị static scan đánh dấu, nhưng test hoặc API có thể gọi trực tiếp. Em không xếp chúng vào nhóm xoá.

Scanner ban đầu cũng đánh dấu `move_initialization_batch_to_device`, nhưng kiểm tra import cho thấy `thesis_multitask_state_memory_mixin.py:17` import nó bằng alias `_move_initialization_batch_to_device` và gọi tại dòng 171. Em đã loại symbol này khỏi danh sách xoá.

### C. Xoá cây config phụ `scripts/configs/experiment/`

`scripts/configs/experiment/` chứa 365 YAML. Source và test active trỏ tới `configs/experiment/`; các generator cũng ghi vào `configs/experiment/`. Không có Python runtime/test trỏ tới `scripts/configs/experiment/`.

So sánh hai cây cho thấy 272 file cùng tên nhưng khác nội dung, 245 file chỉ có trong cây canonical và 3 file chỉ có trong cây `scripts/`. Điều này nguy hiểm hơn duplicate thuần túy, vì người dùng có thể chạy nhầm config cùng tên nhưng khác semantics.

Hướng tối giản:

1. Chuyển ba config chỉ có trong `scripts/configs/` sang cây canonical nếu chúng vẫn cần.
2. Sửa command/docs còn trỏ tới cây cũ.
3. Xoá toàn bộ `scripts/configs/`.
4. Thêm test cấm source/config reference tới `scripts/configs/`.

Đây là thay đổi giảm nhiều file nhất với rủi ro runtime thấp.

### D. Dọn config generator không còn thuộc ma trận hiện hành

Generator baseline online chỉ khai báo 5 method. CANDI/M2N2 chỉ có variant `reference_adapter_redlamp_encoder`; STUMPY/KMeans-AD/IForest chỉ có `main` (`scripts/benchmarks/generate_online_streaming_benchmark_configs.py:27-34`). Mỗi method phải có `3 entity × 3 seed × 2 mode = 18` file.

| Thư mục | Hiện có | Generator hiện tại | Dư |
| --- | ---: | ---: | ---: |
| `configs/experiment/online_benchmark/candi/` | 90 | 18 | 72 |
| `configs/experiment/online_benchmark/m2n2/` | 90 | 18 | 72 |
| `configs/experiment/online_benchmark/stumpy/` | 72 | 18 | 54 |
| `configs/experiment/online_benchmark/kmeans_ad/` | 72 | 18 | 54 |
| `configs/experiment/online_benchmark/iforest/` | 72 | 18 | 54 |
| **Tổng** | **396** | **90** | **306** |

Contract xác nhận generator chỉ nên tạo một config `main` cho mỗi baseline/entity/seed và không chạy config baseline cũ mang nhãn `A0/A1/A2` (`documents/spec/online_benchmark_contract.md:413-428`). Vì generator không dọn output cũ, các file stale tồn tại vô thời hạn.

Generator offline ghi STUMPY vào directory `stumpy/`, không phải `stumpy_channel_ab/` (`scripts/benchmarks/generate_offline_benchmark_configs.py:28-33`, `scripts/benchmarks/generate_offline_benchmark_configs.py:125-131`). Do đó, 18 file trong `configs/experiment/offline_benchmark/stumpy_channel_ab/` là ứng viên stale.

Ngoài ra có ba config ad hoc không do generator hiện tại sinh:

- `configs/experiment/offline_benchmark/thesis/*__smoke5.yaml`;
- `configs/experiment/online_benchmark/thesis/*__smoke5.yaml`;
- `configs/experiment/online_benchmark/thesis/*__smoke_cuda.yaml`.

Nếu các lần debug đã xong, nên xoá ba file này. Nếu cần tái lập, nên chuyển tham số thành CLI override thay vì giữ preset một lần.

Generator cần được chỉnh để ghi manifest expected paths và phát hiện stale files. Không nên tự động xoá ngoài root cụ thể của chính generator.

### E. Config có thể xoá sau khi xác nhận không còn lệnh chạy thủ công

Tìm theo full path không thấy source, test hoặc docs active trỏ tới các preset sau:

- `configs/data/smd_smoke.yaml`;
- `configs/model/redlamp_baseline_redlamp_aligned.yaml`;
- `configs/model/redlamp_cnn_baseline.yaml`;
- `configs/model/thesis_multitask_redlamp_multiclass_cnn_simple.yaml`;
- `configs/model/thesis_multitask_redlamp_multiclass_redlamp_aligned.yaml`;
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`;
- `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml`;
- `configs/task/multitask_tsad_redlamp_multiclass_window20_redlamp_aligned.yaml`;
- `configs/task/multitask_tsad_window10_binary.yaml`.

Hai file trong danh sách này đã trùng byte-for-byte với preset khác. Các file còn lại cần kiểm tra command history hoặc job remote, vì config có thể được truyền bằng tay mà không xuất hiện trong source.

Hai nhóm `thesis_q95` gồm 18 offline config và 54 online config không phải code chết. Full spec cho phép cả q95 và q99 (`documents/spec/full-spec-v3.md:647`). Chỉ xoá 72 file này nếu anh chốt protocol/report cuối cùng không cần sensitivity theo threshold quantile.

### F. Bỏ nhánh legacy two-view trong online stream

`OnlineWindowBatcher` vẫn nhận `view_noise_std`, `view_dropout_probability` và `include_legacy_views`; class còn giữ `_build_view()` và hai nhánh ghi `view_a/view_b` (`src/data/stream.py:205-284`). Tất cả caller active đều truyền `include_legacy_views=False`. Không có config/source/test nào truyền `True`.

Full spec yêu cầu active online input chỉ có một causal window và bypass legacy views (`documents/spec/full-spec-v3.md:187-199`). Vì vậy có thể xoá:

- ba tham số/state `view_noise_std`, `view_dropout_probability`, `include_legacy_views` khỏi `OnlineWindowBatcher`;
- `_build_view()`;
- hai nhánh tạo/log `view_a` và `view_b`;
- việc luân chuyển hai giá trị view qua `online_calibration.py`, `online_engine_run.py`, runner offline/online và config validation;
- các test chỉ truyền `0.0` để giữ API cũ.

Đây là một nhánh code có thể xoá theo natural control flow. Không cần thêm flag migration mới.

### G. Chuẩn hoá tên checkpoint baseline online

CANDI và M2N2 nhận cả `pretrained_encoder_checkpoint` và alias `pretrained_model_checkpoint`. `AdaptiveStreamingBaselineBase` chọn alias bằng `pretrained_model_checkpoint or pretrained_encoder_checkpoint`, sau đó lưu state với tên `pretrained_model_checkpoint` (`src/baselines/online/adaptive.py:84-205`).

Generator, config, test và contract đều dùng tên `pretrained_encoder_checkpoint` (`documents/spec/online_benchmark_contract.md:406-424`). Nội dung loader cũng chỉ lấy encoder tensor. Vì vậy nên:

1. Bỏ tham số alias `pretrained_model_checkpoint` khỏi adaptive base, CANDI và M2N2.
2. Đổi state nội bộ về `pretrained_encoder_checkpoint`.
3. Sửa error message hiện đang nói nhầm rằng `pretrained_model_checkpoint` là bắt buộc.

Không có checked-in config nào dùng alias cũ. Rủi ro còn lại chỉ là caller Python bên ngoài repository.

`balance_binary_classes_within_batch` cũng là alias legacy cho `train_balance_classes`. Checked-in config không dùng alias, nhưng một test vẫn truyền trực tiếp. Có thể xoá alias và `resolve_balance_classes_setting()` sau khi chốt không cần tương thích API bên ngoài.

### H. Hợp nhất facade và wrapper

Hiện có ba kiểu facade khó trace:

- `src/models/online_adaptation.py` thay `sys.modules[__name__]` bằng module implementation.
- `src/models/redlamp_baseline.py` dùng wildcard import từ `baseline_impl`.
- `src/engine/online_tta/online_engine.py` re-export cả symbol private lẫn public từ nhiều helper.

Các facade này đang được test/import nên không xoá ngay. Hướng tối giản là mỗi public file sở hữu public API và export tường minh. Implementation helper chỉ giữ primitive có tên rõ. Không dùng `sys.modules` hoặc `import *`.

Các script top-level sau chỉ là wrapper 5-11 dòng sang package con:

- `scripts/train.py`, `scripts/evaluate.py`;
- `scripts/run_ablation.py`, `scripts/run_multiseed_experiments.py`;
- `scripts/run_offline_benchmark.py`, `scripts/run_online_streaming_benchmark.py`;
- `scripts/run_thesis_offline_benchmark.py`, `scripts/run_thesis_online_benchmark.py`;
- `scripts/run_two_stage_offline_pretraining.py`, `scripts/run_online_adaptation.py`;
- `scripts/generate_benchmark_smoke_configs.py`;
- `scripts/generate_offline_benchmark_configs.py`;
- `scripts/generate_online_benchmark_configs.py`;
- `scripts/generate_online_streaming_benchmark_configs.py`;
- `scripts/generate_smd_benchmark_configs.py`;
- `scripts/summarize_benchmark_results.py`.

Nên chọn đúng một canonical CLI layer. Phương án ít migration cho người dùng nhất là giữ tên top-level public, chuyển implementation vào function được import tường minh, rồi xoá wrapper package còn lại. Phương án giảm file nhiều nhất là migrate docs/tests sang `scripts.<package>.<module>` và xoá toàn bộ wrapper top-level. Cần chọn một, không giữ cả hai.

Hai shell wrapper `scripts/launch_tmux_comparative_smd_experiment.sh` và `scripts/launch_tmux_two_stage_experiment.sh` chỉ `exec` sang `scripts/launchers/`. Có thể xoá sau khi sửa command trong docs.

### I. Thu gọn `ThesisMultitaskModel`

`src/models/thesis_multitask.py` tự mô tả là self-contained, nhưng class thật sự kế thừa bốn mixin public (`src/models/thesis_multitask.py:1-85`). Thư mục `src/models/thesis_multitask_impl/` có 20 file và 4.981 dòng. Mixin state và loss tiếp tục kế thừa nhiều mixin nhỏ hơn. Public entrypoint vì vậy không cho người đọc thấy constructor, forward, loss, memory update và checkpoint lifecycle trong một luồng rõ.

Điều này xung đột trực tiếp với quy ước model trong `codebase_preferences.md:71-75`.

Không nên gộp 4.981 dòng vào một file. Trình tự tối giản an toàn là:

1. Xoá các alias, branch và output field không còn caller.
2. Giữ `ThesisMultitaskModel` sở hữu `__init__`, `forward`, train/validation step, memory lifecycle và checkpoint hooks.
3. Chuyển phép toán thuần thành function thuần trong các module theo trách nhiệm: routing geometry, loss math, memory math, serialization schema.
4. Bỏ MRO mixin và các wrapper method chỉ chuyển tiếp `self`.
5. Giữ checkpoint key và output contract bằng snapshot/test đã được refresh có chủ đích.

`thesis_multitask_state_memory_mixin.py` dài 664 dòng và có 6 method vượt 50 dòng. Nên tách theo ba lifecycle thật: thu token pool, khởi tạo memory, cập nhật/calibrate memory. Không tách theo mixin mới.

### J. Chín file `src/` vượt hard limit

Scanner có sẵn trong `tests/codebase_compliance.py` ghi nhận:

| File | Dòng | Chỉnh sửa tối giản |
| --- | ---: | --- |
| `src/data/augment.py` | 979 | Giữ orchestrator ngắn; chuyển 11 anomaly-family math thành function thuần theo family. Không đổi label/mask semantics. |
| `src/engine/trainer.py` | 933 | Tách epoch execution, validation aggregation, checkpoint decision và diagnostics; `Trainer.train` hiện dài 399 dòng. |
| `src/core/config.py` | 818 | Giữ loading/precedence; chuyển section validation vào validator có owner rõ. |
| `src/core/config_model_validation.py` | 805 | Tách schema validation khỏi cross-field semantics; `_validate_model_and_task_config` hiện dài 480 dòng. |
| `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py` | 664 | Bỏ mixin lifecycle; tách function theo ba bước memory nêu trên. |
| `src/metrics/pointwise.py` | 622 | Không tách chỉ vì line count. Trước hết xác định adapter/metric legacy thật sự không còn dùng; giữ phép toán metric cùng nhau. |
| `src/engine/online_tta/online_engine_run.py` | 570 | Thay context `dict` nhiều field bằng vài dataclass nhỏ theo phase; rút ngắn build/run/finalize. File hiện có TODO về quá nhiều field và call complexity. |
| `src/engine/evaluator.py` | 554 | Tách loop thu output khỏi timeline reconstruction/metric export; `Evaluator.evaluate` dài 140 dòng. |
| `src/baselines/online/adaptive.py` | 517 | Xoá checkpoint alias, gom calibration common path, giữ method-specific update trong CANDI/M2N2. |

Scanner tìm 92 callable vi phạm. Các điểm ưu tiên cao nhất là config validation 480/223 dòng, trainer 399 dòng, threshold artifact validation 259/169 dòng, routing forward 228 dòng, config parsing 186 dòng và RedLamp constructor 185 dòng.

`tests/compliance/test_codebase_compliance_scanner.py` chỉ test scanner trên file tạm, không chạy scanner trên `src/` thật. Vì vậy giới hạn trong preferences hiện không chặn regression. Nên thêm một compliance test chạy trên repository. Nếu không thể sửa 101 vi phạm ngay, dùng allowlist baseline cụ thể và chỉ cho phép con số giảm.

### K. Trách nhiệm trùng trong trainer, evaluator, config và runner

Các cặp nên gom về một owner:

- `src/core/config.py` và `src/core/config_model_validation.py`: loading/default/precedence thuộc `config.py`; schema và cross-field semantics thuộc validator nhỏ có tên theo section.
- `src/engine/trainer.py` và `src/engine/evaluator.py`: pointwise timeline reconstruction, threshold application và metric aggregation cần dùng cùng primitive thay vì hai flow gần giống nhau.
- `scripts/benchmarks/run_thesis_offline_benchmark.py` 858 dòng và `_internal/run_thesis_offline_benchmark_helpers.py` 619 dòng: runner chỉ nên parse/coordinate; artifact collection/export nên có một owner.
- `scripts/benchmarks/run_online_streaming_benchmark.py` 505 dòng: baseline selection, calibration, run, report cần thành bốn stage có contract tường minh.

Các helper I/O lặp nhiều nhất là `_load_json` trong 9 file, `_write_json` trong 7 file và `_utc_now_iso` trong 11 file. Chỉ nên gom các implementation thật sự giống nhau vào `scripts/ops/_report_io.py`. Không nên tạo utility framework chung cho mọi script.

Năm generator lặp `_entity_token`; bốn generator lặp `_output_dir`. `scripts/benchmarks/_config_generation_helpers.py` đã là owner tự nhiên. Nên chuyển identity/path primitives vào đó, nhưng giữ method-specific config builder trong từng generator.

### L. Generic online path cũ

`src/engine/online_loop.py` và `scripts/experiments/run_online_adaptation.py` tạo một generic online path khác với THESIS online TTA engine hiện hành. Benchmark THESIS hiện đi qua `scripts/benchmarks/run_thesis_online_benchmark.py` và `src/engine/online_tta/`.

Tuy nhiên, full spec v3 vẫn nêu `src/engine/online_loop.py` là online loop cần event dispatcher (`documents/spec/full-spec-v3.md:125-127`). Vì vậy evidence chưa đủ để xoá ngay.

Cần chốt một trong hai:

- nếu `online_tta` là runtime duy nhất, migrate test/demo còn lại rồi xoá `online_loop.py`, generic experiment runner và wrapper `scripts/run_online_adaptation.py`;
- nếu generic loop vẫn là public demo API, đổi tên rõ vai trò và không cho nó chia sẻ tên/contract với THESIS runtime.

### M. Xung đột quanh RedLamp encoder loader

`documents/spec/online_benchmark_contract.md:406-424` nói CANDI/M2N2 phải dùng `src/baselines/online/redlamp_encoder_checkpoint.py`. Module này có test riêng và strict-load `encoder.*`.

Nhưng runtime `src/baselines/online/adaptive.py` hiện import `RedLampReconstructionModel` và loader từ `src/models/online_redlamp_reconstruction.py`; nó không import `redlamp_encoder_checkpoint.py`. `src/models/simple_window_cnn_autoencoder.py` hiện chủ yếu được test dùng để tạo checkpoint.

Đây là xung đột giữa documented contract và executable runtime. Không nên xoá loader cũ hoặc model test helper trước khi xác định implementation nào là canonical. Sau khi chốt:

- nếu runtime mới đúng, cập nhật contract/test rồi xoá `redlamp_encoder_checkpoint.py` và `simple_window_cnn_autoencoder.py`;
- nếu contract cũ đúng, adaptive runtime phải quay lại loader encoder và xoá reconstruction loader song song.

### N. Script vận hành một lần

Các entrypoint sau không có importer hoặc test theo full path. Điều này không chứng minh chúng chết, vì người dùng có thể chạy bằng tay:

- `scripts/ops/build_phase7_audit_artifacts.py`;
- `scripts/ops/debug_pilot_combination_uq_reporting.py`;
- `scripts/ops/render_online_phase_metric_table.py`;
- `scripts/ops/render_offline_phase_metric_table.py`;
- `scripts/ops/read_offline_uq_table2.py`;
- `scripts/ops/re_evaluate_and_prune_thesis_runs_remaining.py`;
- `scripts/ops/inspect_thesis_run_artifacts.py`;
- `scripts/analysis/kl_divergence_anomaly_archive.py`;
- `scripts/visualization/visualize_smd_benchmark_train_test.py`;
- `scripts/visualization/visualize_classification_diagnostics.py`;
- `scripts/visualization/visualize_training_metrics.py`.

Nên xoá `debug_*` và `*_remaining.py` khi task đã đóng. Các renderer/visualizer nên giữ chỉ khi report hoặc artifact manifest ghi rõ command tái lập. Nếu hai renderer chỉ khác input schema, dùng một renderer với hai adapter nhỏ.

### O. Reference codebases không có reference ngoài chính nó

Không có source runtime import trực tiếp từ `bsc-thesis-ref-codebases/`. Nhiều repository vẫn có giá trị nghiên cứu và được `codebase_preferences.md` chỉ định giữ lại, nên không thể xoá toàn bộ thư mục.

Bốn checkout sau không có reference ngoài chính thư mục của chúng:

| Checkout | Kích thước gần đúng | Kết luận |
| --- | ---: | --- |
| `bsc-thesis-ref-codebases/MtsCID/` | 11 MB | Chuyển ra archive/submodule hoặc xoá nếu không còn được trích trong thesis. |
| `bsc-thesis-ref-codebases/h-pad/` | 3.6 MB | Cùng điều kiện. |
| `bsc-thesis-ref-codebases/mulan-main/` | 348 KB | Cùng điều kiện. |
| `bsc-thesis-ref-codebases/sto-transformer-main/` | 232 KB | Cùng điều kiện. |

Không nên xoá CANDI, M2N2, STUMPY, CARLA, RedLamp, Time-Series-Library hoặc DALL-E trong đợt này. Chúng có vai trò baseline, provenance hoặc được quy ước repository yêu cầu.

### P. Test code có thể thu gọn

Test có 19.601 dòng. Hai file `tests/runtime/test_learning_rate_scheduler.py` và `tests/runtime/test_learning_rate_scheduler_additional.py` lặp fixtures/classes/helpers ở 161 dòng đầu. Nên gom setup chung vào fixture cục bộ, sau đó giữ test case theo behavior.

Các helper `_build_batch`, `_build_model`, `_build_reference_checkpoint` lặp nhiều trong test online/model. Chỉ nên gom fixture có cùng semantics. Không nên tạo factory nhận hàng chục option vì nó làm test khó đọc.

Chỉ xoá test compatibility sau khi xoá chính alias/path mà test bảo vệ. Không xoá test chỉ để làm suite ngắn hơn.

### Q. Các phần không nên xoá mặc định

- `direct_branch_routing`: code, test, script và config đều rất mới. Đây là ablation/proposal đang hoạt động, không phải dead code. Chỉ xoá sau khi kết quả cuối cùng bác bỏ nó.
- `reference_checkpoint_path`: generator không ghi trực tiếp, nhưng wrapper resolve metadata rồi bổ sung field cho runtime. Ontology hiện xem đây là field canonical trỏ tới `stage_b_best_checkpoint`. Không xoá như một alias thông thường.
- q95 config: spec cho phép q95/q99; cần quyết định protocol/report.
- metric math trong `src/metrics/`: không xoá theo line count; phải kiểm tra report caller và metric contract.
- checkpoint schema fields mang tên legacy: full spec nói tên artifact được giữ để tương thích. Việc tên cũ không tự động cho phép schema migration.

## Recommended deletion order

Trình tự này giảm rủi ro và cho phép mỗi commit có một lý do rõ:

1. **Artifact hygiene:** `.DS_Store`, LaTeX build outputs, `.gitignore`.
2. **Exact duplicates:** script recalibration và hai cặp preset.
3. **Dead source:** `threshold_calibration.py`, shim RedLamp MLP và symbol không caller đã xác minh.
4. **Config SSOT:** migrate ba file duy nhất, xoá `scripts/configs/`, thêm test cấm cây phụ.
5. **Generated stale configs:** xoá 306 online baseline + 18 STUMPY cũ + 3 debug configs; thêm stale detection.
6. **Legacy one-window cleanup:** xoá two-view branch và tham số luân chuyển.
7. **Canonical naming:** checkpoint alias và class-balancing alias.
8. **Facade/wrapper:** chọn một public layer, migrate import/CLI, xoá layer còn lại.
9. **Structural refactor:** model lifecycle, config validation, trainer/evaluator, online runner.
10. **Decision-gated cleanup:** generic online path, q95, direct routing, reference checkouts và RedLamp loader conflict.

Mỗi bước nên là một thay đổi độc lập. Sau mỗi bước, chạy test hẹp theo module, test config generation, checkpoint roundtrip, rồi một full-flow smoke combination theo quy ước repository.

## Verification performed

### Static checks

- Dùng `git ls-files`, `rg`, `find`, `cmp`, `diff -qr`, AST và `wc -l` để kiểm file, caller, duplicate, matrix config và line limit.
- Scanner `tests.codebase_compliance.scan_source_size_violations(Path("src"))` trả 9 file violations và 92 callable violations.
- `cmp` xác nhận script recalibration và hai cặp preset trùng byte-for-byte.
- Search source/test xác nhận runtime và generator active chỉ trỏ tới `configs/experiment/`.

### Narrow tests

Lệnh đã chạy:

```text
.venv/bin/python -m pytest -q \
  tests/compliance/test_codebase_compliance_scanner.py \
  tests/compliance/test_src_refactor_contracts.py \
  tests/online/test_online_streaming_benchmark_config_generation.py
```

Kết quả: 7 test pass, 1 test fail. Test fail là `test_registry_and_model_output_surface_match_snapshot`: snapshot `thesis_state_dict_keys` không còn khớp model hiện tại; khác biệt đầu tiên là `encoder.network.network.1.bias` so với `encoder.network.0.bias`.

Failure này tồn tại trước khi có thay đổi implementation trong research này. Nó cho thấy snapshot refactor hiện stale. Cần xác định thay đổi key là có chủ đích hay regression trước khi dùng test này làm safety net.

## Evidence

- `prompts/1_research_prompt.md:1-340` — quy trình research, phân loại evidence và yêu cầu ghi report.
- `codebase_preferences.md:71-75` — public model phải sở hữu lifecycle; helper không được phân tán lifecycle bằng mixin.
- `codebase_preferences.md:91-104` — least-codepaths, 50 dòng/callable và 500 dòng/file.
- `documents/spec/full-spec-v3.md:118-128` — owner hiện hành cho offline model, trainer, evaluator, online entrypoint/loop/stream.
- `documents/spec/full-spec-v3.md:187-199` — online v3 dùng đúng một causal window và bypass legacy views.
- `src/data/stream.py:205-284` — implementation two-view legacy vẫn còn trong batcher.
- `src/engine/online_tta/threshold_calibration.py:1-55` — module dead-code candidate gồm bốn primitive không có caller.
- `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:19-50` — ma trận baseline online canonical.
- `documents/spec/online_benchmark_contract.md:406-428` — contract RedLamp checkpoint và cấm chạy config baseline A0/A1/A2 cũ.
- `scripts/benchmarks/generate_offline_benchmark_configs.py:21-33` — generator offline chọn directory `stumpy`.
- `scripts/benchmarks/generate_offline_benchmark_configs.py:120-132` — generator ghi file vào canonical config root.
- `src/models/thesis_multitask.py:1-85` — public THESIS class và MRO mixin hiện tại.
- `tests/codebase_compliance.py:1-74` — AST scanner cho hard limits.
- `tests/compliance/test_codebase_compliance_scanner.py:1-52` — test scanner chỉ dùng source tạm.
- `src/baselines/online/adaptive.py:84-205` — hai checkpoint parameter và tên state nội bộ hiện tại.
- `src/baselines/online/redlamp_encoder_checkpoint.py:1-116` — strict encoder-only loader được contract nêu.
- `src/engine/artifact_sinks.py:14-32` — protocol và no-op implementation không caller.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| Offline baseline entities | `machine-1-6`, `machine-3-4`, `machine-3-9` | `scripts/benchmarks/generate_offline_benchmark_configs.py:26` | Generator offline |
| Benchmark seeds | `6`, `8`, `36` | `scripts/benchmarks/generate_offline_benchmark_configs.py:27`; `generate_online_streaming_benchmark_configs.py:21` | Offline/online |
| Online baseline variants | CANDI/M2N2: `reference_adapter_redlamp_encoder`; traditional: `main` | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:27-34` | Generator online |
| Offline protocol | q99, EWMA 0.9 profile | `scripts/benchmarks/generate_offline_benchmark_configs.py:29` | Baseline offline |
| Online deep baseline protocol | q99.5, EWMA 0.9 profile | `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:44` | CANDI/M2N2 online |
| Active online views | `include_legacy_views=False` | `src/engine/online_tta/online_calibration.py:37-39`; `src/baselines/online/base.py:95-97` | THESIS/baseline online |
| Public THESIS online input | one causal window | `documents/spec/full-spec-v3.md:187-199` | Full spec v3 |

## Conflicts and uncertainties

1. Contract RedLamp nói runtime dùng `redlamp_encoder_checkpoint.py`, nhưng adaptive runtime import implementation reconstruction khác. Chưa thể xoá một bên mà không chốt owner.
2. `scripts/configs/` không có caller active, nhưng ba file chỉ có ở cây này cần được phân loại trước khi xoá cả cây.
3. Static caller search không thấy lệnh shell, notebook cell động, command history hoặc remote job không lưu trong Git.
4. q95 là protocol hợp lệ theo spec dù generator q99 hiện hành không sinh nó.
5. Direct branch routing là code mới, nhưng chưa có bằng chứng trong phạm vi research này rằng nó sẽ thuộc final method.
6. Snapshot state dict đang stale. Chưa biết source hay snapshot là phía sai nếu chỉ dựa trên test output.

## Open questions

1. Anh có cần giữ API Python bên ngoài repository cho các alias `pretrained_model_checkpoint`, `balance_binary_classes_within_batch` và `redlamp_mlp_baseline` không?
2. q95 có còn là protocol sensitivity cần đưa vào luận văn không?
3. `direct_branch_routing` là ablation tạm thời hay ứng viên final method?
4. Generic `online_loop.py` còn là public demo API hay `src/engine/online_tta/` đã thay thế hoàn toàn?
5. RedLamp baseline online phải load encoder-only theo contract cũ hay load reconstruction model theo runtime hiện tại?
6. Bốn reference checkout không có reference còn được dùng cho chapter/appendix nào ngoài Git không?

## Final assessment

Phần nên làm đầu tiên không phải refactor model. Nên xoá artifact và exact duplicate, hợp nhất config SSOT, dọn stale generator outputs, rồi xoá nhánh two-view legacy. Bốn bước này giảm hàng trăm file và một đường chạy cũ mà không đổi thuật toán THESIS.

Sau đó mới xử lý mixin model, config validation, trainer/evaluator và online engine. Các thay đổi này cần snapshot/checkpoint/full-flow safety net đúng trước, vì test snapshot hiện đang stale.

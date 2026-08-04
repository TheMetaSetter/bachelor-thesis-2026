---
date: 2026-07-28 18:58:21 +07
researcher: TheMetaSetter
git_commit: 71e798d488ddbd9edcbd100a41dc0384f96558be
branch: dev
repository: bachelor-thesis-2026
topic: "Identify legacy processing-flow candidates against full-spec-v2 and full-spec-v3"
tags: [research, legacy-flow, simplification, smd, anomaly-detection]
status: complete
last_updated: 2026-07-28
last_updated_by: TheMetaSetter
---

# Research: Legacy processing-flow markers

## 1. Research question

Mục tiêu của nghiên cứu này là xác định những nhánh xử lý không còn thuộc processing flow được quy định trong `full-spec-v2.md` và `full-spec-v3.md`. Tài liệu này chỉ đánh dấu, phân loại và nêu điều kiện xoá. Chưa có code nào bị xoá trong lượt nghiên cứu này.

## 2. Kết luận ngắn

Flow chính hiện tại là SMD, mô hình THESIS hai stage, khởi tạo memory từ train, hiệu chỉnh threshold trên clean validation, sau đó chạy offline evaluation và online A0/A1/A2. V3 bổ sung stochastic query với `M=10`, cửa sổ `L=20`, và giữ baseline theo native protocol riêng.

Các marker legacy có độ tin cậy cao tập trung ở bốn nhóm:

1. runtime/config cho three-stage cũ và các phase Stage 1/Stage 2/Stage 3;
2. model `reconstruction_mlp_ae`, vì model này không nằm trong main experiment matrix của hai spec;
3. compatibility path cho checkpoint và alias config cũ;
4. các test fixture còn dùng contract cũ như `view_a` và `view_b`.

Không được xoá toàn bộ baseline chỉ vì tên có chữ `legacy` hoặc vì baseline không dùng THESIS triage. `full-spec-v2.md` quy định các flow baseline là một phần của full benchmark và baseline phải giữ native protocol riêng.

## 3. Contract và active flow được giữ lại

### 3.1 THESIS flow

```text
SMD config
  -> load_experiment_config
  -> SMD loader/scaler/window/collate
  -> O0 hoặc O1, Stage A multitask pretraining
  -> initialize continuous/discrete memory from train only
  -> freeze encoder and memories
  -> Stage B fusion/prediction-head fine-tuning
  -> clean-validation threshold calibration
  -> offline evaluation hoặc online A0/A1/A2
  -> metrics, provenance, checkpoint, report, demo
```

Các contract cần bảo toàn là batch offline dạng `[B, 20, D]`, online scorer không nhận label, output ổn định gồm `hidden`, `recon`, `logits`, `point_scores`, `window_scores`, `aux`, và chỉ projector được cập nhật ở online THESIS. V3 yêu cầu dense stochastic retrieval, discrete top-3 Gumbel retrieval và đúng 10 mẫu Monte Carlo trong các thí nghiệm stochastic.

### 3.2 Baseline flow

```text
benchmark config
  -> native traditional/neural/online baseline
  -> baseline-specific scoring and calibration
  -> benchmark artifacts and report
```

Các baseline trong v2 gồm RedLamp, traditional baselines và online baselines như CANDI, M2N2, STUMPY, KMeansAD và Isolation Forest. Chúng không được kế thừa THESIS triage, PNN hoặc projector update, nhưng điều đó không làm chúng thành code cần xoá.

## 4. Bảng marker cần xử lý

| ID | Vị trí | Marker | Bằng chứng | Phân loại hiện tại | Điều kiện xoá |
|---|---|---|---|---|---|
| L01 | `src/models/thesis_multitask_impl/thesis_multitask_components.py:23-25,213-255,332` | `STAGE3_PHASE_LEGACY_NAME`, `ThreeStageRuntimeConfig`, các phase Stage 1/2/3 | V2 và v3 đều mô tả lifecycle hai stage A/B; code vẫn đọc three-stage | Legacy candidate, độ tin cậy cao | Xác nhận không cần load config/checkpoint three-stage cũ; migrate tests và config trước |
| L02 | `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:56-100,160-200` | Nhánh `stage1_classification`, `stage1_reconstruction`, `stage2_recovery`, `multitask_pretraining`, Stage 3 | Các nhánh này điều khiển objective, freeze và trainable set ngoài lifecycle A/B hiện hành | Legacy processing branch, độ tin cậy cao | Xoá sau khi L01 đã được xử lý và test phase A/B đầy đủ |
| L03 | `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py` | Stage 3 memory/fusion substeps | State schedule còn giữ tên và logic Stage 3 song song với Stage B | Legacy processing branch | Xác nhận không còn checkpoint/state round-trip cũ |
| L04 | `src/core/config.py:21-26,236-300` và `src/core/config_model_validation.py` | Alias và validator cho `three_stage`, `stage3_prototype_warmup`, `training_phase` | Code ghi rõ đây là historical/legacy compatibility; active rerun dùng two-stage | Legacy compatibility boundary | Chốt policy checkpoint cũ; sau đó xoá alias, validator và fixture tương ứng |
| L05 | `configs/model/thesis_multitask_three_stage_window20.yaml`, `configs/model/thesis_multitask_three_stage_comparative_smd.yaml` | Config three-stage | Không thuộc active two-stage configs của v2/v3 | Legacy config | Xoá cùng migration tests/docs; không xoá riêng lẻ |
| L06 | `src/core/runtime_components.py:17`, `src/core/config_model_validation.py`, `src/models/reconstruction_mlp_ae.py`, `configs/model/reconstruction_mlp_ae.yaml` | Standalone reconstruction MLP autoencoder | Không có trong main experiment matrix của hai spec; vẫn được registry và test cũ dùng | Non-main model candidate, độ tin cậy cao | Xác nhận không còn benchmark/demo cần model này; migrate/remove dedicated tests and registry snapshot |
| L07 | `src/data/stream.py:45,54-84` | `stream_window_mode="legacy_stride"` | Active online configs/code dùng `sliding_stride_1` hoặc `nonoverlap_tail`; legacy mode chỉ là một nhánh stride tổng quát | Legacy mode candidate | Kiểm tra mọi config/artifact reader; giữ nguyên class stream, chỉ xoá mode nếu không còn caller |
| L08 | `src/engine/online_tta/checkpoint_resolution.py:31-52,124-132`; `src/models/online_impl/online_adaptation.py:35-43` | Flat checkpoint fallback `resolve_legacy_reference_checkpoint_path` | Metadata resolver là canonical path của benchmark; fallback phục vụ layout cũ | Legacy compatibility candidate | Mọi online experiment chuyển sang metadata-only và test fallback được thay bằng test rejection |
| L09 | `src/models/online_impl/online_adaptation.py:191-200` | `online_encoder_params` target group | Online engine active chỉ chấp nhận `projector_params`; spec cũng khoá projector-only update | Unsupported legacy branch candidate | Kiểm tra direct callers ngoài engine; sau đó xoá branch và thu hẹp type/config validation |
| L10 | `tests/online/test_online_stream.py:77-78` và các test online tạo `view_a/view_b` | Cũ online fixture contract | Full-spec contract test yêu cầu active online batch không có `view_a/view_b`; fixture cũ còn shape 100 | Legacy test contract | Cập nhật fixture theo active batch `[1,20,D]`, không xoá production code chỉ vì fixture cũ |
| L11 | `src/data/api.py`, `src/data/datasets/anomaly_archive.py`, các analysis/visualization script liên quan | AnomalyArchive dataset path | Hai spec chính tập trung vào SMD; repo vẫn có registry, parser, analysis và tests cho AnomalyArchive | Non-main dataset candidate, chưa đủ bằng chứng để gọi legacy | Xác nhận project scope chỉ giữ SMD; tách research/analysis archival trước khi xoá |
| L12 | `src/engine/online_tta/triage.py:44-64`, `src/baselines/online/*` | Tên `classify_legacy_baseline_window` | V2 yêu cầu baseline giữ native protocol riêng; hàm này đang phục vụ đúng việc cô lập baseline khỏi THESIS triage | False positive: active baseline flow | Giữ lại; chỉ đổi tên nếu muốn rõ nghĩa và có plan riêng |
| L13 | `scripts/ops/*` cho prune/backfill/re-evaluate | Operational artifact tooling | Không phải scientific processing flow chính, nhưng có thể cần cho provenance và report | Out of scope, chưa tự động xoá | Chỉ xoá theo retention/operations plan riêng |

## 5. Caller và test evidence

- Active online benchmark configs dùng `target_param_group: projector_params` và `stream_window_mode: sliding_stride_1`.
- Online engine từ chối target group khác `projector_params`; vì vậy `online_encoder_params` hiện là nhánh rộng hơn active engine contract.
- `resolve_stage_b_checkpoint` ưu tiên metadata, nhưng vẫn gọi fallback flat path nếu `reference_checkpoint_path` được cung cấp. Đây là seam compatibility cần xử lý riêng.
- Các test cũ cho `view_a/view_b` không chứng minh production flow cần contract đó; ngược lại, full-spec online contract tests đã kiểm tra rằng active batch không chứa hai field này.
- `reconstruction_mlp_ae` và `anomaly_archive` vẫn có caller/test trực tiếp. Vì vậy chúng là non-main candidates, chưa phải bằng chứng đủ để xoá ngay.

## 6. Historical context

Codebase từng hỗ trợ nhiều biến thể thí nghiệm và compatibility layer. Những lớp này giải thích vì sao hiện còn three-stage names, alias config, flat checkpoint fallback và test fixture nhiều view. Tuy nhiên, lịch sử đó không thay đổi normative lifecycle hiện tại của v2/v3.

## 7. Open questions và deletion gates

Trước khi xoá cần chốt bốn điểm:

1. “Main experiments” có nghĩa là toàn bộ 189 benchmark cells trong v2, hay chỉ minimal THESIS design T0/T1/T2 trong v3? Nếu bao gồm toàn bộ v2 thì baseline flow phải giữ.
2. Có cần load checkpoint/config cũ để tái lập kết quả không? Nếu không, có thể xoá L01-L05 và L08 sau khi cập nhật migration tests.
3. Có giữ model `reconstruction_mlp_ae` như smoke/test-only model không? Nếu không, phải xử lý registry, config, tests và compliance snapshot cùng lúc.
4. Có coi SMD là dataset duy nhất của project scope không? Nếu có, AnomalyArchive nên được tách khỏi runtime chính trước khi xoá.

## 8. Đề xuất thứ tự triển khai sau research

Không xoá trực tiếp theo tên marker. Thứ tự an toàn là: chốt scope v2/v3 và checkpoint policy; tạo plan cho L01-L05; migrate config/tests; xoá three-stage branches; sau đó xử lý L06-L09; cuối cùng dọn test fixtures L10 và đánh giá lại L11. Baselines L12 và operational tooling L13 không nằm trong deletion batch này.

## 9. Sources

- `prompts/1_research_prompt.md`
- `documents/spec/full-spec-v2.md`
- `documents/spec/full-spec-v3.md`
- `documents/abstract-design-notes/design_starter.md`
- `codebase_preferences.md`
- `src/models/thesis_multitask_impl/*`
- `src/core/config.py`
- `src/core/config_model_validation.py`
- `src/data/stream.py`
- `src/engine/online_tta/*`
- `src/models/online_impl/online_adaptation.py`
- `tests/online/*`


---
date: 2026-07-11 22:54:13 +07:00
researcher: TheMetaSetter / Artificial Intelligence Agent
git_commit: fbfd011ac85e94d559201fd2153161e5523ff8af
branch: dev
repository: bachelor-thesis-2026
topic: "Code map and execution-flow notes for full-spec-v2 and full-spec-v3"
tags: [research, time-series, anomaly-detection, spec-audit, code-map]
status: complete
last_updated: 2026-07-11
last_updated_by: Artificial Intelligence Agent
---

# Research: Code map and execution-flow notes for full-spec-v2 and full-spec-v3

**Date**: 2026-07-11 22:54:13 +07:00  
**Researcher**: TheMetaSetter / Artificial Intelligence Agent  
**Git Commit**: `fbfd011ac85e94d559201fd2153161e5523ff8af`  
**Branch**: `dev`

## Research Question

Từ `full-spec-v2.md` và `full-spec-v3.md`, ghi chú các đoạn code liên quan trực tiếp đến spec, đồng thời mô tả kỹ các luồng xử lý, luồng tính toán và luồng thí nghiệm chính để dùng làm tài liệu chuẩn bị codebase cho thí nghiệm kế tiếp.

## Summary

Repository hiện tại đã có một xương sống khá rõ cho ba lớp chính: THESIS offline, THESIS online adaptation, và benchmark cho các baseline deep learning / traditional machine learning. So với v2, code đã bám sát contract hơn ở batch, output, checkpoint, threshold artifact, online runtime state, triage, verification buffer, và artifact integrity. So với v3, code đã chạm được một phần quan trọng của hướng stochastic retrieval thông qua `cosine_topk` và `gumbel_softmax`, nhưng chưa thấy một đường chạy Monte Carlo 10 mẫu được hiện thực thành luồng inference riêng với các summary variance giống như spec v3 mô tả. Điểm này là khoảng trống cần lưu ý khi chuẩn bị codebase cho vòng thí nghiệm sau.

Nhìn tổng thể, luồng THESIS hiện có thể đọc theo bốn tầng:

1. contract dữ liệu / output được khóa ở `src/core/contracts.py`;
2. offline THESIS xử lý encoder, hai nhánh prototype, fusion, loss, memory lifecycle trong `src/models/thesis_multitask*.py`;
3. online TTA dùng checkpoint offline, one-window causal stream, triage bốn vùng, verification buffer, projector-only update trong `src/models/online_adaptation.py` và `src/engine/online_tta/`;
4. benchmark wrappers kết nối THESIS với RedLamp, CANDI, M2N2, STUMPY, KMeansAD, và IForest trong `scripts/` và `src/baselines/`.

## Detailed Findings

### 1. Contract dữ liệu và output

Điểm bám spec trực tiếp nhất nằm ở `src/core/contracts.py:44-173`. File này khóa bốn hợp đồng chính:

- raw sequence có `x`, `point_labels`, `mask`, `timestamps`, `meta`;
- offline batch có tensor 3D `[B, L, D]` và metadata list;
- online batch dùng đúng cùng batch nền, nhưng không đòi `view_a/view_b`;
- model output top-level cố định gồm `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, `aux`.

Đoạn đáng chú ý nhất là `validate_online_batch()` và `validate_legacy_two_view_batch()` ở `src/core/contracts.py:112-129`. Luồng full-spec hiện tại đã tách rõ:

- full-spec online chỉ cần một window causal;
- hai view là contract lịch sử, chỉ còn dùng cho đường legacy.

Điều này khớp trực tiếp với tinh thần của v2 và càng quan trọng hơn trong v3, vì v3 nhấn mạnh online causal input đơn cửa sổ và label-free scoring.

### 2. THESIS offline: encoder, prototype branches, fusion, losses

File trung tâm là `src/models/thesis_multitask.py:1-8` cùng các mixin đi kèm.

Ở mức cấu hình, `src/models/thesis_multitask_components.py:29-257` định nghĩa:

- `MultitaskArchitectureConfig` cho `input_dim`, `window_size`, `hidden_dim`, `num_classes`;
- `PrototypeBranchConfig` cho continuous / discrete memory;
- `ScheduleAndWarmupConfig` cho temperature schedule và warm-up;
- `ObjectiveConfig` cho `lambda_recon`, `lambda_cls`, `lambda_contrastive`, score loss, và các regularizer khác;
- `ThreeStageRuntimeConfig` cho `training_phase`, `fusion_mode`, `discrete_query_mode`, `discrete_topk`.

Ở đây có hai dấu hiệu code bám spec khá rõ:

- `discrete_query_mode` có hai nhánh hợp lệ: `cosine_topk` và `gumbel_softmax` (`src/models/thesis_multitask_components.py:199-206`);
- `enable_score_loss` và `score_loss_type` đã có contract riêng cho point-wise balanced score loss (`src/models/thesis_multitask_components.py:104-147`).

Luồng khởi tạo model đi qua `src/models/thesis_multitask_setup_mixin.py:362-429` và `src/models/thesis_multitask_state_mixin.py:612-720`.

- `src/models/thesis_multitask_setup_mixin.py:368-429` dựng continuous prototype bank, discrete codebook, và đặt `discrete_assignment` theo phase / query mode.
- `src/models/thesis_multitask_state_mixin.py:612-720` khởi tạo memory từ token pool bằng k-means, sau đó gán codeword anomaly mask và radii Q99 cho verification metadata.

Phần này là cầu nối trực tiếp tới spec v3 ở chỗ:

- prototype geometry không phải chỉ là tensor ngẫu nhiên;
- discrete codebook có metadata an toàn cho lọc anomaly;
- memory initialization dùng token pool từ train split / synthetic train split, rồi đóng thành checkpoint state.

Luồng forward chính nằm ở `src/models/thesis_multitask_routing_mixin.py:471-655`.

Ở đó:

- encoder nhận batch window và trả `hidden` ở mức timestep;
- continuous branch đọc prototype bank bằng soft attention hoặc bypass theo phase;
- discrete branch dùng `cosine_topk` hoặc `gumbel_softmax`;
- fusion gộp hai nhánh rồi sinh `hidden_reconstruction` và `hidden_classification`;
- reconstruction head xuất `recon`;
- classification head trải phẳng `window_size * hidden_dim` rồi sinh `logits`;
- point score được tính bằng MSE theo timestep, window score là mean của point scores.

Đoạn core nhất là `src/models/thesis_multitask_routing_mixin.py:500-610`, nơi code quyết định cách memory được đọc và cách output contract được dựng. Đây là chỗ bám trực tiếp vào spec v2/v3 về:

- one encoder pass;
- two prototype branches;
- task-specific fusion;
- output top-level ổn định.

Luồng Stage A / Stage B được assemble ở `src/models/thesis_multitask_loss_mixin.py:771-950`.

- `training_step()` và `test_step()` đều gọi `_shared_step()` ở `src/models/thesis_multitask_loss_mixin.py:920-950`.
- `_shared_step()` dựng contrastive pair nếu phase yêu cầu, rồi tính `reconstruction_loss`, `classification_loss`, `score_loss`, và `contrastive_loss` (`src/models/thesis_multitask_loss_mixin.py:779-845`).
- Nếu `enable_score_loss` và đang ở Stage A, classification branch loss được ghép với score loss theo trung bình 1/2, đúng với contract O1 của v2 (`src/models/thesis_multitask_loss_mixin.py:818-827`).

Điều này nghĩa là code hiện tại đã có đủ chỗ để ghi nhận cả:

- Stage A multitask pretraining;
- Stage B fusion fine-tuning;
- point-score auxiliary loss;
- contrastive loss.

Nhưng với v3, phần stochastic retrieval vẫn chỉ thấy dưới dạng lựa chọn truy vấn và temperature, chưa thấy vòng Monte Carlo inference riêng biệt cùng sample-mean / sample-variance summary.

### 3. THESIS offline: luồng tính toán chính

Luồng tính toán của THESIS offline có thể viết ngắn gọn như sau:

```text
raw sequence
  -> validate_raw_sequence
  -> window / batch contract
  -> encoder hidden state
  -> continuous prototype lookup
  -> discrete prototype lookup
  -> task-specific fusion
  -> recon + logits
  -> point_scores + window_scores
  -> loss assembly
  -> checkpoint / evaluator
```

ASCII chi tiết hơn:

```text
[raw train/val/test sequence]
          |
          v
 [windowing + collation]
          |
          v
 [encoder -> hidden: [B, L, H]]
       /                 \
      v                   v
[continuous bank]   [discrete codebook]
      |                   |
      v                   v
[prototype_context]   [quantized_hidden]
       \                 /
        v               v
      [fusion / gate / concat projection]
                 |
        +--------+--------+
        |                 |
        v                 v
     [recon]          [logits]
        |                 |
        +--------+--------+
                 v
      [point_scores, window_scores]
                 |
                 v
       [Stage A / Stage B losses]
                 |
                 v
         [checkpoint / eval]
```

Phần này khớp với `src/models/thesis_multitask_routing_mixin.py:471-655`, `src/models/thesis_multitask_loss_mixin.py:771-950`, và hợp đồng output trong `src/core/contracts.py:132-149`.

### 4. Online THESIS: state, thresholding, triage, verification, projector-only update

Đầu mối online rõ nhất là `src/models/online_adaptation.py:29-259`.

Ở đây:

- `_resolve_reference_checkpoint_path()` ép online path quay về Stage-B checkpoint nếu cần;
- `ThesisMultitaskEncoderAdapter` deepcopy model offline rồi freeze toàn bộ tham số;
- `encode_source()` chỉ lấy một lần hidden từ frozen source encoder;
- `score_from_hidden()` tính lại reconstruction / classification từ hidden được chiếu qua projector;
- `latent_window_score` là khoảng cách cosine đến prototype gần nhất, lấy từ continuous prototype bank.

`NearIdentityMLPProjector` được thiết kế residual và gần identity theo mặc định (`src/models/online_adaptation.py:162-191`). Đây là đúng tinh thần projector-only adaptation của spec: đầu online step đầu tiên phải nằm sát latent space đã hiệu chỉnh, không nhảy quá xa.

Phần bảo vệ contract của model online còn có:

- `clean_stream_only=True` là guard cứng;
- `score_source="projected_hidden"` là guard cứng;
- reference encoder và online encoder đều load từ cùng checkpoint offline nhưng freeze geometry (`src/models/online_adaptation.py:194-259`).

Luồng online TTA chính nằm ở `src/engine/online_tta/online_engine.py:1000-1305`.

Trình tự thực tế là:

1. dựng online stream stride 1 từ clean validation / test sequence;
2. calibrate threshold artifact từ clean validation;
3. gom sequence test theo entity;
4. score từng window online;
5. tính EWMA point score;
6. triage bằng bốn vùng;
7. đẩy gray-zone vào verification buffer;
8. verification cycle kiểm tra prototype signature và token mask;
9. chỉ projector được update;
10. ghi checkpoint + artifact manifest + runtime state.

Luồng này bám vào các module con sau:

- `src/engine/online_tta/online_calibration.py:15-144` dựng stride-1 stream và thu clean validation scores;
- `src/engine/thresholding.py:30-43` chọn threshold clean-validation và EWMA;
- `src/engine/online_tta/triage.py:17-60` chia bốn vùng `normal`, `hard_old_normality`, `gray_zone`, `strong_anomaly`;
- `src/engine/online_tta/verification_buffer.py:7-85` giữ buffer có TTL và non-overlap;
- `src/engine/online_tta/verification_adapter.py:20-106` rebuild entry, lọc known anomaly token, và gắn PNN mask;
- `src/engine/online_tta/signature_verification.py:11-173` sinh `PrototypeVerificationMetadata`, nearest discrete codeword, ordered continuous signature, và recurrent signature set;
- `src/engine/online_tta/online_losses.py:7-116` định nghĩa masked PNN reconstruction, hard-old hinge loss, và token multi-positive InfoNCE;
- `src/engine/online_tta/runtime_state.py:9-124` serialize / resume state theo entity và online variant.

ASCII luồng online:

```text
[offline checkpoint + protocol thresholds]
                 |
                 v
        [stride-1 online stream]
                 |
                 v
     [frozen source encoder -> hidden]
                 |
                 v
        [projector / projected_hidden]
                 |
                 v
     [recon score + latent score + EWMA]
                 |
                 v
            [triage]
      /        |          \
     v         v           v
 [normal] [gray_zone] [strong_anomaly]
     |         |             |
     |         v             |
     |   [verification buffer]|
     |         |             |
     |         v             |
     |   [PNN / recurrent signature check]
     |         |             |
     |         v             |
     |   [A1/A2 projector-only update]
     |         |             |
     +---------+-------------+
                 |
                 v
        [checkpoint + manifest + state]
```

Trong code, chỉ `projector_params` được phép train trong online TTA. Guard đó nằm ở `src/engine/online_tta/online_engine.py:102-145`, `src/engine/online_tta/online_optimizer.py:19-69`, và được kiểm lại trong `src/engine/online_tta/online_engine.py:1102-1155`.

Đây là phần khớp mạnh nhất với v2 và cũng là phần chuẩn bị trực tiếp cho v3 ở cấp runtime: one-window, source-once, projector-only mutation, entity-scoped threshold artifact, và resume theo identity.

### 5. Luồng checkpoint, artifact integrity, và resume

`src/engine/checkpoint.py:12-149` là owner của save/load checkpoint chuẩn.

- payload luôn có `model_state_dict`, `scaler_state_dict`, `config`, `epoch`, `metric_history`;
- `extra_state` được giữ riêng;
- khi load, model được nạp state, rồi extra state được trả về cho owner của model.

THESIS offline benchmark và online benchmark đều dựa vào cấu trúc này:

- offline benchmark: `scripts/run_thesis_offline_benchmark.py:89-340`;
- online benchmark: `scripts/run_thesis_online_benchmark.py:91-152`;
- online runner chi tiết: `src/engine/online_tta/online_engine.py:1128-1305`.

Artifact integrity được tách riêng trong `src/core/artifact_integrity.py:11-62`. Ở đây manifest checksum là deterministic, có `identity` và `artifacts`, rồi verify lại digest từng file. Điều này rất quan trọng cho v2/v3 vì user yêu cầu run có thể resume và kết quả phải không bị lệch do artefact drift.

`src/protocols/threshold_artifact.py:8-96` định nghĩa threshold artifact versioned, gồm:

- offline point threshold;
- online EWMA threshold;
- input-window threshold;
- latent-window low/high thresholds;
- provenance `test_label_usage: metrics_only`.

Đây là phần nối rất trực tiếp với spec v2/v3 vì threshold không còn là một số đơn giản, mà là một object có provenance và score rule rõ ràng.

### 6. Benchmark flow: THESIS, deep baselines, traditional ML

Luồng offline benchmark chuẩn nằm ở `scripts/run_thesis_offline_benchmark.py:89-340`.

Nó làm đúng thứ tự:

- dựng data bundle;
- load checkpoint theo manifest;
- tính clean validation metrics;
- lấy threshold từ clean validation;
- đánh giá synthetic validation;
- đánh giá test;
- xuất score npz, metrics json, protocol json, thresholds json.

Luồng online benchmark chuẩn nằm ở `scripts/run_thesis_online_benchmark.py:91-152`.

- chọn variant `A0`, `A1`, hoặc `A2`;
- gọi `run_thesis_online_tta_experiment`;
- normalize records;
- viết report + integrity manifest.

Trong benchmark space, code hiện chia thành hai nhóm baseline chính:

1. deep learning / online streaming baselines:
   - `src/baselines/online/frozen.py:3-249` là frozen streaming baseline core;
   - `src/baselines/online/adaptive.py:3-314` là adaptive streaming baseline core.
2. traditional ML baselines:
   - `src/baselines/traditional/stumpy_channel_ab.py:3-258`;
   - `src/baselines/traditional/kmeans_ad.py:1-146`;
   - `src/baselines/traditional/iforest.py:1-142`.

Flow của các baseline này gần giống nhau ở mức benchmark contract:

```text
train reference
  -> clean validation calibration
  -> stride-1 or window-based scoring
  -> causal endpoint scores / point scores
  -> EWMA threshold
  -> online record + metric history
```

Điểm khác nhau chủ yếu là scorer:

- STUMPY dùng per-channel AB-join rồi aggregate channel max;
- KMeansAD dùng cluster distance;
- IForest dùng decision function đảo dấu;
- frozen/adaptive deep baselines thì chạy theo reference scorer và online policy riêng.

### 7. Code bám trực tiếp đến spec v3 nhưng còn thiếu một phần rõ ràng

Phần v3 hiện có dấu vết code tốt ở:

- `cosine_topk` discrete branch (`src/models/thesis_multitask_routing_mixin.py:120-149`, `src/models/thesis_multitask_setup_mixin.py:384-393`);
- `ordered_continuous_signature(topk=3)` (`src/engine/online_tta/signature_verification.py:90-99`);
- prototype verification metadata với anomaly mask / radii (`src/engine/online_tta/signature_verification.py:11-61`);
- exact four-region triage (`src/engine/online_tta/triage.py:17-41`);
- entity-scoped runtime resume (`src/engine/online_tta/runtime_state.py:9-124`);
- artifact integrity manifests (`src/core/artifact_integrity.py:23-52`).

Nhưng v3 mô tả một contract mạnh hơn nữa:

- stochastic inference mặc định;
- exactly ten Monte Carlo retrieval samples;
- mean / variance summaries ở level output.

Sau khi rà code hiện tại, em chưa thấy một luồng Monte Carlo 10-sample riêng biệt đi từ model đến evaluator / benchmark output. Nói cách khác, code đang có stochastic-capable components, nhưng chưa có full v3 MC aggregation path được hiện thực thành một contract end-to-end. Đây là thông tin quan trọng để lập kế hoạch chuẩn bị codebase cho lần thí nghiệm kế tiếp.

## Pipeline Documentation

```text
Spec v2/v3
  -> contract layers in src/core/contracts.py
  -> offline THESIS model in src/models/thesis_multitask*.py
  -> online adaptation in src/models/online_adaptation.py + src/engine/online_tta/
  -> evaluation / threshold / checkpoint / integrity in src/engine/*.py and src/protocols/*
  -> benchmark orchestration in scripts/run_thesis_offline_benchmark.py and scripts/run_thesis_online_benchmark.py
  -> baselines in src/baselines/*
```

```text
THESIS offline:
  raw sequence
    -> validate
    -> encode
    -> continuous branch
    -> discrete branch
    -> fusion
    -> recon + logits
    -> losses
    -> checkpoint / evaluator

THESIS online:
  offline checkpoint
    -> frozen source encoding
    -> projector
    -> score + EWMA
    -> triage
    -> verification
    -> projector-only update
    -> artifact manifest + runtime state
```

## Historical Context (from documents/)

`full-spec-v2.md` là contract experiment hiện tại cho hai-stage THESIS offline và A0/A1/A2 online TTA. `full-spec-v3.md` là bản siết contract mới hơn, nhấn mạnh stochastic retrieval, deterministic geometry, and stricter runtime ownership. Trong code hiện tại:

- v2 đã được hiện thực khá đầy đủ ở offline/online/benchmark layer;
- v3 đã có một phần nền ở query mode, prototype metadata, verification, và integrity;
- phần MC stochastic inference 10 sample chưa thấy là luồng chính thức.

Tài liệu nghiên cứu này vì vậy nên được dùng như một bản đồ code để chuẩn bị bước triển khai tiếp theo, thay vì coi là spec cuối cùng.

## Open Questions

1. `full-spec-v3.md` yêu cầu stochastic inference với `monte_carlo_samples: 10`. Hiện chưa thấy một đường chạy end-to-end trong code tạo ra sample stack, sample mean, và sample variance summaries cho output top-level.
2. Nên xác nhận xem experiment kế tiếp có cần giữ `cosine_topk` như đường mặc định hay chuyển sang `gumbel_softmax` cho discrete branch khi áp v3.
3. Cần quyết định phần nào của `aux` sẽ trở thành public contract cho benchmark / demo nếu full v3 MC output được bật.
4. Nếu chuẩn bị codebase cho thí nghiệm kế tiếp, nên kiểm tra lại hệ baseline matrix để bảo đảm deep baseline và traditional baseline đang được so sánh trên cùng protocol threshold / window size.

## Code References

- `src/core/contracts.py:44-173` - batch / online / output / evaluation contracts
- `src/models/thesis_multitask_components.py:29-257` - locked model config and runtime config
- `src/models/thesis_multitask_setup_mixin.py:362-429` - prototype memory and discrete assignment ownership
- `src/models/thesis_multitask_state_mixin.py:612-720` - memory initialization and anomaly verification metadata
- `src/models/thesis_multitask_routing_mixin.py:471-655` - offline forward path, point scores, window scores
- `src/models/thesis_multitask_loss_mixin.py:771-950` - Stage A / Stage B objective assembly
- `src/models/online_adaptation.py:29-259` - frozen source encoder and near-identity projector
- `src/engine/online_tta/online_engine.py:1000-1305` - full online flow, thresholding, verification, manifests, resume state
- `src/engine/online_tta/triage.py:17-60` - exact four-region triage and legacy baseline triage
- `src/engine/online_tta/verification_buffer.py:7-85` - non-overlap buffer and TTL lifecycle
- `src/engine/online_tta/verification_adapter.py:20-106` - PNN mask and recurrent signature verification
- `src/engine/online_tta/signature_verification.py:11-173` - prototype metadata, ordered signatures, recurrent signature logic
- `src/engine/online_tta/online_losses.py:7-116` - online A1/A2 losses
- `src/engine/online_tta/online_calibration.py:15-144` - stride-1 calibration stream and score collection
- `src/engine/online_tta/runtime_state.py:9-124` - resume state ownership
- `src/protocols/threshold_artifact.py:8-96` - threshold artifact schema and provenance
- `src/core/artifact_integrity.py:11-62` - manifest and checksum verification
- `src/engine/evaluator.py:202-460` - overlap reconstruction and metric computation
- `src/engine/checkpoint.py:12-149` - checkpoint save/load and extra state
- `src/baselines/online/frozen.py:3-249` - frozen online baselines
- `src/baselines/online/adaptive.py:3-314` - adaptive online baselines
- `src/baselines/traditional/stumpy_channel_ab.py:3-258` - STUMPY baseline
- `src/baselines/traditional/kmeans_ad.py:1-146` - KMeansAD baseline
- `src/baselines/traditional/iforest.py:1-142` - Isolation Forest baseline
- `scripts/run_thesis_offline_benchmark.py:89-340` - offline benchmark wrapper
- `scripts/run_thesis_online_benchmark.py:91-152` - online benchmark wrapper
- `tests/online/test_full_spec_online_contract.py:20-55` - one-window online contract and exact triage regions
- `tests/models/test_multitask_shapes.py:96-119` - cosine-topk discrete branch contract
- `tests/models/test_thesis_multitask_point_score_loss.py:77-88` - Stage A point-score loss contract
- `tests/online/test_online_prototype_metadata_contract.py:26-62` - prototype metadata roundtrip and anomaly mask contract


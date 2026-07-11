---
date: 2026-07-11 21:21:21 +07
researcher: Artificial Intelligence Agent
git_commit: fbfd011ac85e94d559201fd2153161e5523ff8af
branch: dev
repository: bachelor-thesis-2026
topic: "Main processing flow documentation for codebase onboarding"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-07-11
last_updated_by: Artificial Intelligence Agent
---

# Research: Main Processing Flow Documentation for Codebase Onboarding

**Date**: 2026-07-11 21:21:21 +07  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: `fbfd011ac85e94d559201fd2153161e5523ff8af`  
**Branch**: `dev`

## Mục tiêu đọc nhanh
Ý tưởng chính ở đây là: repo này chạy theo một dòng chảy khá rõ ràng.

`raw SMD sequence -> clean + normalize -> window -> batch -> model -> score -> metrics`

Trong luồng offline, model học trên các cửa sổ thời gian cố định. Trong luồng evaluation, điểm bất thường của từng window được ghép ngược về timeline gốc của từng entity. Trong luồng online, hệ thống lấy test stream theo thứ tự thời gian, chấm điểm, rồi chỉ cập nhật một nhóm tham số nhỏ ở projector.

## Bản đồ tổng quát

```text
(^_^)/  RAW SMD FILES
          |
          v
   +-------------------+
   | parser + split    |
   | train / val / test|
   +-------------------+
          |
          v
   +-------------------+
   | scaler            |
   | fit on train only |
   +-------------------+
          |
          v
   +-------------------+
   | window dataset    |
   | L = 20 in thesis  |
   +-------------------+
          |
          v
   +-------------------+
   | collate batch     |
   | [B, L, D]         |
   +-------------------+
          |
          v
   +-------------------+
   | model.forward     |
   | hidden / recon /  |
   | logits / scores   |
   +-------------------+
          |
          v
   +-------------------+
   | trainer / eval    |
   | checkpoint / log  |
   +-------------------+
```

## 1. Dữ liệu đi vào như thế nào

SMD được đọc từ ba thư mục `train`, `test`, `test_label` trong parser `SMDDatasetParser`. Parser này tạo ra một “raw sequence” cho từng entity, rồi tách `train` thành `train` và `val` bằng tỷ lệ `validation_split_ratio`. `test` giữ nguyên nhãn gốc. Xem [src/data/datasets/smd.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/Đ%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/datasets/smd.py#L15-L182).

Mỗi raw sequence có dạng:

```text
{
  x: Tensor[L, D],
  point_labels: Tensor[L] or None,
  mask: None,
  timestamps: None,
  meta: {dataset_name, entity_id, split, series_id, num_channels, sequence_length, ...}
}
```

Contracts ở tầng core bắt buộc batch chuẩn phải giữ các field `x`, `point_labels`, `mask`, `timestamps`, `meta`. Xem [src/core/contracts.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/core/contracts.py#L1-L166).

Ở cấu hình thesis đang active, `window_size = 20` và `stride = 1` trong benchmark protocol. Điều này được khóa cứng bởi validator của protocol. Xem [src/protocols/smd_benchmark_protocol.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/protocols/smd_benchmark_protocol.py#L27-L57) và config mẫu [configs/data/smd_benchmark_machine_3_9_window20.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/configs/data/smd_benchmark_machine_3_9_window20.yaml#L1-L11).

## 2. Window hóa và batch hóa

Sau khi parse xong, `SequenceStandardScaler` được fit chỉ trên split `train`, rồi transform lên cả ba split. Kế tiếp, `WindowDataset` chỉ lưu index của từng window, không copy toàn bộ dữ liệu trước. Cách này giữ code dễ đọc và tránh phình RAM không cần thiết. Xem [src/data/loaders.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/loaders.py#L150-L193) và [src/data/loaders.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/loaders.py#L231-L293).

`collate_windows` ghép nhiều window thành một batch:

```text
batch["x"] -> Tensor[B, L, D]
batch["point_labels"] -> Tensor[B, L] or None
batch["mask"] -> Tensor[B, L, D] or None
batch["timestamps"] -> Tensor[B, L] or None
batch["meta"] -> list[dict]
```

Xem [src/data/collate.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/collate.py#L1-L31).

ASCII nhìn nhanh:

```text
(^_^)  sequence
   |
   v
[slice 0:20]
[slice 1:21]
[slice 2:22]
   |
   v
collate_windows
   |
   v
Tensor[B, 20, D]
```

## 3. Synthetic anomaly injection chạy ở đâu

Luồng offline multitask có chèn synthetic anomaly. Tài liệu nguồn duy nhất của phần này là [src/data/augment.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/augment.py#L1-L170), nơi taxonomy RedLamp 11 lớp được liệt kê rõ:

`spike, flip, speedup, noise, cutoff, average, scale, wander, contextual, upsidedown, mixture`

Injector chọn một đoạn liên tục trong window, rồi sửa một số channel trong đoạn đó. Mỗi batch sau augmentation có thêm:

- `classification_labels`
- `synthetic_anomaly_mask`

Trong config task mẫu, `classification_label_mode = redlamp_multiclass` và `use_synthetic_augmentation = true`. Xem [configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml#L1-L14).

Nôm na thì: dữ liệu sạch vẫn là dữ liệu gốc, còn dữ liệu synthetic chỉ là bản window bị sửa có kiểm soát để model học thêm lớp phân loại bất thường.

## 4. Model offline chính đi theo flow nào

Model chính là `ThesisMultitaskModel`. File này là entrypoint public của model, còn logic được chia nhỏ bằng mixin để giữ runtime trong một model file duy nhất. Xem [src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/models/thesis_multitask.py#L1-L53).

Luồng `forward()` thực tế như sau:

```text
batch x [B, L, D]
   |
   v
encoder
   |
   v
hidden [B, L, H]
   |
   +--> continuous prototype branch
   |
   +--> discrete codebook branch
   |
   v
fusion per task
   |
   +--> reconstruction head -> recon [B, L, D]
   |
   +--> classification head  -> logits
   |
   v
point_scores, window_scores
```

Phần chính trong code nằm ở [src/models/thesis_multitask_routing_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/models/thesis_multitask_routing_mixin.py#L471-L655).

Điểm rất quan trọng là output top-level của model đã được khóa bởi contract:

```text
hidden, pooled, recon, logits, point_scores, window_scores, aux
```

Xem [src/core/contracts.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/core/contracts.py#L137-L151).

## 5. Training loss được ghép ở đâu

Train step, validation step, synthetic validation step, và test step đều gọi cùng một `_shared_step()`. Đây là chỗ model tự ghép các loss thành tổng loss. Xem [src/models/thesis_multitask_loss_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/models/thesis_multitask_loss_mixin.py#L771-L950).

Trình tự chính:

1. Nếu bật two-view contrastive, model tạo một cặp batch sạch và batch synthetic.
2. Model chạy `forward()` để lấy `reconstruction_loss`, `classification_loss`, các optional losses, và `score_loss`.
3. Các loss đó được cộng lại thành `total_loss`.
4. `training_step()` trả ra dict có `loss`, `log`, `outputs`, `loss_terms`, `batch`.

Trong config mẫu, `lambda_recon = 0.5`, `lambda_cls = 0.5`, `enable_two_view_contrastive = true`, `lambda_contrastive = 0.3`, còn các loss phụ như diversity / covariance / variance đều tắt. Xem [configs/model/thesis_multitask_two_stage_window20.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/configs/model/thesis_multitask_two_stage_window20.yaml#L1-L56).

ASCII ngắn:

```text
(^o^)  clean batch ----\
                         > loss assembly -> total_loss -> optimizer
(*^_^*) synthetic batch -/
```

## 6. Trainer và checkpoint chạy thế nào

Script offline training là điểm vào chuẩn: nó load config, register components, build dataset, build model, rồi đưa tất cả vào `Trainer`. Xem [scripts/train.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/scripts/train.py#L3-L220).

Trong `Trainer`, mỗi batch được chuyển lên device, gọi `model.training_step()`, backward, optimizer step, rồi logging / checkpoint. Checkpoint lưu cả `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, scaler state và extra state. Xem [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/engine/trainer.py#L534-L912) và [src/engine/checkpoint.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/engine/checkpoint.py#L1-L135).

## 7. Evaluation ghép window score về timeline gốc như thế nào

Script evaluate rebuild model từ config, load checkpoint, rồi chạy `Evaluator.evaluate()` trên test loader. Xem [scripts/evaluate.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/scripts/evaluate.py#L3-L220).

Evaluator không quan tâm model là gì, miễn là model trả ra `point_scores` và `window_scores`. Sau đó nó gom các window score chồng lấn về entity timeline gốc bằng cộng dồn và đếm số lần phủ. Chỗ này nằm ở [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/engine/evaluator.py#L122-L227) và [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/engine/evaluator.py#L335-L460).

ASCII dễ nhìn:

```text
(>_<)  windows with overlap
   |      |      |
   v      v      v
 score   score   score
   \      |      /
    v     v     v
 average back to full timeline
```

## 8. Online adaptation đi theo luồng nào

Luồng online là một nhánh riêng nhưng vẫn tái dùng cùng data contract và cùng encoder geometry. Script [scripts/run_online_adaptation.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/scripts/run_online_adaptation.py#L3-L220) tạo `SMDOnlineStream`, rồi `OnlineWindowBatcher` thêm hai view `view_a` và `view_b`. Xem [src/data/stream.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/stream.py#L1-L220).

Trong `OnlineLoop`, mỗi step làm đúng ba việc:

1. chạy model trước update để đo trạng thái hiện tại
2. gọi `model.training_step()` rồi update optimizer
3. chạy lại model sau update để đo thay đổi

Chỉ group tham số `target_param_group` được phép update, và ở slice này nó là `projector_params`. Xem [src/engine/online_loop.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/engine/online_loop.py#L14-L219).

Model online tương ứng là [src/models/online_adaptation.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/models/online_adaptation.py#L1-L260). Điểm dễ nhớ là:

- reference encoder bị frozen
- online encoder giữ cùng geometry offline
- projector là residual gần identity
- output vẫn có `recon`, `logits`, `point_scores`, `window_scores`

ASCII nhỏ:

```text
(^_^)  stream batch
   |
   +--> frozen reference encoder
   |
   +--> online encoder + projector
   |
   +--> alignment loss
   |
   v
  update projector only
```

## 9. Cấu hình nào đang khóa các quyết định chính

`configs/model/thesis_multitask_two_stage_window20.yaml` cho thấy model thesis hiện tại dùng:

- `window_size: 20`
- `encoder_family: cnn_simple`
- `continuous_num_prototypes: 32`
- `discrete_codebook_size: 60`
- `training_phase: stage_a_multitask_pretraining`
- `fusion_mode: task_specific_concat_projection`
- `discrete_query_mode: cosine_topk`

Xem [configs/model/thesis_multitask_two_stage_window20.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/configs/model/thesis_multitask_two_stage_window20.yaml#L1-L72).

Nhìn đơn giản: `data` nói window bao nhiêu, `model` phải khớp đúng window đó, còn `protocol` khóa cách đánh giá cho khỏi lệch benchmark. [src/core/config.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/core/config.py#L1-L168) có `_resolve_thesis_model_window_size()` để bắt model.window_size khớp data.window_size.

## 10. Tóm tắt ngắn cho người mới

Nếu em mới on-board, cứ nhớ 4 câu này:

1. SMD được đọc thành raw sequence theo entity.
2. Raw sequence được chuẩn hóa rồi cắt thành window cố định.
3. Model nhận batch `[B, L, D]`, trả về reconstruction, logits, và point scores.
4. Evaluation và online adaptation đều quay lại cùng một timeline entity, chỉ khác cách dùng score và cách update tham số.

Nếu muốn hình dung một dòng chạy ngắn gọn:

```text
SMD file
  -> parser
  -> scaler
  -> window dataset
  -> batch
  -> thesis model
  -> loss / score
  -> checkpoint / metrics
```

## Code References

- [src/data/datasets/smd.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/datasets/smd.py#L15-L182) - SMD parser
- [src/data/loaders.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/loaders.py#L150-L293) - scaling, windowing, loaders
- [src/data/augment.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Kho%C3%A1%20lu%E1%BA%ADn%20t%E1%BB%91t%20nghi%E1%BB%87p/bachelor-thesis-2026/src/data/augment.py#L1-L170) - synthetic anomaly injector
- [src/models/thesis_multitask.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/models/thesis_multitask.py#L1-L53) - public thesis model
- [src/models/thesis_multitask_routing_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/models/thesis_multitask_routing_mixin.py#L471-L655) - forward path
- [src/models/thesis_multitask_loss_mixin.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/models/thesis_multitask_loss_mixin.py#L771-L950) - loss assembly
- [src/engine/trainer.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/engine/trainer.py#L534-L912) - offline training loop
- [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/engine/evaluator.py#L122-L460) - window-to-timeline evaluation
- [src/engine/online_loop.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/engine/online_loop.py#L14-L219) - projector-only online update
- [src/models/online_adaptation.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90%C3%80I%20H%E1%BB%8CC%20QU%E1%BB%90C%20GIA%20TPHCM/%C4%90H%20KHOA%20H%E1%BB%8CC%20T%E1%BB%B0%20NHI%C3%8AN/Khoa%CC%81%20lua%CC%A3%CC%A3n%20to%CC%82%CC%81t%20nghi%C3%AA%CC%A3p/bachelor-thesis-2026/src/models/online_adaptation.py#L1-L260) - online adaptation model

## Pipeline Documentation

Luồng chính của codebase hiện tại có thể đọc như một pipeline 3 tầng:

1. `offline training` học representation và heads trên window cố định.
2. `offline evaluation` ghép point score về timeline gốc rồi tính metric.
3. `online adaptation` giữ encoder geometry cũ, chỉ cập nhật projector nhỏ trên stream sạch.

Điểm dễ hiểu nhất cho người mới là: `window` luôn là đơn vị làm việc cơ bản, còn `entity timeline` là đơn vị cuối cùng để báo cáo kết quả. Dữ liệu có thể đi qua nhiều bước, nhưng contract `[B, L, D]` và `point_scores`/`window_scores` là cái nối mọi thứ lại với nhau.

## Historical Context

Thiết kế trong repo đang bám sát các ghi chú ở `documents/abstract-design-notes/design_starter.md` và `documents/abstract-design-notes/idea.md`, nơi nhấn mạnh ba ý:

- giữ contract rõ ràng cho batch và output
- để one model one file
- giữ window length cố định trong thesis path

Những ghi chú này giải thích vì sao model multitask, trainer, evaluator, và online loop đều tách vai rất rõ, nhưng vẫn dùng chung vocabulary của batch và output.

## Open Questions

- Cấu hình tiện ích ở `src/data/api.py` vẫn có default `window_size = 100`, trong khi benchmark thesis hiện khóa `window_size = 20`. Khi đọc code cần nhìn đúng layer đang dùng.
- `OnlineLoop` hiện chỉ update `target_param_group`, nên người mới cần xem config task để biết chính xác group nào đang train.
- `training_phase` có nhiều chế độ lịch sử, nhưng file config thesis mẫu đang đi theo slice `stage_a_multitask_pretraining`.


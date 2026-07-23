# Offline Report Minimal Retention Policy

Date: 2026-07-21

## Mục tiêu

Giữ lại lượng artifact nhỏ nhất nhưng vẫn đủ để dựng lại hai bảng offline:

1. Bảng `VUS-PR`, `affiliation F1`, và `VUS-ROC`.
2. Bảng so sánh `mean of means` và `mean of variances` giữa các split.

Chính sách này áp dụng sau khi đã kiểm tra summary của từng run và đã tạo
`offline_report_data.json` trên máy local.

## Cách hiểu hai thống kê UQ

Với một split, một cửa sổ có `P` point và stochastic inference có `M` lần
forward pass. Gọi anomaly score của point `p` ở lần forward pass `m` là
`a[p, m]`.

Mean của point `p` là:

```text
point_mean[p] = mean_m(a[p, m])
```

`mean of means` là trung bình của các `point_mean[p]` trên các point trong
split. Trong artifact compact, giá trị tương ứng ở point-level là:

```text
splits.<split>.point_score_summary.mean
```

Variance của point `p` qua `M` lần forward pass là:

```text
point_variance[p] = variance_m(a[p, m])
```

`mean of variances` là trung bình của `point_variance[p]` trên các point.
Trong artifact compact, giá trị tương ứng là:

```text
splits.<split>.uncertainty_summary.point_anomaly_score_variance_mean
```

Nếu report cần window-level statistics thì dùng:

```text
splits.<split>.window_score_summary.mean
splits.<split>.uncertainty_summary.window_anomaly_score_variance_mean
```

## File bắt buộc giữ lại cho mỗi run

### Bảng metric chính

```text
two_stage/stage_b_fusion_finetuning/evaluation_metrics.json
```

Đây là nguồn canonical cho:

```text
vus_pr
affiliation_f1
vus_roc
```

### Bảng mean of means và mean of variances

```text
two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json
```

File này cần có các split:

```text
clean_validation
synthetic_validation
test
```

Để so sánh validation với testing ở point-level, tối thiểu phải có:

```text
splits.clean_validation.point_score_summary.mean
splits.clean_validation.uncertainty_summary.point_anomaly_score_variance_mean
splits.test.point_score_summary.mean
splits.test.uncertainty_summary.point_anomaly_score_variance_mean
```

Nên giữ thêm các field window-level nếu bảng report có phân tích theo window.

`uq_summary.run` cũng phải còn đầy đủ `variant_name`, `entity_id`, `seed`,
`stage_name`, và `checkpoint_sha256` để nhóm và truy xuất provenance của từng
combination.

## File nhỏ nên giữ để audit

Các file sau không bắt buộc để vẽ bảng, nhưng nên giữ vì chúng nhẹ và giúp
người đọc kiểm tra protocol:

```text
two_stage/stage_b_fusion_finetuning/thresholds/thresholds.json
two_stage/stage_b_fusion_finetuning/protocol/resolved_protocol.json
two_stage/stage_b_fusion_finetuning/retention/machine-*/offline/retention_bundle_manifest.json
two_stage/stage_b_fusion_finetuning/retention/machine-*/offline/retention_summary.json
```

`metrics/offline_metrics.json` nên được giữ tạm cho tới khi xử lý xong việc
đối chiếu với `evaluation_metrics.json`. Nếu đã chốt `evaluation_metrics.json`
là nguồn canonical và report bundle đã được kiểm tra, file này có thể xóa.

## Checkpoint cần giữ

Giữ checkpoint được dùng để tạo kết quả report, tối thiểu là:

```text
two_stage/stage_b_fusion_finetuning/checkpoints/best.pt
```

Nếu muốn audit đầy đủ quá trình huấn luyện thì giữ thêm:

```text
two_stage/stage_b_fusion_finetuning/checkpoints/final.pt
```

Không xóa checkpoint chỉ vì đã xóa raw trace. `checkpoint_sha256` trong
`uq_summary.json` và retention manifest phải khớp với checkpoint được giữ.

## Report bundle trên máy local

Sau khi thu thập dữ liệu từ các run remote và baseline local, tạo:

```text
outputs/reporting/offline_phase_tables/offline_report_data.json
```

Bundle này cần giữ row-level data cho mỗi run:

```text
run identity
VUS-PR / affiliation F1 / VUS-ROC
point_score_summary
window_score_summary
uncertainty_summary
trace-audit flags
```

Sau khi bundle đã được tạo và kiểm tra, nó có thể dùng để dựng lại hai bảng
mà không cần đọc raw trace.

## Bảng so sánh validation và testing

Có thể tính bảng này theo từng combination. Combination phải được xác định
bằng ít nhất:

```text
variant_name + entity_id + seed + stage_name
```

Bảng point-level đề xuất:

| Combination | Mean of means - clean validation | Mean of means - test | Mean of variances - clean validation | Mean of variances - test | Test minus validation |
|---|---:|---:|---:|---:|---:|
| O0 / machine_1_6 / seed6 | ... | ... | ... | ... | ... |

Trong đó:

```text
mean_of_means_validation
  = uq_summary.splits.clean_validation.point_score_summary.mean

mean_of_means_test
  = uq_summary.splits.test.point_score_summary.mean

mean_of_variances_validation
  = uq_summary.splits.clean_validation.uncertainty_summary.point_anomaly_score_variance_mean

mean_of_variances_test
  = uq_summary.splits.test.uncertainty_summary.point_anomaly_score_variance_mean

delta
  = mean_of_variances_test - mean_of_variances_validation
```

`clean_validation` nên là split validation chính để so sánh với `test`. Có thể
thêm `synthetic_validation` thành một block riêng, nhưng không nên trộn nó vào
clean validation vì hai split có mục đích khác nhau.

Nếu cần một dòng tổng hợp qua nhiều seed, trước hết tính các giá trị trên cho
từng run, sau đó lấy trung bình các run. Không gộp raw point của mọi run thành
một mảng duy nhất nếu mục tiêu là mỗi run có trọng số bằng nhau.

## Artifact có thể xóa sau khi kiểm tra

Sau khi tất cả required summary fields đều tồn tại và không phải `null`, có
`offline_report_data.json`, và checkpoint/provenance đã được kiểm tra, có thể
xóa:

```text
clean_validation_traces.json
synthetic_validation_traces.json
test_traces.json
```

Ở cả hai vị trí nếu tồn tại:

```text
traces/
retention/*/offline/
```

Các file sau cũng không bắt buộc để vẽ hai bảng:

```text
*_point_scores.npz
```

Tuy nhiên, các file này nhỏ nên có thể giữ lại để kiểm tra nhanh point-score.

## Điều kiện trước khi xóa raw trace

Chỉ xóa sau khi từng run đạt đủ các điều kiện:

1. `evaluation_metrics.json` có đủ `vus_pr`, `affiliation_f1`, `vus_roc`.
2. `uq_summary.json` có đủ summary của `clean_validation` và `test`.
3. Các field variance cần báo cáo không phải `null`.
4. `variant_name`, `entity_id`, `seed`, và `stage_name` không bị thiếu.
5. `offline_report_data.json` đã chứa row của run đó.
6. Checkpoint được giữ lại và checksum provenance đã được kiểm tra.
7. Nếu xóa file được nêu trong retention manifest, phải rebuild manifest theo
   retention policy mới, chẳng hạn `summary_only`.

Raw trace vẫn cần giữ nếu còn khả năng phải backfill UQ, tính lại metric,
kiểm tra từng point, hoặc thực hiện EDA chi tiết.

## Chiến lược làm việc với thư mục kết quả sâu trên remote

Không nên viết workflow dựa trên một absolute path dài cố định. Thay vào đó,
workflow nên dùng một root ổn định và các artifact marker:

```text
REMOTE_REPO=/root/bachelor-thesis-2026
EXPERIMENT_ROOT=$REMOTE_REPO/outputs/benchmark
```

Các marker chính để nhận diện một run là:

```text
evaluation_metrics.json
metrics/uq_summary.json
checkpoints/best.pt
two_stage/stage_b_fusion_finetuning/
```

Quy trình đề xuất:

1. Đọc lại `ssh-gpu.txt` trước mỗi phiên remote.
2. Từ `EXPERIMENT_ROOT`, dùng `find` chỉ để lập danh sách các
   `evaluation_metrics.json` và `metrics/uq_summary.json`.
3. Với mỗi metric file, suy ra `run_root` bằng anchor `two_stage`, không bằng
   số lượng `.parent` cố định.
4. Chuẩn hóa mỗi run thành một record gồm `variant_name`, `entity_id`, `seed`,
   `stage_name`, `run_root`, và các absolute path cần thiết.
5. Kiểm tra required artifacts và checksum trước khi tạo danh sách prune.
6. Xuất danh sách prune vào một manifest nhỏ; chạy dry-run và kiểm tra số file,
   tổng dung lượng, cùng các path tuyệt đối.
7. Chỉ sau khi dry-run đúng mới thao tác trên đúng các path trong manifest.

Không nên dùng lệnh kiểu `rm -rf outputs/...` hoặc glob rộng trên toàn bộ
`outputs`. Việc xóa phải nhận một danh sách file cụ thể được tạo từ manifest,
trong đó checkpoint và summary artifacts đã bị loại khỏi danh sách xóa.

Một run remote nên được nhìn dưới dạng logical record thay vì một chuỗi path
dài:

```text
(variant, entity, seed, stage)
        |
        +-- run_root
        +-- evaluation_metrics.json
        +-- metrics/uq_summary.json
        +-- checkpoints/best.pt
        +-- traces/*.json
        +-- retention/*/offline/*.json
```

Cách này cho phép cùng một workflow xử lý `O0/O1`, nhiều machine, nhiều seed,
và cả các run có độ sâu thư mục khác nhau mà không cần sửa tay từng path.

## Chiến lược xử lý schema phân mảnh giữa các method

Không nên giả định mọi method lưu kết quả cùng tên file hoặc cùng cấu trúc
JSON. Mỗi method có thể có một schema adapter riêng, nhưng tất cả adapter phải
trả về cùng một canonical row trước khi tạo bảng.

### 1. Tách ba lớp dữ liệu

```text
raw method artifact
        |
        v
method-specific adapter
        |
        v
canonical run record
        |
        v
report table / pruning manifest
```

`raw method artifact` có thể là `evaluation_metrics.json`,
`offline_metrics.json`, `metrics.json`, hoặc một file JSON nằm trong thư mục
method riêng. Adapter đọc đúng schema gốc; code aggregate không đọc trực tiếp
từng schema method.

### 2. Canonical run record

Mỗi method sau khi chuẩn hóa phải tạo được record có tối thiểu:

```text
run_id
method_name
variant_name
entity_id
seed
stage_name
metric_source_path
uq_source_path
checkpoint_path
protocol_source_path
metrics.vus_pr
metrics.affiliation_f1
metrics.vus_roc
splits.clean_validation
splits.synthetic_validation
splits.test
provenance
```

Mỗi split nên có các field canonical:

```text
point_score_mean
window_score_mean
point_anomaly_score_variance_mean
window_anomaly_score_variance_mean
```

Field không tồn tại trong method đó phải được ghi rõ là `null` kèm lý do,
không được tự suy đoán hoặc thay thế bằng metric gần giống.

### 3. Mapping metric phải khai báo rõ

Ví dụ mapping có thể là:

```text
vus_pr          <- vus_pr hoặc pr_auc
affiliation_f1  <- affiliation_f1
vus_roc         <- vus_roc hoặc roc_auc
```

Mỗi fallback phải được ghi vào metadata:

```text
metric_source_key
metric_mapping_rule
metric_semantic_status
```

`pr_auc` chỉ được map thành `vus_pr` khi protocol xác nhận đây là cùng một
định nghĩa. Nếu không xác nhận được thì giữ ở field riêng như `pr_auc`, không
được gộp vào cột `vus_pr`.

### 4. Mapping split phải giữ nguyên tên canonical

Các tên như `val`, `validation`, `clean_val`, `test_eval`, hoặc `evaluation`
chỉ được map về tên canonical khi có bằng chứng từ protocol:

```text
clean_val / validation_clean  -> clean_validation
synthetic_val                -> synthetic_validation
test / evaluation_test       -> test
```

Nếu `validation` không biết là clean hay synthetic thì giữ
`validation_unclassified`, không tự động đưa vào `clean_validation`.

### 5. Tách thiếu dữ liệu khỏi lỗi schema

Canonical record nên phân biệt:

```text
available       = có field và đã đọc được
missing         = method không sinh field đó
invalid         = field có nhưng sai kiểu hoặc sai schema
not_comparable  = có số nhưng protocol khác
```

Chỉ record có `available` và `comparable` mới được đưa vào aggregate. Các
record còn lại vẫn giữ trong audit manifest để giải thích coverage gap.

### 6. Manifest lưu cả nguồn và kết quả chuẩn hóa

Một manifest compact nên có:

```text
run_id
method_name
source_files
canonical_fields_present
missing_fields
mapping_rules
protocol_status
comparability_status
checkpoint_sha256
```

Nhờ vậy, sau khi prune raw artifact vẫn biết mỗi cột report được lấy từ file
nào và bằng rule nào. Không nên xóa source artifact trước khi manifest
canonical đã được tạo và kiểm tra.

### 7. Quy trình xử lý

```text
discover files
    -> classify method/schema
    -> load with adapter
    -> validate types and protocol
    -> normalize to canonical rows
    -> compare coverage and provenance
    -> write report bundle
    -> generate prune manifest
    -> prune only validated raw files
```

Nguyên tắc quan trọng là aggregate sau normalization, không aggregate trong
lúc đang dò path. Điều này tránh việc một method dùng `f1`, method khác dùng
`affiliation_f1`, còn method thứ ba dùng `pr_auc`, rồi vô tình đặt chúng vào
cùng một cột dù ý nghĩa protocol khác nhau.

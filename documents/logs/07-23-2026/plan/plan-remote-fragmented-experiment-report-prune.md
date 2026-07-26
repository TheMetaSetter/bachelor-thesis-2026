# Kế hoạch audit và prune output thực nghiệm remote

Date: 2026-07-23

## Mục tiêu

Xử lý tuần tự cây kết quả remote có path sâu và schema phân mảnh giữa các method, tạo đủ dữ liệu cho hai bảng offline:

1. `VUS-PR`, `affiliation F1`, `VUS-ROC`.
2. `mean of means` và `mean of variances` giữa validation và testing.

Sau khi summary và provenance được kiểm tra, xóa raw traces và artifact trùng lặp không cần thiết, nhưng giữ checkpoint audit của từng stage.

## Hợp đồng summary tối thiểu

Với point `p` và `M` stochastic forward passes, gọi anomaly score là `a[p, m]`:

```text
point_mean[p] = mean_m(a[p, m])
point_variance[p] = variance_m(a[p, m])
```

`mean of means` là trung bình của `point_mean[p]` trên các point. `mean of variances` là trung bình của `point_variance[p]` trên các point. Các field compact tương ứng là:

```text
splits.<split>.point_score_summary.mean
splits.<split>.uncertainty_summary.point_anomaly_score_variance_mean
```

Nếu cần window-level report, giữ thêm:

```text
splits.<split>.window_score_summary.mean
splits.<split>.uncertainty_summary.window_anomaly_score_variance_mean
```

## Nguyên tắc cố định

Các nguyên tắc này áp dụng cho mọi phiên kiểm tra, chuẩn hóa và prune dữ liệu thực nghiệm trên remote. Mục tiêu là tìm đúng combination, giữ lại đúng bằng chứng cần cho report và audit, rồi mới cân nhắc giải phóng dung lượng:

```text
đọc thông tin remote
        ↓
discover artifact marker + logical identity
        ↓
normalize thành canonical run record
        ↓
validate metric + protocol + provenance
        ↓
manifest → dry-run → prune chính xác → kiểm tra hậu quả

( =^･ω･^= )  chỉ đi sang bước sau khi bước trước đã có bằng chứng
```

1. **Xác nhận remote trước khi kết nối.** Đọc lại `ssh-gpu.txt` trước mỗi phiên remote để lấy đúng endpoint, repository root và thông tin xác thực. Không in password ra terminal hoặc đưa password vào log.

2. **Giữ remote inspection ở chế độ read-only.** Khi chỉ kiểm tra dữ liệu, chỉ được đọc file, liệt kê path, tính checksum và kiểm tra process. Không được chỉnh sửa, di chuyển hoặc xóa artifact. Chỉ chuyển sang thao tác ghi khi phase hiện tại đã được cho phép rõ ràng.

3. **Xử lý tuần tự từng method.** Với một method, hoàn tất discovery, phân loại schema, tạo adapter, validate và ghi manifest rồi mới chuyển sang method tiếp theo. Cách này giữ lỗi trong phạm vi một method, giúp biết chính xác lỗi bắt đầu từ schema hoặc artifact nào.

4. **Coi remote result tree là cây kết quả bị phân mảnh (fragmented).** Không giả định các method có cùng độ sâu path, cùng tên thư mục hoặc cùng schema. Một artifact chỉ được chọn sau khi đã đối chiếu path, metadata và stage mà nó thuộc về.

5. **Discovery bằng artifact marker và logical identity.** Tìm lần chạy dựa trên artifact marker, tức là file hoặc thư mục đặc trưng cho kết quả của một lần chạy, chẳng hạn `evaluation_metrics.json` hoặc `metrics/uq_summary.json`. Đồng thời xác định logical identity, tức là các thông tin nhận diện combination như method, variant, entity, seed, phase và stage. Không suy ra `run_root` bằng số lượng `.parent` cố định.

6. **Chỉ aggregate sau khi normalize.** Trước hết chuyển schema của từng method thành canonical run record, tức là bản ghi có cấu trúc chung để các kết quả khác đối chiếu. Record phải được kiểm tra metric, split, protocol, checkpoint và provenance. Chỉ record hợp lệ và comparable mới được đưa vào aggregate.

7. **Tối thiểu hóa dữ liệu lưu xuống ổ đĩa.** Tính intermediate values trực tiếp trong lúc chạy (on-the-fly) khi có thể. Chỉ lưu summary statistics, tức các số liệu tổng hợp cần cho report; provenance cho biết kết quả đến từ config/checkpoint/protocol nào; cùng diagnostics được chọn để phát hiện lỗi. Không lưu toàn bộ output của từng forward pass mặc định.

8. **Giữ đủ checkpoint cho audit.** Mỗi stage bắt buộc giữ checkpoint khởi tạo và checkpoint tốt nhất. Record phải ghi rõ role, đường dẫn tuyệt đối (absolute path) và checksum của từng checkpoint để nhà nghiên cứu sau này có thể kiểm tra lại.

9. **Prune chỉ sau khi qua đủ cổng kiểm tra an toàn (safety gate).** Trước khi xóa, phải tạo prune manifest, tức là tệp liệt kê chính xác từng path sẽ bị xóa và lý do xóa. Sau đó chạy dry-run, tức là chạy thử mà chưa thay đổi dữ liệu, rồi kiểm tra số file, tổng dung lượng và đường dẫn tuyệt đối. Chỉ prune đúng các path đã được manifest phê duyệt; cuối cùng kiểm tra lại artifact giữ lại, checksum và dung lượng sau thao tác.

## Phạm vi và hierarchy

Experiment root được xác định theo loại experiment:

```text
/root/bachelor-thesis-2026/outputs/benchmark/
/root/bachelor-thesis-2026/outputs/benchmark_smoke/
```

Canonical hierarchy mục tiêu:

```text
<experiment_type>/<dataset_name>/<entity_name>/<seed_value>/<method_name>/<phase_name>/<stage_name>/
```

Remote tree lịch sử có thể dùng thứ tự khác như `smd`, `two_stage`, hoặc `stage_b_fusion_finetuning`; các tree này phải được xử lý qua discovery và adapter, không được xem là canonical hierarchy mới.

## Phase 0 — Chốt phạm vi và môi trường

### Stage 0.1 — Xác nhận endpoint

Đọc `ssh-gpu.txt`, xác nhận repository remote và ghi thời điểm kiểm tra.

Trước khi tin một host key mới, phải lấy fingerprint chính thức từ provider, console quản trị hoặc quản trị viên qua một kênh độc lập với phiên SSH. So sánh fingerprint đó với fingerprint mà remote cung cấp qua `ssh-keyscan` hoặc cảnh báo SSH. Chỉ thêm host key vào `known_hosts` khi hai fingerprint khớp; không tự động xóa entry cũ và tin key mới chỉ vì endpoint vẫn dùng cùng IP/port.

CLI tối thiểu cho quy trình này, với `<host>` và `<port>` thay bằng endpoint thật:

```bash
# 1. Lấy fingerprint key mà endpoint hiện đang cung cấp qua mạng.
ssh-keyscan -p <port> -t ed25519 <host> 2>/dev/null \
  | ssh-keygen -lf - -E sha256

# 2. Xem các entry hiện có trong known_hosts cho đúng endpoint.
ssh-keygen -F '[<host>]:<port>'

# 3. Chỉ sau khi fingerprint khớp nguồn độc lập mới thay entry cũ.
ssh-keygen -R '[<host>]:<port>'
ssh-keyscan -p <port> -t ed25519 <host> >> ~/.ssh/known_hosts

# 4. Kiểm tra kết nối sau khi cập nhật; đây chưa phải lệnh thay đổi remote.
ssh -p <port> <user>@<host> 'hostname; pwd'
```

Không chạy bước 3 chỉ vì `ssh-keyscan` trả về một key mới; bước đó cần được đối chiếu trước với fingerprint từ provider/console hoặc quản trị viên.

### Stage 0.2 — Xác nhận experiment root

Kiểm tra `outputs/benchmark` và các root liên quan tồn tại; xác nhận không có job đang ghi vào target tree.

### Gate 0

Endpoint, repository root và experiment root đã rõ; host-key verification qua kênh độc lập là điều kiện bắt buộc trước khi tin key mới; chưa có write operation.

### Kết quả thực hiện Phase 0 — 2026-07-23

- SSH endpoint: `root@159.48.242.1:20714`.
- Remote host: `intriguing-meerkat`.
- Repository: `/root/bachelor-thesis-2026` tồn tại.
- `EXPERIMENT_ROOT=/root/bachelor-thesis-2026/outputs/benchmark` tồn tại.
- Các namespace cấp đầu đã thấy: `online`, `online_streaming`, `smd`.
- `/root/bachelor-thesis-2026/outputs/benchmark_smoke` hiện không tồn tại.
- Không thấy process liên quan đến repository, benchmark, training hoặc evaluation đang chạy tại thời điểm kiểm tra.
- Thao tác đã thực hiện: read-only; chưa tạo manifest, chưa sửa và chưa xóa artifact remote.
- Lưu ý bảo mật: fingerprint remote đã được lấy và kiểm tra kỹ thuật; chưa có xác nhận độc lập từ provider/console được ghi nhận trong phiên này.

### Kết quả thực hiện lại Phase 0 — 2026-07-24

- Thời điểm kiểm tra: `2026-07-24T15:06:38+0700`.
- SSH endpoint hiện tại: `root@159.48.242.1:20717`; remote host: `undaunted-deer`.
- Fingerprint từ `/etc/ssh/ssh_host_ed25519_key.pub` trên remote: `SHA256:tns8Pb6vV+uPja/lqRFl9v6y6wQ4Nn8G7CUFxJnwyqI`.
- Fingerprint từ `ssh-keyscan` trên local: `SHA256:tns8Pb6vV+uPja/lqRFl9v6y6wQ4Nn8G7CUFxJnwyqI`.
- Fingerprint của entry `ssh-ed25519` tương ứng trong local `known_hosts` cũng khớp; kết nối read-only thành công với `SSH_EXIT=0`.
- Repository `/root/bachelor-thesis-2026` và `EXPERIMENT_ROOT=/root/bachelor-thesis-2026/outputs/benchmark` tồn tại; các namespace cấp đầu là `online`, `online_streaming`, `smd`.
- `/root/bachelor-thesis-2026/outputs/benchmark_smoke` không tồn tại; không thấy process liên quan đến `benchmark`, `training` hoặc `evaluation` đang chạy.
- Thao tác chỉ đọc; chưa tạo manifest, chưa sửa và chưa xóa artifact remote.
- Gate 0 về kỹ thuật đã đạt; vẫn cần fingerprint từ provider/console hoặc quản trị viên qua kênh độc lập nếu muốn hoàn tất điều kiện xác nhận host key theo chính sách bảo mật.

### Kết quả bắt đầu lại Phase 0 — 2026-07-25

`ssh-gpu.txt` hiện trỏ tới `root@159.48.242.1:20714`. Fingerprint `ed25519` nhận được qua `ssh-keyscan` là `SHA256:MKyi8oDazDpvAtBxsUfmqE+xsCh5BEsv/+1HEXMGrY0`, nhưng endpoint này chưa có entry tương ứng trong local `known_hosts`. Vì chưa có fingerprint độc lập từ provider, console quản trị hoặc quản trị viên để đối chiếu, phiên được dừng trước khi SSH vào remote. Phase 1–5 chưa thực hiện; chưa sửa hoặc xóa artifact remote.

### Kết quả tiếp tục Phase 0 — 2026-07-25

Fingerprint do anh kiểm tra trực tiếp trên remote là `SHA256:MKyi8oDazDpvAtBxsUfmqE+xsCh5BEsv/+1HEXMGrY0`, khớp fingerprint từ `ssh-keyscan` trên local. Entry cũ của remote ở port `20717` được xóa khỏi `known_hosts`; key `ed25519` của endpoint `159.48.242.1:20714` được thêm và xác minh lại trên local. Kết nối read-only thành công tới host `terrific-chimpanzee`; repository và `EXPERIMENT_ROOT` tồn tại, các namespace là `online`, `online_streaming`, `smd`, `benchmark_smoke` không tồn tại và không có process benchmark/training/evaluation đang chạy. Gate 0 đạt; chưa sửa hoặc xóa artifact remote.

## Phase 1 — Discovery read-only

### Stage 1.1 — Lập artifact inventory

Từ `EXPERIMENT_ROOT`, tìm các marker:

```text
evaluation_metrics.json
metrics/uq_summary.json
checkpoints/best.pt
checkpoint khởi tạo
thresholds.json
resolved_protocol.json
retention_bundle_manifest.json
```

### Stage 1.2 — Tạo candidate records

Ghi `metric_path`, `uq_path`, checkpoint paths, protocol path, manifest path và parent directories; chưa kết luận identity chỉ bằng depth hoặc filename.

### Stage 1.3 — Phân loại tree

Đánh dấu candidate là canonical, historical two-stage, baseline-specific hoặc unknown hierarchy.

### Gate 1

Inventory read-only hoàn tất và chưa sửa/xóa remote artifact.

### Kết quả thực hiện Stage 1 — 2026-07-23

Inventory được lập read-only từ: `/root/bachelor-thesis-2026/outputs/benchmark/`.

Các nhóm dưới `smd` và marker chính:

| Nhóm | `evaluation_metrics.json` | `uq_summary.json` | `offline_metrics.json` | `best.pt` | `stage_b_init.pt` | `resolved_protocol.json` |
|---|---:|---:|---:|---:|---:|---:|
| `offline_benchmark` | 0 | 0 | 27 | 0 | 0 | 0 |
| `redlamp_baseline` | 9 | 0 | 0 | 9 | 0 | 0 |
| `thesis` | 18 | 73 | 72 | 36 | 18 | 72 |
| Toàn bộ `benchmark` | 27 | 73 | 99 | 45 | 18 | 72 |

Các marker khác trên toàn bộ experiment root: `thresholds.json` 63 file, `retention_bundle_manifest.json` 36 file và `retention_summary.json` 36 file. Ba loại raw trace đều xuất hiện 72 bản sao: `test_traces.json`, `synthetic_validation_traces.json` và `clean_validation_traces.json`. Tổng dung lượng tương ứng được remote báo theo thứ tự khoảng 5.28 GB, 1.06 GB và 1.06 GB.

Candidate path mẫu đã xác nhận:

```text
/root/bachelor-thesis-2026/outputs/benchmark/smd/redlamp_baseline/.../evaluation_metrics.json
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/.../two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/.../two_stage/initializations/stage_b_init.pt
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/.../two_stage/stage_b_fusion_finetuning/checkpoints/best.pt
```

Inventory cho thấy cần adapter theo nhóm: `offline_benchmark` chưa có hai marker canonical chính; `redlamp_baseline` có metric và checkpoint nhưng không có UQ summary; `thesis` có UQ summary và provenance phong phú nhưng có 73 UQ files cho 72 trace groups, nên phải kiểm tra identity/duplicate ở Stage 2 trước khi tạo canonical records. Không dùng số lượng file này để kết luận report coverage hoặc đánh dấu prune.

Thao tác Stage 1 chỉ đọc; chưa tạo prune manifest, chưa sửa và chưa xóa remote artifact.

### Kết quả thực hiện lại Stage 1 — 2026-07-24

Inventory mới được lập read-only từ `/root/bachelor-thesis-2026/outputs/benchmark` qua endpoint `root@159.48.242.1:20717` trên remote host `undaunted-deer`.

| Marker | Số file |
|---|---:|
| `evaluation_metrics.json` | 27 |
| `offline_metrics.json` | 99 |
| `metrics/uq_summary.json` | 36 |
| `best.pt` | 45 |
| `stage_b_init.pt` | 18 |
| `thresholds.json` | 63 |
| `resolved_protocol.json` | 72 |
| `retention_bundle_manifest.json` | 36 |
| `retention_summary.json` | 36 |
| `test_traces.json` | 72, khoảng 5.0 GB |
| `synthetic_validation_traces.json` | 72, khoảng 1011 MB |
| `clean_validation_traces.json` | 72, khoảng 1011 MB |

Phân loại candidate theo tree: `offline_benchmark` có 27 `offline_metrics.json` nhưng không có `evaluation_metrics.json`, UQ summary, checkpoint hoặc protocol; đây là nhóm `metric_only` và không cần UQ cho hai bảng report hiện tại. `redlamp_baseline` có 9 `evaluation_metrics.json` và 9 `best.pt`, nhưng không có UQ summary hoặc protocol; đây là nhóm `baseline-specific` và chưa đủ provenance để gộp vào canonical report. `thesis` có 18 Stage B candidates; mỗi candidate có đủ `evaluation_metrics.json`, `metrics/uq_summary.json` của cùng Stage B, `checkpoints/best.pt`, `initializations/stage_b_init.pt` và `protocol/resolved_protocol.json` theo path tương ứng; đây là nhóm `historical two-stage` cần tiếp tục kiểm tra identity, schema và comparability ở Phase 2.

Tổng số `uq_summary.json` là 73, nhưng chỉ 18 file nằm đúng tại `two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json`. 18 file khác nằm ở `thesis/.../metrics/uq_summary.json`, còn 37 file nằm dưới `retention/.../offline/uq_summary.json`, trong đó có thể có bản sao hoặc path identity anomaly. Vì vậy, Stage 1 không dùng tổng số 73 để kết luận số combination hoặc report coverage.

Candidate path chuẩn hóa để chuyển sang Stage 2:

```text
/root/bachelor-thesis-2026/outputs/benchmark/smd/redlamp_baseline/.../evaluation_metrics.json
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O{0,1}/machine_*/seed*/two_stage/stage_b_fusion_finetuning/evaluation_metrics.json
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O{0,1}/machine_*/seed*/two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O{0,1}/machine_*/seed*/two_stage/stage_b_fusion_finetuning/checkpoints/best.pt
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O{0,1}/machine_*/seed*/two_stage/initializations/stage_b_init.pt
/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O{0,1}/machine_*/seed*/two_stage/stage_b_fusion_finetuning/protocol/resolved_protocol.json
```

Các tree cấp đầu `online` và `online_streaming` được ghi nhận nhưng không đưa vào inventory offline report hiện tại. Gate 1 đạt ở mức inventory: candidate paths đã được phát hiện và phân loại read-only; chưa tạo prune manifest, chưa sửa và chưa xóa remote artifact. Trước khi aggregate hoặc prune, Phase 2 vẫn phải xử lý tuần tự từng method và xác minh identity/provenance của từng candidate.

### Kết quả thực hiện lại Stage 1 — 2026-07-25

Inventory được lập tuần tự theo từng method dưới `/root/bachelor-thesis-2026/outputs/benchmark/smd`. `offline_benchmark` có 27 `offline_metrics.json` và không có `evaluation_metrics.json` hoặc UQ summary. `redlamp_baseline` có 9 `evaluation_metrics.json` và 9 `best.pt`, nhưng không có UQ summary hoặc protocol. `thesis` có 18 Stage B `evaluation_metrics.json`, 18 UQ summary đúng vị trí Stage B, 18 Stage B initialization checkpoint, 18 Stage B protocol và các checkpoint liên quan. Toàn cây có 73 UQ summary, 72 protocol, 63 threshold, 36 retention manifest, 36 retention summary và 72 bản sao của mỗi loại raw trace.

Chưa dùng tổng số UQ summary để suy ra số combination; chỉ 18 UQ summary đúng anchor `two_stage/stage_b_fusion_finetuning` được bind vào Stage B. Gate 1 đạt ở mức discovery; remote chỉ được đọc.

## Phase 2 — Xử lý tuần tự từng method

Mỗi method phải hoàn tất toàn bộ stage 2.1–2.5 trước khi chuyển method kế tiếp.

### Stage 2.1 — Xác định method boundary

Phân biệt experiment type, dataset namespace, method, variant, entity, seed, phase và stage. Không gọi method directory là experiment root.

### Stage 2.2 — Chọn schema adapter

Adapter đọc schema gốc như `evaluation_metrics.json`, `offline_metrics.json`, `metrics.json` hoặc method-specific report JSON; mọi fallback phải ghi source key và mapping rule. Luồng chuẩn hóa có ba lớp:

```text
raw method artifact -> method-specific adapter -> canonical run record
```

Không để code aggregate đọc trực tiếp schema riêng của từng method.

### Stage 2.3 — Trích xuất identity

Chuẩn hóa `experiment_type`, `dataset_name`, `entity_name`, `seed_value`, `method_name`, `variant_name`, `phase_name`, và `stage_name`. Identity không rõ phải ghi `unknown` hoặc `missing`, không đoán im lặng.

### Stage 2.4 — Trích xuất report fields

Lấy `vus_pr`, `affiliation_f1`, `vus_roc` và các split `clean_validation`, `synthetic_validation`, `test` nếu có. Với mỗi split, ưu tiên `point_score_mean`, `window_score_mean`, `point_anomaly_score_variance_mean`, và `window_anomaly_score_variance_mean`.

### Stage 2.5 — Kết luận method

Ghi trạng thái `ready`, `partial`, `invalid`, `not_comparable`, hoặc `blocked`. Nếu lỗi xảy ra, dừng ở method đó để chẩn đoán; không để lỗi lan sang method khác.

### Gate 2

Method có canonical records, source paths, mapping rules, missing-field report và comparability status.

### Bối cảnh cần biết trước khi đọc kết quả

#### Document này viết về gì?

Document này viết về **artifact kết quả thí nghiệm trên remote**. Nó không mô tả kiến trúc neural network; nó mô tả cách kiểm tra, ghép đúng và giữ lại các file kết quả để dựng report an toàn.

Các đối tượng liên quan là:

- experiment root `/root/bachelor-thesis-2026/outputs/benchmark/`;
- các nhóm `offline_benchmark`, `redlamp_baseline`, và `thesis`;
- metric, UQ summary, checkpoint, protocol, raw trace và retention copy;
- workflow từ inventory đến report bundle và prune manifest.

Một combination là một lần chạy cụ thể, được xác định bởi method, variant, entity, seed, phase và stage. Ví dụ:

```text
thesis / O1 / machine_1_6 / seed6 / two_stage / stage_b_fusion_finetuning
```

#### Nó nằm ở đâu trong luồng tổng quan của codebase?

Benchmark code tạo artifact vào `outputs/`. Các script report đọc summary từ đó để tạo dữ liệu report local:

```text
benchmark code → outputs/benchmark/.../stage/
       ↓ tìm và kiểm tra
canonical run record
       ↓
offline_report_data.json
       ↓
hai bảng report

( =^･ω･^= )  file → identity → protocol → summary → report
                         phải cùng một combination
```

Phase 2 là bước kiểm tra file đang có nghĩa gì, thuộc combination nào, và có được phép ghép với file khác hay không. Phase 2 không sửa model và không xóa remote artifact.

#### Hai bảng report là gì?

Bảng 1 gồm:

```text
VUS-PR, affiliation F1, VUS-ROC
```

Nguồn ưu tiên là `evaluation_metrics.json`.

Bảng 2 gồm hai summary UQ. Với mỗi point, model có nhiều anomaly score từ stochastic forward pass (với 10 pass cho một point):

```text
point mean     = trung bình các score của point
point variance = variance của các score của point

mean of means     = trung bình các point mean
mean of variances = trung bình các point variance
```

Nguồn ưu tiên là `metrics/uq_summary.json`, với ba split riêng:

```text
clean_validation | synthetic_validation | test
```

`clean_validation` thường được so sánh với `test`; `synthetic_validation` phải báo cáo riêng.

#### Nếu không giải quyết vấn đề này thì sao?

Nếu chỉ nhìn tên file hoặc path, codebase có thể ghép metric của run này với UQ của run khác, đếm retention copy thành run mới, dùng smoke result làm final benchmark, hoặc xóa nhầm raw trace/checkpoint cần cho audit.

Code vẫn có thể chạy, nhưng report và provenance sẽ không còn đáng tin.

#### Vì sao cần giải quyết?

Cây remote sâu và phân mảnh: mỗi method có thể dùng schema, độ sâu và quy tắc đặt tên khác nhau. Raw trace lại lớn hơn summary rất nhiều. Mục tiêu là giữ:

```text
metric + UQ summary + provenance + checkpoint
```

và chỉ xem raw trace là ứng viên xóa sau khi các thành phần trên đã được kiểm tra.

#### Thuật ngữ cần biết

- `evaluation_metrics.json`: metric chính cho bảng 1.
- `uq_summary.json`: summary UQ cho bảng 2.
- `best.pt`: checkpoint tốt nhất.
- `stage_b_init.pt`: checkpoint khởi tạo Stage B.
- `truncated_smoke_evaluation`: test không phủ hết timeline; codebase hiện dùng nhãn này cho cả chạy smoke lẫn trường hợp `test_stride` để lại phần đuôi chưa được đánh giá.
- `non_comparable`: chưa được phép so sánh công bằng với benchmark hoàn chỉnh.
- `identity_conflict`: path và metadata mô tả variant/run khác nhau.
- `partial`: chỉ có một phần thông tin cần thiết.
- `provenance`: thông tin cho biết kết quả đến từ config, checkpoint, protocol và run nào.

### Kết quả thực hiện Phase 2 — 2026-07-23

Em đã kiểm tra tuần tự từng nhóm, không gom tất cả method vào một lần xử lý.

#### 1. Nhóm `offline_benchmark`

Có 27 file `offline_metrics.json`. Các file này đã chứa đủ ba metric chính: `vus_pr`, `affiliation_f1`, và `vus_roc`, nên có thể dùng làm nguồn metric dự phòng nếu schema và protocol được xác nhận. Theo phạm vi hai bảng report hiện tại, nhóm này **không bắt buộc có summary về uncertainty quantification**; UQ ở đây là `not_applicable`, không phải một artifact bị thiếu bắt buộc.

Nhóm này vẫn không có checkpoint hoặc `resolved_protocol.json`. Vì vậy:

- có thể có dữ liệu cho bảng metric chính;
- không tham gia bảng `mean of means` và `mean of variances`;
- chưa đủ provenance để tự động xem là kết quả benchmark hoàn chỉnh.

Trạng thái phù hợp hơn: `metric_only`, với `uq_not_applicable`.

#### 2. Nhóm `redlamp_baseline`

Có 9 file `evaluation_metrics.json` và 9 checkpoint `best.pt`, nhưng không có UQ summary hoặc resolved protocol trong cây kết quả này.

Chín file metric của nhóm này đều có hai trạng thái:

```text
protocol_status=truncated_smoke_evaluation
benchmark_comparability=non_comparable
```

Các file có dạng:

```text
outputs/benchmark/smd/redlamp_baseline/machine_*/seed*/evaluation_metrics.json
```

Không nên diễn giải ngay rằng toàn bộ run là một smoke run có chủ ý. Audit protocol của remote cho thấy nguyên nhân trực tiếp là `test_stride=20`, khiến phần đuôi timeline test không được phủ. Các trạng thái vẫn phải được giữ trong record và không được che đi khi dựng bảng.

#### 3. Nhóm `thesis`

Có 18 combination thuộc Stage B. Các file summary ở đúng vị trí Stage B đều đạt những điều kiện kỹ thuật sau:

- có đủ `vus_pr`, `affiliation_f1`, và `vus_roc`;
- có summary không-null cho `clean_validation` và `test`;
- có checkpoint tốt nhất `best.pt`;
- có checkpoint khởi tạo `stage_b_init.pt`;
- checksum của checkpoint đã được tính.

Nhưng cả 18 combination đều có:

```text
protocol_status=truncated_smoke_evaluation
benchmark_comparability=non_comparable
```

Do đó các summary hiện có đủ về mặt cấu trúc, nhưng chưa đủ điều kiện để khẳng định đây là kết quả benchmark cuối cùng. Các file metric cụ thể có dạng:

```text
outputs/benchmark/smd/thesis/O[01]/machine_*/seed*/two_stage/
    stage_b_fusion_finetuning/evaluation_metrics.json
```

### Nguyên nhân của `truncated_smoke_evaluation`

Hai trạng thái này không được ghi ngẫu nhiên. Luồng hiện tại làm như sau:

```text
test_stride=20
        ↓
WindowDataset tạo start = 0, 20, 40, ... và không thêm tail window
        ↓
phần đuôi timeline không có prediction
        ↓
evaluated_num_points < raw_num_points
        ↓
non_comparable + truncated_smoke_evaluation

( =^･ω･^= )  metric không “bịa” trạng thái;
             protocol hiện tại tạo ra trạng thái đó.
```

Ví dụ trên remote:

- `machine_1_6`: `23680/23689` điểm được đánh giá;
- `machine_3_4`: `23680/23687` điểm được đánh giá;
- `machine_3_9`: `28700/28713` điểm được đánh giá.

Vì vậy code đang phản ánh đúng coverage thực tế, nhưng nhãn `truncated_smoke_evaluation` không phù hợp nếu đây được coi là benchmark chính thức. Không được sửa giá trị trong JSON bằng tay. Cần sửa protocol/config để phủ toàn timeline, chẳng hạn dùng `test_stride=1` hoặc cơ chế end-align phù hợp, rồi đánh giá lại. Chỉ khi coverage đầy đủ thì mới được ghi `benchmark_comparable_full_timeline`.

### Các vấn đề cần giải quyết trước khi gộp dữ liệu

#### Vấn đề 1 — Có nhiều file UQ ở các vị trí khác nhau

Trong một run `thesis`, có thể tồn tại cả hai dạng:

```text
<run>/metrics/uq_summary.json
<run>/two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json
```

Không được chọn file đầu tiên chỉ vì nó gần `<run>`. File cấp `<run>` có thể chứa variance bằng `null`, trong khi file gắn trực tiếp với Stage B mới chứa summary UQ không-null.

Quy tắc adapter phải là:

```text
evaluation_metrics.json của Stage B
        +
uq_summary.json của Stage B
        +
protocol/checkpoint của cùng Stage B
```

Như vậy các artifact được ghép với nhau cùng một stage, thay vì ghép theo độ sâu thư mục hoặc khoảng cách path.

#### Vấn đề 2 — Metadata không khớp với tên thư mục

Kiểm tra bổ sung toàn bộ 9 combination dưới `O1` cho thấy config, experiment name và manifest đều xác định đây là `O1`:

- path có `/thesis/O1/...`;
- `experiment_name` có `__O1__`;
- `two_stage_manifest.json` có experiment name `O1`;
- Stage B resolved config có `experiment_variant: two_stage_point_score_supervised_v1`, khác với `O0` là `two_stage_base_v1`;
- Stage B checkpoint và `stage_b_init.pt` nằm trong cùng cây `O1`.

Tuy nhiên artifact summary lại không nhất quán:

- cả 9 `thresholds/thresholds.json` của `O1` đều ghi `variant_name=O0`;
- 8/9 Stage B `uq_summary.json` của `O1` ghi `variant_name=O0`;
- `O1/machine_1_6/seed6` ghi `variant_name=O1`, nhưng lại có `experiment_name=stage_b_fusion_finetuning` và đường dẫn config bị lặp prefix, nên vẫn là `identity_conflict`.

Nguyên nhân trong code đã được xác định: artifact builder đọc `experiment_config.get("offline_variant", "O0")`, trong khi các config hiện tại dùng trường `experiment_variant`, không dùng `offline_variant`. Vì vậy builder rơi vào giá trị mặc định `O0` khi tạo threshold và UQ metadata.

Điều này tạo ra hai câu trả lời khác nhau cho cùng một câu hỏi:

```text
Tên variant theo path: O1
Tên variant theo metadata: O0
```

Kết luận provenance hiện tại: run và checkpoint thật sự thuộc cây `O1`, nhưng metadata của threshold/UQ bị ghi sai hoặc bị ghi đè. Không được dùng các field `variant_name` sai đó để aggregate. Adapter phải lấy identity từ config, experiment name, manifest và checkpoint cùng Stage B; nếu còn mâu thuẫn thì record vẫn phải được đánh dấu `identity_conflict`.

#### Vấn đề 3 — Một retention copy có tên path bất thường

Có 73 file `uq_summary.json` nhưng chỉ có 72 nhóm trace. File dư là một bản retention copy của `O1/machine_3_9/seed6`:

```text
retention/machine_3_9/offline/uq_summary.json
```

Trong khi naming convention thông thường là:

```text
retention/machine-3-9/offline/uq_summary.json
```

Vì vậy đây là bản sao có path không nhất quán, không phải combination thứ 73. Không được đếm nó như một run mới.

### Kết luận của Gate 2

Gate 2 hiện `blocked` đối với việc tạo canonical records cuối cùng và gộp dữ liệu vào report. Lý do là:

1. test coverage của các nhóm `redlamp_baseline` và `thesis` bị truncated do `test_stride=20`, nên chưa comparable;
2. một số identity trong path và metadata không khớp;
3. cần chốt quy tắc chọn Stage B UQ summary, sửa nguồn identity của artifact builder và loại bản retention duplicate;
4. `offline_benchmark` chỉ là nguồn metric chính, không cần UQ summary.

Vì vậy chưa tạo `offline_report_data.json`, chưa tạo prune manifest và chưa thực hiện bất kỳ thao tác sửa/xóa nào trên remote. Đây là trạng thái bảo thủ để tránh đưa nhầm hoặc đưa quá mức kết quả chưa đủ provenance vào báo cáo.

### Kết quả thực hiện lại Phase 2 — 2026-07-24

Stage 2 đã được thực hiện lại theo đúng thứ tự từng method: `offline_benchmark` → `redlamp_baseline` → `thesis`. Mỗi method được kiểm tra boundary, schema, identity, report fields và provenance trước khi chuyển sang method tiếp theo. Không có artifact nào trên remote bị sửa hoặc xóa.

#### Cách đọc kết quả

Mục tiêu của Stage 2 là xác định mỗi file đang nói về combination nào và có thể ghép an toàn với file nào khác hay không. Một canonical record là một bản ghi duy nhất đại diện cho một combination, được tạo từ các artifact cùng stage:

```text
metric Stage B + UQ Stage B + protocol Stage B + checkpoint Stage B
                              ↓
                    một canonical record
                              ↓
                         report hoặc prune
```

Trong `uq_summary.json`, hai số liệu cần cho bảng thứ hai được map như sau: `point_score_summary.mean` là `mean of means`, tức trung bình các mean anomaly score của từng point; `uncertainty_summary.point_anomaly_score_variance_mean` là `mean of variances`, tức trung bình variance anomaly score của từng point. Các field tương ứng của `clean_validation`, `synthetic_validation` và `test` đều có giá trị số trong 18 Stage B candidates của `thesis`; vì vậy về mặt field summary, có thể lập bảng so sánh validation–testing sau khi giải quyết comparability và identity.

#### 1. `offline_benchmark` — `metric_only`

Boundary được xác định là `/root/bachelor-thesis-2026/outputs/benchmark/smd/offline_benchmark/`. Có 3 variant: `iforest`, `kmeans_ad`, `stumpy_channel_ab`; mỗi variant có 3 entity và 3 seed, tổng cộng 27 combinations.

Toàn bộ 27 `offline_metrics.json` dùng cùng một schema và đều có ba field số `vus_pr`, `affiliation_f1`, `vus_roc`. Adapter của method này có thể đọc trực tiếp ba field đó. Không có `evaluation_metrics.json`, `metrics/uq_summary.json`, checkpoint hoặc `resolved_protocol.json` tương ứng.

Kết luận: nhóm này có thể cung cấp metric chính cho bảng thứ nhất dưới dạng fallback source, nhưng không tham gia bảng `mean of means` và `mean of variances`. UQ là `not_applicable` theo phạm vi report hiện tại, còn provenance benchmark vẫn là `partial` vì thiếu protocol và checkpoint.

#### 2. `redlamp_baseline` — `not_comparable`

Boundary được xác định là `/root/bachelor-thesis-2026/outputs/benchmark/smd/redlamp_baseline/`. Có 3 entity và 3 seed, tổng cộng 9 combinations. Mỗi combination có một `evaluation_metrics.json` và một `checkpoints/best.pt`; không có UQ summary hoặc `resolved_protocol.json`, đồng thời path không chỉ ra rõ variant, phase và stage.

Cả 9 metric file đều có đủ `vus_pr`, `affiliation_f1`, `vus_roc`, nhưng tất cả cùng mang:

```text
protocol_status=truncated_smoke_evaluation
benchmark_comparability=non_comparable
```

Coverage thực tế là 3 run có `23680/23687` points, 3 run có `23680/23689` points và 3 run có `28700/28713` points. Nghĩa là mỗi run chỉ đánh giá một phần timeline test. Vì vậy không đưa nhóm này vào aggregate benchmark cuối cùng.

#### 3. `thesis` — 18 Stage B candidates, nhưng vẫn bị chặn

Boundary được xác định là `/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O0/` và `/root/bachelor-thesis-2026/outputs/benchmark/smd/thesis/O1/`. Có 9 combinations dưới `O0` và 9 combinations dưới `O1`.

Mỗi Stage B candidate có đủ bộ artifact đúng stage:

```text
two_stage/stage_b_fusion_finetuning/evaluation_metrics.json
two_stage/stage_b_fusion_finetuning/metrics/uq_summary.json
two_stage/stage_b_fusion_finetuning/protocol/resolved_protocol.json
two_stage/stage_b_fusion_finetuning/checkpoints/best.pt
two_stage/initializations/stage_b_init.pt
```

18 file metric đều có đủ ba metric chính và 18 file UQ đều có các field cần cho `clean_validation`, `synthetic_validation` và `test`. Cả 18 `best.pt` đều khớp checksum SHA-256 ghi trong UQ; cả 18 `stage_b_init.pt` đều tồn tại và không rỗng.

Tuy nhiên cả 18 metric file đều có `23680 < 23687`, `23680 < 23689` hoặc `28700 < 28713`, nên đều bị đánh dấu `truncated_smoke_evaluation` và `non_comparable`. `resolved_protocol.json` còn ghi `offline_window_stride=20` và `offline_tail_policy=end_align`, nhưng `WindowDataset` hiện tại chỉ tạo start index bằng `range(..., stride)` và không thêm end-aligned tail window. Đây là dấu hiệu protocol khai báo một chính sách nhưng loader chưa thực hiện đầy đủ chính sách đó.

#### Kiểm tra identity của `O0` và `O1`

`O0` nhất quán: path là `O0`, experiment name có `__O0__`, resolved config có `experiment_variant=two_stage_base_v1`, và UQ metadata ghi `variant_name=O0`.

`O1` không nhất quán. Path, experiment name của 8 run, two-stage manifest và resolved config đều chỉ về `O1`; resolved config có `experiment_variant=two_stage_point_score_supervised_v1`. Tuy nhiên cả 9 `thresholds/thresholds.json` ghi `variant_name=O0`, 8/9 Stage B UQ ghi `variant_name=O0`, còn `O1/machine_1_6/seed6` ghi `variant_name=O1` nhưng lại có `experiment_name=stage_b_fusion_finetuning`, đường dẫn config bị lặp prefix và các scalar log `query/num_samples_eval`/`query/num_samples_train` là `null`.

Do đó, kết luận an toàn là cả 9 run dưới `O1` đều có `identity_conflict`, dù checkpoint và report fields của chúng vẫn tồn tại. Adapter không được lấy `variant_name` từ threshold/UQ một cách mù quáng; phải ưu tiên path, resolved config, experiment name, manifest và checkpoint cùng Stage B, đồng thời giữ cờ conflict trong canonical record.

#### Kết luận Gate 2 ngày 2026-07-24

Gate 2 tiếp tục `blocked` đối với việc tạo canonical records cuối cùng, aggregate report và prune manifest. Có ba nguyên nhân chính:

1. `redlamp_baseline` và `thesis` chưa phủ đầy đủ timeline test nên chưa `benchmark_comparable`.
2. 9 run `O1` có conflict giữa path/config/experiment provenance và metadata của threshold/UQ.
3. Một UQ record của `O1/machine_1_6/seed6` có metadata malformed và thiếu thông tin số lần stochastic inference trong scalar logs, dù các summary field chính vẫn tồn tại.

Vì vậy chưa tạo `offline_report_data.json`, chưa tạo prune manifest và chưa thực hiện thao tác sửa/xóa remote. Các raw trace chỉ được xem là ứng viên prune sau khi protocol được sửa và rerun, identity được xác nhận, canonical records được tạo, rồi dry-run được kiểm tra.

### Quyết định áp dụng sau Phase 2 — 2026-07-24

- `offline_benchmark` được phép cung cấp metric hiện có cho bảng 1 từ `offline_metrics.json`; không yêu cầu UQ summary vì đây là nhóm traditional machine learning. Trạng thái provenance vẫn phải ghi `partial` và không được diễn giải là benchmark provenance đầy đủ.
- Coverage thiếu 7, 9 hoặc 13 points ở phần đuôi được chấp nhận cho report hiện tại vì đây là tail gap rất nhỏ, tương ứng với một end-aligned window chưa được thêm. Canonical record phải giữ nguyên `evaluated_num_points`, `raw_num_points`, coverage ratio và `coverage_policy=near_complete_tail_gap`; không sửa ngược các status cũ trong JSON.
- Áp dụng Strategy 1 + Strategy 2 cho toàn bộ run: resolve identity theo thứ tự path, resolved config, experiment name, manifest, checkpoint path/SHA-256 rồi mới dùng threshold/UQ metadata; đồng thời bind các artifact cùng Stage B bằng output directory và checkpoint SHA-256.
- Áp dụng Strategy 3 cho exception `thesis/O1/machine_1_6/seed6` bằng reconciliation entry riêng. Identity canonical được resolve là `O1`, còn metadata gốc, đường dẫn config bị lặp và scalar logs bị thiếu vẫn phải được giữ trong diagnostics.
- Quy tắc resolve identity đã được ghi vào `outputs/reporting/offline_phase_tables/identity_reconciliation.json`. Manifest này là lớp reconciliation local; không sửa artifact gốc trên remote và không yêu cầu chạy lại Stage A hoặc Stage B.

### Kết quả thực hiện lại Phase 2 — 2026-07-25

Ba method được xử lý tuần tự bằng adapter riêng. `offline_benchmark` tạo 27 records, cả 27 có đủ `vus_pr`, `affiliation_f1`, `vus_roc`; UQ được ghi `not_applicable`. `redlamp_baseline` tạo 9 records, cả 9 có ba metric chính nhưng vẫn giữ nguyên `protocol_status=truncated_smoke_evaluation` và `benchmark_comparability=non_comparable`. `thesis` tạo 18 Stage B records, cả 18 đủ metric chính và UQ cho `clean_validation` và `test`; 9 record có identity diagnostics do metadata variant thấp hơn không khớp, nhưng identity được resolve theo path, config, experiment name, manifest và checkpoint binding.

Không chạy lại Stage A hoặc Stage B. Tail gap nhỏ được giữ trong coverage diagnostics theo policy đã chấp thuận; không sửa status gốc trong artifact remote. Gate 2 đạt ở mức có thể tạo canonical records kèm diagnostics và tiếp tục sang Phase 3–5.

## Phase 3 — Chuẩn hóa record và provenance

### Stage 3.1 — Tạo canonical run record

Mỗi record có `run_id`, `experiment_type`, `dataset_name`, `entity_name`, `seed_value`, `method_name`, `variant_name`, `phase_name`, `stage_name`, metric/UQ source paths, checkpoint khởi tạo, checkpoint tốt nhất, protocol path và provenance.

### Stage 3.2 — Kiểm tra metric mapping

Không tự động đồng nhất `pr_auc` với `vus_pr`, F1 chung với `affiliation_f1`, hoặc `roc_auc` với `vus_roc` nếu protocol chưa xác nhận semantics tương đương. Nếu fallback hợp lệ, ghi `metric_source_key`, `metric_mapping_rule`, và `metric_semantic_status`.

### Stage 3.3 — Kiểm tra split mapping

Tách `clean_validation`, `synthetic_validation` và `test`. Nếu validation không rõ loại, dùng `validation_unclassified` thay vì đoán. `clean_validation` là split validation chính để so sánh với `test`; `synthetic_validation` phải được báo cáo riêng, không trộn vào clean validation.

### Stage 3.4 — Kiểm tra checkpoint

Xác nhận checkpoint khởi tạo và tốt nhất tồn tại; ghi role, path và checksum.

### Gate 3

Không còn ambiguity chưa được ghi nhận về identity, metric, split hoặc checkpoint provenance.

## Phase 4 — Kiểm tra summary và tạo report bundle

### Stage 4.1 — Kiểm tra required fields

Bảng metric cần ba metric chính. Bảng validation–testing point-level cần:

```text
clean_validation.point_score_summary.mean
clean_validation.uncertainty_summary.point_anomaly_score_variance_mean
test.point_score_summary.mean
test.uncertainty_summary.point_anomaly_score_variance_mean
```

Các giá trị phải được kiểm tra riêng cho từng combination `variant/entity/seed/stage`, không chỉ kiểm tra một file đại diện.

### Stage 4.2 — Kiểm tra null và comparability

Phân biệt `missing`, `null`, `invalid`, `not_comparable`, và `near_complete_tail_gap`. Record có đủ field và tail gap nhỏ được aggregate khi policy đã ghi rõ coverage; không được biến `near_complete_tail_gap` thành `benchmark_comparable_full_timeline`.

### Stage 4.3 — Tạo report bundle local

Tạo `outputs/reporting/offline_phase_tables/offline_report_data.json` gồm row-level identity, metric, UQ summaries, source paths, mapping rules và provenance/trace-audit flags. Bundle phải đủ để dựng lại hai bảng mà không cần đọc raw trace.

### Stage 4.4 — Tạo bảng validation–testing

Theo từng combination, lưu mean of means của validation/test, mean of variances của validation/test, và:

```text
mean_of_means_validation
  = clean_validation.point_score_summary.mean

mean_of_means_test
  = test.point_score_summary.mean

mean_of_variances_validation
  = clean_validation.uncertainty_summary.point_anomaly_score_variance_mean

mean_of_variances_test
  = test.uncertainty_summary.point_anomaly_score_variance_mean

test_minus_validation
  = mean_of_variances_test - mean_of_variances_validation
```

Tính từng run trước, sau đó mới aggregate qua seed nếu cần; không gộp raw point của nhiều run khi mỗi run cần có trọng số bằng nhau.

### Gate 4

Report bundle đã tạo; required fields, checkpoint và provenance đã được kiểm tra.

### Kết quả thực hiện Phase 3–4 — 2026-07-24

Đã áp dụng resolver identity theo Strategy 1 + Strategy 2 cho toàn bộ artifact được thu thập, sau đó áp dụng reconciliation entry cho exception `thesis/O1/machine_1_6/seed6`. Resolver không sửa JSON gốc trên remote và không chạy lại Stage A hoặc Stage B.

Kết quả canonical records:

| Nhóm | Số record | Metric table 1 | UQ table 2 |
|---|---:|---:|---:|
| `offline_benchmark` | 27 | 27 | `not_applicable` |
| `redlamp_baseline` | 9 | 9 | `not_available` |
| `thesis/O0` | 9 | 9 | 9 |
| `thesis/O1` | 9 | 9 | 9 |
| Tổng cộng | 54 | 54 | 18 |

`O1` được resolve thành variant `O1` cho cả 9 record dựa trên path, resolved config, manifest, experiment name và checkpoint binding. Các metadata conflict của threshold/UQ không bị xóa; chúng được lưu trong `raw_metadata_variants` và `diagnostics`. Exception `O1/machine_1_6/seed6` có thêm diagnostics về experiment name không đầy đủ, config path bị lặp prefix và scalar log thiếu `query/num_samples_eval`.

Tất cả 18 checkpoint Stage B có SHA-256 khớp giữa UQ metadata và file `best.pt`. Các field `mean_of_means` và `mean_of_variances` của `clean_validation`, `synthetic_validation` và `test` đều có trong 18 UQ records. Coverage tail gap được giữ trong mỗi record với `raw_num_points`, `evaluated_num_points`, `missing_points`, status gốc và `coverage_policy=near_complete_tail_gap`.

Report bundle local đã tạo:

```text
outputs/reporting/offline_phase_tables/offline_report_data.json
outputs/reporting/offline_phase_tables/offline_report_data.md
outputs/reporting/offline_phase_tables/identity_reconciliation.json
```

Gate 3 đạt ở mức identity đã được resolve và mọi conflict còn lại đều có diagnostics. Gate 4 đạt: bundle có 54 row-level records, dựng được bảng metric chính từ 54 record và bảng validation–testing từ 18 record UQ. Chưa tạo prune manifest và chưa xóa raw trace trên remote.

Kiểm tra local sau khi tạo bundle: JSON parse thành công, tất cả 54 record đủ metric bảng 1, tất cả 18 record `thesis` đủ field bảng 2, và không có checkpoint SHA-256 mismatch. Hai script ops mới compile thành công. Full test suite có `442 passed`, `1 skipped`, `10 failed`; các failure nằm ở snapshot config và các test memory/training model hiện hữu, không nằm trong các file resolver/report mới.

### Kết quả thực hiện lại Phase 3–4 — 2026-07-25

Report bundle được dựng lại từ remote hiện tại và ghi trên local. Bundle có 54 records: 27 từ `offline_benchmark`, 9 từ `redlamp_baseline` và 18 Stage B từ `thesis`. Bảng metric chính có đủ 54 records; bảng UQ có đủ 18 records Stage B. Mỗi record Stage B có `mean_of_means` và `mean_of_variances` cho `clean_validation` và `test`.

Đã bổ sung `validation_testing_comparison` cho 18 Stage B records, gồm `mean_of_means_validation`, `mean_of_means_test`, `mean_of_variances_validation`, `mean_of_variances_test` và hai delta `test_minus_validation`. Không có comparison field nào bị thiếu. Kiểm tra remote xác nhận 18 Stage B có đủ UQ, protocol, threshold, `best.pt`, `stage_b_init.pt`; SHA-256 của cả 18 `best.pt` khớp giá trị trong UQ summary.

Report bundle hiện tại:

```text
outputs/reporting/offline_phase_tables/offline_report_data.json
outputs/reporting/offline_phase_tables/offline_report_data.md
outputs/reporting/offline_phase_tables/identity_reconciliation.json
```

Gate 3–4 đạt. Các trạng thái `truncated_smoke_evaluation` và `non_comparable` vẫn được bảo toàn trong diagnostics; chúng không bị đổi thành trạng thái benchmark comparable.

## Phase 5 — Tạo prune manifest và dry-run

### Stage 5.1 — Phân loại artifact

Bắt buộc giữ:

```text
evaluation_metrics.json
metrics/uq_summary.json
checkpoint khởi tạo của stage
checkpoint tốt nhất của stage
```

Nên giữ vì nhẹ và hữu ích cho audit:

```text
thresholds.json
resolved_protocol.json
retention_bundle_manifest.json
retention_summary.json
```

Giữ tạm `metrics/offline_metrics.json` nếu nó chưa được đối chiếu với `evaluation_metrics.json`. Chỉ xóa sau khi đã chốt nguồn canonical và report bundle đã được kiểm tra. Các file `*_point_scores.npz` không bắt buộc cho hai bảng nhưng có thể giữ vì thường rất nhỏ.

Ứng viên xóa sau Gate 4 là `clean_validation_traces.json`, `synthetic_validation_traces.json`, `test_traces.json`, duplicate retention traces và raw per-forward-pass tensors.

### Stage 5.2 — Tạo manifest

Mỗi dòng ghi absolute path, logical run ID, method, artifact role, size, checksum nếu cần, action `keep/delete/review`, và lý do.

### Stage 5.3 — Dry-run

Kiểm tra số file, tổng dung lượng, phạm vi absolute path, required summaries và đảm bảo không checkpoint nào bị đánh dấu delete.

### Stage 5.4 — Kiểm tra điều kiện xóa từng run

Chỉ đánh dấu raw trace là `delete` khi run đó đã đạt đồng thời các điều kiện phù hợp với method:

1. Artifact metric của method có `vus_pr`, `affiliation_f1`, `vus_roc`.
2. Nếu method yêu cầu UQ, `uq_summary.json` có summary không-null cho `clean_validation` và `test`; với `offline_benchmark`, UQ được ghi `not_applicable`.
3. Identity `variant`, `entity`, `seed`, `phase`, `stage` không bị thiếu hoặc đã được reconciliation manifest resolve.
4. Row của run đã có trong `offline_report_data.json`.
5. Checkpoint khởi tạo/tốt nhất và checksum provenance đã được kiểm tra nếu method có checkpoint.
6. Coverage policy và diagnostics đã được ghi trong canonical record.
7. Manifest sẽ được rebuild nếu artifact entry bị xóa.

### Gate 5

Dry-run chính xác và manifest không chứa artifact được bảo vệ.

### Kết quả thực hiện Phase 5 — 2026-07-24

Đã lập inventory read-only mới từ `/root/bachelor-thesis-2026/outputs/benchmark/smd` và tạo prune manifest local bằng `scripts/ops/build_prune_manifest.py`. Manifest dùng absolute path, logical run ID, method, variant, entity, seed, phase, stage, artifact role, dung lượng, action và lý do.

Kết quả dry-run:

| Action | Số artifact | Ý nghĩa |
|---|---:|---|
| `keep` | 630 | Summary, UQ, protocol, threshold, retention metadata, checkpoint và `.npz` nhỏ |
| `review` | 99 | `offline_metrics.json`, giữ lại để xử lý source discrepancy |
| `delete` candidate | 216 | Ba loại raw trace của 18 run THESIS, gồm cả bản trong `traces/` và `retention/` |

216 raw trace có tổng dung lượng `7,395,484,474` bytes, xấp xỉ `6.888 GiB`. Tất cả artifact được đánh dấu `delete` đều là raw trace; không có evaluation metric, UQ summary, protocol, threshold, retention manifest/summary hoặc checkpoint nào bị đánh dấu delete. Không có identity conflict nào còn ở trạng thái chưa resolve trong các raw trace candidate; `O1` được chấp nhận theo reconciliation đã ghi ở Phase 3–4.

Hậu kiểm local đạt: 945 entry đều có absolute path, số raw trace trong manifest khớp inventory remote (`216` file và cùng tổng byte), `protected_delete_count=0`, và các assertion về action đều thành công. Manifest được lưu tại `outputs/reporting/offline_phase_tables/prune_manifest.json`, kèm bản tóm tắt `prune_manifest.md`.

Gate 5 đạt ở mức dry-run. Phase 6 chưa được thực hiện: chưa xóa, di chuyển hoặc sửa bất kỳ artifact nào trên remote.

### Kết quả thực hiện lại Phase 5 — 2026-07-25

Manifest được tạo lại từ inventory đầy đủ của remote hiện tại tại `outputs/reporting/offline_phase_tables/prune_manifest.json`, kèm bản tóm tắt `prune_manifest.md`. Manifest có 5.511 entries: `keep=667`, `review=4.628`, `delete candidate=216`. Số `review` lớn vì các log, W&B metadata, epoch outputs, `final.pt` và artifact chưa có quy tắc xóa riêng đều được giữ lại để tránh xóa nhầm; trong đó 99 `offline_metrics.json` vẫn giữ ở `review` vì fallback metric source discrepancy chưa đóng.

216 raw trace candidate có tổng dung lượng `7,395,484,474` bytes, xấp xỉ `6.888 GiB`. Tất cả candidate `delete` đều chỉ là `clean_validation_traces.json`, `synthetic_validation_traces.json` hoặc `test_traces.json` dưới `outputs/benchmark/smd`; `protected_delete_count=0`. Remote hiện còn 72 file cho mỗi loại raw trace và tổng cây `smd` là `8,194,004,068` bytes.

Hậu kiểm đạt: report bundle có 54 dòng bảng 1 và 18 dòng bảng 2; 18 Stage B có đủ artifact bắt buộc và SHA-256 checkpoint khớp; không có thao tác sửa hoặc xóa artifact remote. Gate 5 đạt ở mức dry-run. Phase 6 chưa thực hiện.

## Phase 6 — Thực thi prune và hậu kiểm

### Stage 6.1 — Phê duyệt phạm vi

Chỉ thực thi sau khi prune manifest đã được xem xét; không dùng `rm -rf` trên toàn bộ `outputs` hoặc glob không giới hạn.

### Stage 6.2 — Xóa đúng file trong manifest

Chỉ xóa path có `action=delete`; không xóa checkpoint, summary, protocol hoặc report bundle.

### Stage 6.3 — Cập nhật retention manifest

Nếu manifest cũ trỏ tới trace đã xóa, rebuild theo `summary_only` hoặc cập nhật artifact entries để checksum verification vẫn đúng.

### Stage 6.4 — Hậu kiểm

Đọc lại report bundle, kiểm tra summary và checkpoint checksum, đo lại dung lượng, và xác nhận không chạm vào job hoặc method ngoài phạm vi.

### Gate 6

Summary, report bundle, checkpoint và provenance vẫn đọc/verify được.

### Kết quả thực hiện Phase 6 — 2026-07-24

Đã thực thi đúng các entry có `action=delete` trong prune manifest trên `/root/bachelor-thesis-2026/outputs/benchmark/smd`. Đã xóa 216 raw trace, giải phóng `7,395,484,474` bytes, xấp xỉ `6.888 GiB`. Sau khi xóa, không còn raw trace thuộc ba loại được chỉ định trong experiment root; dung lượng hiện tại của `outputs/benchmark/smd` là `798,472,714` bytes, khoảng `772M` theo `du -sh`.

Đã cập nhật 36 `retention_bundle_manifest.json` sang `summary_only`, xóa 108 trace entry khỏi các manifest và cập nhật 36 `retention_summary.json`. Có 233 artifact còn lại trong retention manifest được kiểm tra checksum thành công. Tất cả 729 entry `keep` hoặc `review` trong prune manifest vẫn tồn tại trên remote; không có checkpoint, metric, UQ summary, protocol hoặc threshold nào bị đánh dấu xóa.

Hậu kiểm report bundle local thành công: có 54 record cho bảng metric chính và 18 record cho bảng validation–testing. Tất cả 54 record có `VUS-PR`, `affiliation F1` và `VUS-ROC`; cả 18 record UQ đều có `mean_of_means` và `mean_of_variances` cho `clean_validation` và `test`. Có đủ 18 checkpoint tốt nhất với SHA-256 khớp report bundle và đủ 18 checkpoint khởi tạo. Không chạy lại Stage A/Stage B và không chạm tới job đang chạy.

Gate 6 đạt. Phase 6 không xóa các artifact `review`; các `offline_metrics.json` vẫn được giữ để xử lý source discrepancy sau.

### Kết quả thực hiện lại Phase 6 — 2026-07-25

Đã preflight lại đúng manifest hiện tại rồi xóa chính xác 216 raw trace bằng absolute path, với tổng kích thước `7,395,484,474` bytes, xấp xỉ `6.888 GiB`. Không dùng glob, không xóa thư mục và không xóa artifact `keep` hoặc `review`. Sau khi xóa, cả ba loại `clean_validation_traces.json`, `synthetic_validation_traces.json` và `test_traces.json` đều còn `0` file; dung lượng hiển thị của `outputs/benchmark/smd` là khoảng `772M`.

Đã cập nhật 36 `retention_bundle_manifest.json` và 36 `retention_summary.json`: xóa tổng cộng 108 trace entry ở mỗi loại metadata, đặt `retention_policy=summary_only`, và đánh dấu `inspection_ready=false` trong retention summary. Một hậu kiểm trung gian phát hiện hash của `retention_summary.json` trong manifest còn cũ do thứ tự cập nhật; đã rehash lại toàn bộ 36 manifest. Hậu kiểm cuối xác nhận 233 retained artifact checksum khớp và `retention_issues=0`.

Toàn bộ 5.295 artifact `keep/review` trong prune manifest vẫn tồn tại. Report bundle local vẫn có 54 record bảng metric chính, 18 record bảng UQ, 18 comparison validation–testing và đủ 18 checkpoint tốt nhất cùng 18 checkpoint khởi tạo; SHA-256 của 18 `best.pt` vẫn khớp UQ summary. Không chạy lại Stage A/Stage B và không chạm tới job đang chạy. Gate 6 đạt; các `offline_metrics.json` vẫn được giữ ở `review`.

## Phase 7 — Đóng gói audit

### Stage 7.1 — Lưu audit artifacts

Giữ local `offline_report_data.json`, `canonical_run_manifest.json`, `prune_manifest.json`, và `coverage_gap_report.json`.

### Stage 7.2 — Ghi kết luận

Ghi methods đã xử lý, methods bị block, số run giữ lại, số/dung lượng file đã xóa, coverage của report fields và limitation còn lại.

### Stage 7.3 — Điều kiện hoàn tất

Hoàn tất khi từng method có trạng thái rõ, hai bảng dựng được từ summary bundle, checkpoint khởi tạo/tốt nhất của mỗi stage còn tồn tại, và prune manifest phản ánh đúng trạng thái artifact sau cleanup.

### Kết quả thực hiện Phase 7 — 2026-07-26

Đã kiểm tra các artifact local trước khi đóng gói. Ban đầu đã có
`offline_report_data.json`, `identity_reconciliation.json`,
`prune_manifest.json` và bản Markdown của prune manifest. Hai artifact cần
thêm cho audit là `canonical_run_manifest.json` và `coverage_gap_report.json`;
đã tạo cả hai từ report bundle, reconciliation policy và prune manifest hiện
có. `canonical_run_manifest.json` giữ identity đã chuẩn hóa, report fields,
provenance, checkpoint evidence và diagnostics cho từng run. `coverage_gap_report.json`
ghi riêng coverage gap, raw status, identity conflict, limitation và kết quả
cleanup để không cần mở lại raw trace.

Stage 7.2 đã ghi nhận trạng thái method như sau:

1. Các method traditional machine learning trong `offline_benchmark`
   (`iforest`, `kmeans_ad`, `stumpy_channel_ab`) có 27 run, đủ metric cho
   bảng 1. UQ không áp dụng cho nhóm này nên bảng 2 không được xem là bị thiếu.
2. `redlamp_baseline` có 9 run đủ metric cho bảng 1. Nhóm này không có UQ;
   một số run có tail gap rất nhỏ và được chấp nhận theo
   `near_complete_tail_gap`, đồng thời giữ diagnostic gốc.
3. `THESIS` có 18 run đủ metric cho bảng 1 và đủ UQ cho bảng 2. Có 9 run có
   metadata variant mâu thuẫn; tất cả đã được resolve bằng path, config,
   manifest và checkpoint, còn raw conflict vẫn được giữ trong diagnostics.

Không có method hoặc record nào bị `blocked`. Bảng 1 có 54/54 record đủ
`VUS-PR`, `affiliation F1` và `VUS-ROC`. Bảng 2 có 18/18 record THESIS đủ
`mean of means` và `mean of variances` cho `clean_validation` và `test`,
đồng thời có 18 dòng so sánh validation–testing.

Prune manifest hiện phản ánh cả kế hoạch và trạng thái sau cleanup: 216 raw
trace đã xóa, giải phóng `7,395,484,474` bytes, khoảng `6.888 GiB`; còn 667
artifact `keep` và 4.628 artifact `review`. Hậu kiểm read-only trên host
`unstoppable-puma` ngày 2026-07-26 xác nhận cả ba loại raw trace còn 0 file,
36 retention manifest, 36 retention summary, 233 retained artifact checksum
khớp và `retention_issues=0`. Mỗi 18 THESIS run vẫn có checkpoint tốt nhất,
checkpoint khởi tạo và SHA-256 của checkpoint tốt nhất khớp UQ summary.

Các limitation vẫn phải hiển thị trong audit bundle: 27 record còn giữ raw
label `protocol_status=truncated_smoke_evaluation` và
`benchmark_comparability=non_comparable`; 27 run có tail gap gần hoàn chỉnh;
`offline_benchmark` và `redlamp_baseline` không có UQ; 9 THESIS run có
metadata conflict đã được resolve. Không được đổi các raw label này thành
full-timeline comparable.

Gate 7 đạt. Bốn artifact audit local cần giữ là
`offline_report_data.json`, `canonical_run_manifest.json`,
`prune_manifest.json` và `coverage_gap_report.json`.

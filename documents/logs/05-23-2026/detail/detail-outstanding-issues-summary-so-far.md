---
date: 2026-05-23 15:30:00 +07 +0700
author: Codex
status: draft-living-document
scope: "Outstanding issues discussed so far"
---

# Tổng hợp chi tiết các vấn đề còn tồn đọng (so far)

Tài liệu này tổng hợp các vấn đề đã thảo luận xuyên suốt phiên làm việc hiện tại, tập trung vào các nhóm lỗi/chệch hành vi đang ảnh hưởng trực tiếp đến chất lượng thí nghiệm và khả năng bảo trì codebase.

## 1) Nhóm vấn đề mô hình và hành vi học

### 1.1. Mô hình đang sụp về dự đoán một lớp (collapse to class 0)
- Mô tả:
  - Confusion matrix train và val_synth cho thấy toàn bộ hàng đều dồn vào cột dự đoán lớp `0`.
  - Đây là dấu hiệu classifier học một chiến lược tầm thường (degenerate solution), không học ranh giới lớp bất thường.
- Tác động:
  - Phần classification mất giá trị chẩn đoán.
  - Tín hiệu supervision cho các nhánh representation/fusion trở nên lệch, ảnh hưởng gián tiếp đến reconstruction path.
- Bằng chứng đã quan sát:
  - Hai heatmap confusion matrix (train, val_synth) của run Exp2 tại epoch 100 đều dồn cột `0`.

### 1.2. Label refurbishment đang thiên vị class 0
- Mô tả:
  - Trong multiclass setup, target class 0 đang được cộng bias (alpha/beta smoothing + cộng thêm vào class 0).
  - Cơ chế này làm tăng prior của class 0 trong target distribution.
- Tác động:
  - Nếu đi cùng phân bố dữ liệu vốn đã lệch về class 0, mô hình càng dễ collapse về class 0.
- Trạng thái:
  - Đã truy vết code-level và xác định đây là yếu tố rủi ro thực sự, chưa có chỉnh sửa triệt để trong phiên này.

### 1.3. Classification loss chưa có class-weighting rõ ràng cho bài toán multiclass synthetic
- Mô tả:
  - CE loss đang chạy theo thiết lập mặc định, chưa có cơ chế cân bằng theo tần suất lớp trong pipeline hiện tại.
- Tác động:
  - Với phân phối class lệch, mô hình bị khuyến khích tối ưu theo lớp chiếm đa số.
- Trạng thái:
  - Đã xác định là một yếu tố quan trọng cần xử lý, chưa đóng.

## 2) Nhóm vấn đề dữ liệu tổng hợp (synthetic anomalies)

### 2.1. Tỉ lệ synthetic anomaly class chưa cân bằng giữa các lớp
- Mô tả:
  - Biểu đồ ratio theo epoch cho thấy class 0 chiếm tỷ lệ rất lớn so với từng lớp anomaly riêng lẻ.
  - Dù các anomaly family có sampling, tổng thể phân phối window-level vẫn lệch mạnh.
- Tác động:
  - Trực tiếp gây lệch learning signal cho classifier.
  - Khi đánh giá confusion matrix theo chuẩn multiclass sẽ dễ xuất hiện chế độ dự đoán một lớp.
- Trạng thái:
  - Chưa được cân bằng lại trong cơ chế inject/sampling ở mức chính sách.

### 2.2. Bug semantics trong logging số lượng anomalous windows cho multiclass
- Mô tả:
  - Đếm `anomalous_windows = int(classification_labels.sum())` là sai nghĩa với multiclass label id.
  - Tổng label id không tương đương số lượng cửa sổ bất thường.
- Tác động:
  - Logging gây hiểu nhầm về mức độ anomaly thực tế.
  - Có thể làm sai phân tích hậu kiểm.
- Trạng thái:
  - Đã phát hiện vị trí cụ thể; cần sửa để dùng điều kiện nhị phân đúng nghĩa (ví dụ `label != 0`).

## 3) Nhóm vấn đề reconstruction loss rung lắc

### 3.1. Reconstruction loss dao động mạnh ở cả Exp1 và Exp2
- Mô tả:
  - Hai đường `train_reconstruction_loss` (màu xám/cam) đều rung lắc đáng kể theo “step” trên UI.
- Tác động:
  - Khó đánh giá xu hướng hội tụ thật.
  - Rủi ro checkpoint selection không phản ánh chất lượng tái tạo ổn định.
- Ghi chú quan trọng:
  - Trục “Step” trên W&B có thể đang đại diện cho epoch-level logging ở code hiện tại, không nhất thiết là batch step.
- Trạng thái:
  - Chưa kết luận root cause duy nhất; cần tracing tiếp theo cơ chế log và các thành phần loss/augmentation ảnh hưởng.

## 4) Nhóm vấn đề quan sát/giám sát và trực quan hóa

### 4.1. Màu trực quan hóa class ratio gây nhầm lẫn
- Mô tả:
  - Hai lớp (ví dụ class 0 và class 10) trước đó dùng màu gần giống nhau.
- Tác động:
  - Đọc biểu đồ khó, dễ kết luận nhầm.
- Trạng thái:
  - Đã cải thiện palette trong script visualization để tách biệt màu tốt hơn.

### 4.2. Exp1 (recon_only) không sinh classification diagnostics là hành vi dự kiến
- Mô tả:
  - Thư mục diagnostics classification không có ở Exp1.
- Tác động:
  - Có thể bị hiểu nhầm là thiếu log; thực tế phù hợp với cấu hình tắt classification path.
- Trạng thái:
  - Đã xác nhận expected behavior.

## 5) Nhóm vấn đề kiến trúc mã và khả năng bảo trì

### 5.1. Epoch loop trong `src/engine/trainer.py` đang quá tải trách nhiệm
- Mô tả:
  - Một vòng lặp epoch đang gánh cùng lúc: train-step orchestration, validation orchestration, diagnostics aggregation, LR scheduling, checkpoint policy.
  - Mật độ trách nhiệm cao, khó đọc top-to-bottom theo nguyên tắc readability-first.
- Tác động:
  - Khó debug regression.
  - Khó mở rộng an toàn cho các thí nghiệm mới.
- Trạng thái:
  - Đã có research audit xác nhận đây là điểm nóng số 1 cần refactor.

### 5.2. Các vòng `for` đa trách nhiệm tương tự còn xuất hiện ở nhiều module
- Mô tả:
  - Các vị trí nổi bật: validation loop trong trainer, online loop, memory bootstrap loop trong thesis_multitask, inject loop trong augment.
- Tác động:
  - Tăng coupling giữa các concern (data/model/logging), giảm maintainability.
- Trạng thái:
  - Đã lập audit file riêng; chưa bước vào refactor plan + implementation đầy đủ.

## 6) Nhóm vấn đề quy trình thí nghiệm/reproducibility

### 6.1. Cần chốt rõ unit logging để tránh hiểu sai giữa epoch-level và step-level
- Mô tả:
  - UI W&B hiển thị “Step”, nhưng metric có thể được log mỗi epoch.
- Tác động:
  - Phân tích sai về độ nhiễu theo batch-step.
- Trạng thái:
  - Đang yêu cầu truy vết thêm line-by-line tại code log metrics và trainer/evaluator hooks.

### 6.2. Cần rà soát nhất quán các config cho 2 thí nghiệm khi so sánh công bằng
- Mô tả:
  - Đã có 2 config riêng cho Exp1/Exp2 và CLI tương ứng, W&B bật.
  - Tuy nhiên, khi diễn giải kết quả cần tách rõ tác động của từng cờ (`enable_classification_path`, diagnostics toggles, evaluator settings).
- Tác động:
  - Tránh đổ lỗi sai nguồn khi phân tích hiện tượng collapse/rung lắc.
- Trạng thái:
  - Chưa có bảng so sánh config-lock đầy đủ trong cùng một tài liệu.

## 7) Các mục đã xử lý một phần nhưng chưa “đóng” hoàn toàn

### 7.1. Trực quan hóa confusion matrix + class ratio
- Đã làm:
  - Đã có script Python để xuất ảnh confusion matrix và stacked class ratio từ outputs.
  - Đã tinh chỉnh màu để dễ phân biệt hơn.
- Chưa đóng:
  - Chưa tích hợp thành bước báo cáo tự động chuẩn hóa cho mọi run.

### 7.2. Stress test test-suite
- Đã làm:
  - Đã xử lý loạt lỗi test và đưa test suite về trạng thái pass tại thời điểm chạy.
- Chưa đóng:
  - Không đồng nghĩa các vấn đề hành vi học (class collapse/recon oscillation) đã được giải quyết.

## 8) Danh sách ưu tiên đề xuất để xử lý tiếp (theo mức độ khẩn cấp kỹ thuật)

1. Chặn collapse classifier về class 0 bằng cách xử lý đồng thời phân phối dữ liệu synthetic + mục tiêu loss/target refinement.
2. Sửa semantics logging multiclass anomalous window count để không làm nhiễu phân tích.
3. Tách nhỏ epoch loop và các loop đa trách nhiệm theo audit readability-first.
4. Chuẩn hóa định nghĩa “step” vs “epoch” trong logging/W&B để diễn giải đồ thị nhất quán.
5. Thiết kế lại báo cáo chẩn đoán tự động cho mỗi run (confusion, class ratio, key focused metrics).

## 9) Trạng thái tài liệu
- Đây là “living document” tổng hợp tồn đọng.
- Sẽ cần cập nhật sau mỗi đợt fix để chuyển từng mục từ `tồn đọng` sang `đã xử lý` kèm bằng chứng kiểm chứng.

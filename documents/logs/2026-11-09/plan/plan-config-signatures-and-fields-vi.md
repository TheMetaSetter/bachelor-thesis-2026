# Kế hoạch: chữ ký hàm và các trường cấu hình cho phiên bản mới

Trạng thái: bản đề xuất để xem xét. Chưa thay đổi mã nguồn hoặc các tệp YAML. Đây là bản diễn giải của [bản tiếng Anh](plan-config-signatures-and-fields.md).

## Mục tiêu và phạm vi

Thiết kế lại `src/core/config.py` và các trường trong những tệp dự kiến thuộc `configs/`.
Có thể viết lại từ đầu. Mã hiện tại chỉ là nguồn tham khảo về cách hoạt động ở mức tổng quan, không ràng buộc cách viết mã mới.
Khi kiểm chứng phiên bản mới, cần giữ các phép so sánh khoa học, ý nghĩa cấu hình và cách xử lý các tệp kết quả cần thiết.

Luồng hiện tại được tóm tắt trong [bản nghiên cứu](../research/research-current-config-flow.md). Thuật ngữ dựa trên `documents/specs/definitions.md`; cây thư mục dự kiến nằm trong `documents/logs/2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md`. Các trường của từng phương pháp bên dưới mới là cấu trúc ban đầu, chưa bao phủ toàn bộ nội dung nghiên cứu.

## Nguyên tắc thiết kế

1. Một **thí nghiệm (experiment)** gồm nhiều **lượt chạy (run)**. Một **cell** xác định cấu hình cho một run. Một **run độc lập (standalone run)** dùng một cell mà không cần experiment.
2. YAML lưu các giá trị cấu hình. Bốn tệp trong `src/protocols/` chứa quy tắc kiểm tra đầu vào và đầu ra theo từng ngữ cảnh. `src/core/config.py` phụ trách đọc YAML và **xử lý các đường dẫn tham chiếu (reference resolution)** nói chung.
3. Các hồ sơ cấu hình (profile) dữ liệu, phương pháp và đánh giá chứa những thiết lập có thể dùng lại. Cell chọn profile và cung cấp thiết lập riêng của run. Hệ thống không tự tạo mọi tổ hợp từ các danh sách.
4. Profile và cell khai báo trường rõ ràng. Không tự ý sao chép, đổi tên hoặc bỏ các thiết lập nghiên cứu khi chưa xác nhận ý nghĩa.
5. Offline và online là các giai đoạn (phase). Thích ứng (adaptation) là hành vi tùy chọn của phương pháp. Các nhãn O0–O2 và A0–A2 chỉ dành cho MUSTARD.

## Luồng dự kiến

```mermaid
flowchart LR
    A[Experiment YAML: liệt kê các cell] --> B[Đọc YAML và các tham chiếu]
    B --> C[Kết hợp từng cell với các profile]
    C --> D[Kiểm tra đầu vào bằng bốn protocol]
    D --> E[Mỗi cell tạo một cấu hình run hoàn chỉnh]
    E --> F[Chạy giai đoạn offline hoặc online]
    F --> G[Kiểm tra đầu ra tại các ranh giới xử lý]
```

`config.py` không chạy experiment và không kiểm tra những đầu ra chưa được tạo. Khi chương trình chạy, các thành phần xử lý sẽ gọi quy tắc của protocol phù hợp tại ranh giới đầu vào hoặc đầu ra.

## Giao diện dự kiến của `src/core/config.py`

`Config` bên dưới là bí danh kiểu (type alias) của `dict[str, Any]`, không phải một hệ thống lớp mới. Đường dẫn dùng `pathlib.Path`. Các hàm này chỉ là đề xuất, chưa khẳng định đã có trong mã hiện tại.

```python
from pathlib import Path
from typing import Any

Config = dict[str, Any]

def load_yaml(path: Path) -> Config: ...

def load_experiment(path: Path) -> Config: ...

def resolve_cell(cell: Config, *, configs_root: Path) -> Config: ...

def resolve_experiment(path: Path, *, configs_root: Path) -> list[Config]: ...

def load_standalone_run(path: Path, *, configs_root: Path) -> Config: ...
```

1. `load_yaml` đọc YAML mã hóa UTF-8 bằng bộ đọc an toàn (safe loader). Hàm từ chối tệp không tồn tại, khóa trùng, tài liệu rỗng hoặc nội dung cấp gốc không phải ánh xạ khóa–giá trị (mapping). Thông báo lỗi nêu tên tệp và trường gây lỗi khi có thể.
2. `load_experiment` đọc định nghĩa experiment có `experiment_id` và danh sách `cells` không rỗng. Hàm chỉ kiểm tra cấu trúc bên ngoài này. Theo đề xuất, experiment có một cell bị từ chối vì định nghĩa đã thống nhất yêu cầu nhiều run; một lượt thực thi riêng dùng `load_standalone_run`.
3. `resolve_cell` đọc các profile mà cell chọn trong `configs_root`, kết hợp chúng với cell và trả về cấu hình đầy đủ cho một run. Protocol run và evaluation kiểm tra danh tính run và phase; protocol data và stream kiểm tra dữ liệu; protocol scoring và calibration kiểm tra thiết lập đánh giá áp dụng được; protocol method updates kiểm tra thiết lập cập nhật áp dụng được. Chỉ gọi quy tắc khi đã có đầu vào cần thiết. Đầu ra được kiểm tra sau tại ranh giới xử lý tương ứng.
4. `resolve_experiment` đọc experiment và xử lý các cell theo đúng thứ tự liệt kê. Hàm từ chối `run_id` trùng và trả về một danh sách. Hàm không thực thi run, không tự mở rộng ma trận và không ghi tệp.
5. `load_standalone_run` đọc một tệp YAML có cấu trúc của một cell rồi gọi `resolve_cell`. Hàm không tạo experiment.
6. Các đường dẫn `data_profile`, `method_profile` và `evaluation_profile` được tính tương đối từ `configs_root`. Đường dẫn tuyệt đối hoặc đi ra ngoài thư mục gốc này bị từ chối. Bên gọi phải truyền `configs_root` rõ ràng để không phải tìm theo thư mục làm việc hiện tại.
7. Hàm xử lý cấu hình giữ nội dung profile trong `data`, `method` và `evaluation`; đồng thời thêm các trường `run_id`, `entity`, `seed`, `phase`, `device`, `output_dir`, `limits` và `offline_run` từ cell. Đường dẫn profile được lưu trong `profile_paths` để biết nguồn gốc (provenance). Hàm không thay đổi dữ liệu đầu vào.
8. Thiết kế không có cơ chế ghi đè lồng nhau tổng quát (general deep overrides). Profile sở hữu các giá trị dùng chung; cell chỉ cung cấp các trường riêng đã được định nghĩa cho run. Nếu cần một bộ giá trị dùng chung khác, tạo profile khác. Nếu sau này cần ghi đè profile, phải quy định riêng thứ tự ưu tiên và nguồn gốc giá trị.
9. Báo `FileNotFoundError` khi thiếu tệp tham chiếu; báo `ValueError` khi YAML sai, đường dẫn không hợp lệ hoặc cấu trúc bên ngoài sai. Các bộ kiểm tra protocol báo lỗi thiết lập không hợp lệ theo ngữ cảnh. Không tự chọn profile thay thế hoặc tự thêm giá trị mặc định cho trường bắt buộc.

## Các trường YAML dự kiến

**Bắt buộc (required)** nghĩa là phải có trong tệp. **Tùy chọn (optional)** nghĩa là có thể bỏ, với ý nghĩa khi bỏ được ghi rõ; không có giá trị mặc định ngầm khác. Kiểu dữ liệu mô tả giá trị trong YAML. Tất cả tên và ví dụ bên dưới đều là đề xuất.

### 1. `configs/data/smd.yaml`

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `dataset` | chuỗi (string) | bắt buộc | Tên tập dữ liệu: `smd`. |
| `root` | chuỗi | bắt buộc | Vị trí dữ liệu, tương đối với thư mục gốc dự án. |
| `validation_split_ratio` | số (number) | bắt buộc | Tỷ lệ dành cho tập validation. |
| `scaler` | chuỗi | bắt buộc | Bộ chuẩn hóa; chỉ fit trên dữ liệu train. Tên hỗ trợ cụ thể chưa được xác định. |
| `window_size` | số nguyên (integer) | bắt buộc | Độ dài cửa sổ dùng chung. |
| `train_stride`, `validation_stride`, `test_stride` | số nguyên | bắt buộc | Bước trượt cửa sổ của từng tập. |
| `batch_size` | số nguyên | bắt buộc | Kích thước batch cho offline. |
| `num_workers` | số nguyên | bắt buộc | Số worker của bộ nạp dữ liệu. |
| `shuffle_train` | đúng/sai (boolean) | bắt buộc | Có xáo trộn các cửa sổ train hay không. |
| `synthetic_profile` | chuỗi hoặc null | tùy chọn | Đường dẫn tương đối với `configs_root`; bỏ trống hoặc null nghĩa là không dùng profile dữ liệu tổng hợp. |

Các tệp trong `configs/data/synthetic_profiles/` chứa `min_segment_fraction`, `max_segment_fraction`, `spike_scale`, `anomaly_visibility_boost` và mapping `family_intensity` khi cần. Những trường cụ thể của từng nhóm bất thường và việc phương pháp nào dùng chúng cần được xem xét riêng. Cell chọn thực thể (entity) và khoảng luồng dùng cho chẩn đoán nếu có.

### 2. `configs/methods/mustard_offline.yaml`

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `method` | chuỗi | bắt buộc | `mustard`. |
| `phase` | chuỗi | bắt buộc | `offline`. |
| `offline_variant` | chuỗi | bắt buộc | O0, O1 hoặc O2 của MUSTARD; bằng chứng chạy O2 còn chưa rõ. |
| `model` | mapping | bắt buộc | Cấu hình kiến trúc, truy xuất (retrieval) và các đầu ra của mô hình (heads). |
| `loss` | mapping | bắt buộc | Các thành phần hàm mất mát, trọng số và tùy chọn riêng. |
| `training` | mapping | bắt buộc | Các giai đoạn huấn luyện, số epoch, bộ tối ưu và cách chọn checkpoint. |

`model`, `loss` và `training` chỉ xác định nhóm chịu trách nhiệm, chưa phải cấu trúc (schema) hoàn chỉnh. Cần đối chiếu nghiên cứu về phương pháp để xác định trường con, giá trị mặc định, khác biệt giữa các biến thể và quy ước checkpoint trước khi triển khai mà vẫn giữ đúng ý nghĩa khoa học. Chỉ `data.window_size` sở hữu kích thước cửa sổ; phương pháp sử dụng giá trị đó, không cung cấp giá trị khác gây xung đột.

### 3. `configs/methods/mustard_online.yaml`

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `method` | chuỗi | bắt buộc | `mustard`. |
| `phase` | chuỗi | bắt buộc | `online`. |
| `online_variant` | chuỗi | bắt buộc | A0, A1 hoặc A2 của MUSTARD. |
| `projector` | mapping | bắt buộc | Kiến trúc và tùy chọn của projector. |
| `update` | mapping | bắt buộc | Chính sách cập nhật, bộ tối ưu, loss và nhóm tham số được cập nhật khi áp dụng. |
| `verification` | mapping | tùy chọn | Thiết lập xác minh khi biến thể đã chọn hỗ trợ. |

A0 không được mặc nhiên có bộ tối ưu để cập nhật. Tính điểm diễn ra trước cập nhật. Trường con cụ thể và tùy chọn nào áp dụng cho từng biến thể vẫn cần xem xét trong quy ước của phương pháp (method contract).

### 4. `configs/methods/baselines/<method>.yaml`

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `method` | chuỗi | bắt buộc | Tên một phương pháp baseline được giữ lại. |
| `phase` | chuỗi | bắt buộc | `offline` hoặc `online`; dùng profile riêng nếu thiết lập khác nhau. |
| `parameters` | mapping | bắt buộc | Chỉ gồm thiết lập mà phương pháp đó chấp nhận. |

Không gán mã biến thể MUSTARD cho baseline. Cần xem xét riêng schema `parameters`, nhu cầu checkpoint và các phase được hỗ trợ của từng phương pháp, bao gồm MOMENT. Không giả định mọi baseline đều có phase online hoặc đều cần profile khi không có thiết lập dùng chung.

### 5. `configs/evaluation_profiles/smd/<profile>.yaml`

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `score_space` | chuỗi | bắt buộc | Không gian đầu vào dùng để tính điểm. |
| `point_score_transform` | chuỗi | bắt buộc | Phép biến đổi điểm tại từng điểm thời gian. |
| `offline_threshold_split`, `online_threshold_split` | chuỗi | bắt buộc | Nguồn hiệu chỉnh (calibration) cho từng phase. |
| `offline_threshold_quantile`, `online_threshold_quantile` | số | bắt buộc | Phân vị (quantile) chọn ngưỡng của từng phase. |
| `offline_tail_policy` | chuỗi | bắt buộc | Cách xử lý cửa sổ cuối không đủ độ dài trong offline. |
| `online_window_stride` | số nguyên | bắt buộc | Bước trượt cửa sổ của luồng online. |
| `ewma_current_weight`, `ewma_previous_weight` | số | bắt buộc | Trọng số làm mượt điểm online bằng EWMA. |
| `window_quantile`, `latent_low_quantile`, `latent_high_quantile` | số | bắt buộc khi bước phân loại xử lý dùng đến | Các phân vị hiệu chỉnh cho bước phân loại xử lý (triage). |
| `test_label_usage` | chuỗi | bắt buộc | `metrics_only`. |
| `point_adjustment` | boolean | bắt buộc | `false`. |

Năm profile trong cây đã đề xuất giữ các lựa chọn tính điểm và hiệu chỉnh riêng. Protocol scoring và calibration kiểm tra các ràng buộc giữa trường, bao gồm nguồn cần dùng khi tính điểm trên đầu vào đã chuẩn hóa. Trường nào không cần cho từng baseline hoặc profile chỉ offline vẫn chưa được quyết định; không bắt buộc thiết lập online cho phương pháp không dùng được chúng trước khi xác định điều kiện áp dụng theo phase. `window_size` thuộc dữ liệu và không lặp lại tại đây.

### 6. `configs/experiments/<name>.yaml`

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `experiment_id` | chuỗi | bắt buộc | Danh tính của phép so sánh dự kiến. |
| `cells` | danh sách mapping | bắt buộc | Ít nhất hai cell được liệt kê rõ theo thứ tự thực thi. |

Mỗi cell có các trường sau:

| Trường | Kiểu | Trạng thái | Ý nghĩa |
| --- | --- | --- | --- |
| `run_id` | chuỗi | bắt buộc | Danh tính run, duy nhất trong experiment. |
| `data_profile`, `method_profile` | chuỗi | bắt buộc | Đường dẫn profile tương đối với thư mục gốc cấu hình. |
| `evaluation_profile` | chuỗi hoặc null | bắt buộc | Đường dẫn tương đối với thư mục gốc cấu hình; chỉ null khi run không đánh giá hoặc hiệu chỉnh. |
| `entity` | chuỗi | bắt buộc | Thực thể do dataset định nghĩa. |
| `seed` | số nguyên | bắt buộc | Seed của run. |
| `phase` | chuỗi | bắt buộc | `offline` hoặc `online`; phải khớp profile phương pháp. |
| `device` | chuỗi | bắt buộc | Thiết bị thực thi. |
| `output_dir` | chuỗi | bắt buộc | Nơi lưu đầu ra run, tương đối với thư mục gốc dự án. |
| `offline_run` | mapping hoặc null | bắt buộc với cell online | Danh tính run trước đó và vị trí tệp kết quả khi cần; chỉ null với phương pháp không phụ thuộc run offline. |
| `limits` | mapping | tùy chọn | Giới hạn chạy và khoảng luồng nếu cần; bỏ trường này nghĩa là không thêm giới hạn. |

Khi có `offline_run`, bên trong gồm `run_id` và `artifact_dir`, đều là chuỗi. Phải xác định rõ cách chọn checkpoint và tệp hiệu chỉnh trước khi triển khai; chỉ có tên thư mục không được dẫn đến việc chọn checkpoint tùy ý. `limits` có thể chứa `max_train_windows`, `max_validation_windows`, `max_test_windows`, `max_online_steps` là số nguyên dương; `stream_start` và `stream_end` là hai số nguyên đi cùng nhau, xác định khoảng nửa kín (half-open range). Các giới hạn phải được ghi lại trong đầu ra run. Run đầy đủ không giới hạn thì bỏ `limits`.

`thesis.yaml`, `baselines.yaml`, `smoke.yaml`, `diagnostics.yaml` và `demo.yaml` chỉ dùng cấu trúc experiment này khi mỗi tệp đại diện cho một phép so sánh nhiều run. Demo hoặc chẩn đoán chỉ có một run dùng tệp cell độc lập; vị trí chính xác của tệp đó trong cây vẫn chưa được quyết định.

## Ví dụ kết nối các cấu hình

```yaml
# configs/experiments/example.yaml
experiment_id: mustard-comparison
cells:
  - run_id: machine-1-6-o0-seed6
    data_profile: data/smd.yaml
    method_profile: methods/mustard_o0_offline.yaml
    evaluation_profile: evaluation_profiles/smd/cleanval_q99_ewma09.yaml
    entity: machine-1-6
    seed: 6
    phase: offline
    device: cpu
    output_dir: outputs/example/smd/machine-1-6/seed6/mustard-o0/offline
  - run_id: machine-1-6-o1-seed6
    data_profile: data/smd.yaml
    method_profile: methods/mustard_o1_offline.yaml
    evaluation_profile: evaluation_profiles/smd/cleanval_q99_ewma09.yaml
    entity: machine-1-6
    seed: 6
    phase: offline
    device: cpu
    output_dir: outputs/example/smd/machine-1-6/seed6/mustard-o1/offline
```

Hai tên tệp phương pháp chỉ minh họa profile biến thể được đề xuất; chúng chưa xuất hiện trong cây chi tiết trước đó. Mỗi profile sẽ khai báo rõ `offline_variant`. Bộ xử lý cấu hình tạo hai mapping cho hai run, mỗi mapping gồm `data`, `method`, `evaluation`, các trường của cell và `profile_paths`. Kế hoạch này không tạo YAML có thể chạy.

## Thuật ngữ và chuyển đổi

1. Trong loader đã kiểm tra, tệp hiện tại được gọi là experiment config nhưng biểu diễn một run. Theo đề xuất, `experiment_id` đặt tên phép so sánh còn `run_id` đặt tên một run; không dùng thay thế cho nhau.
2. `data_config_path` hiện tại được đề xuất đổi thành `data_profile`; `model_config_path` thành `method_profile`; thiết lập tính điểm đang được nạp từ profile riêng chuyển thành `evaluation_profile`. Đây là ánh xạ đề xuất, không phải quy tắc chuyển đổi tự động.
3. Các trường task có thể thuộc dữ liệu, phương pháp hoặc cell. Cần xác định theo ý nghĩa và nơi sử dụng trước khi viết lại; giữ trạng thái chưa rõ cho đến khi có đủ bằng chứng.
4. Cần quyết định rõ cách chuyển đổi đường dẫn tệp kết quả, cách chọn checkpoint, tiếp tục chạy (resume) và danh tính đầu ra, cũng như cách kiểm tra tương đương (parity). Cho phép viết lại từ đầu không đồng nghĩa cho phép xóa kết quả hoặc các phép so sánh khoa học hiện tại.

## Các bước chuẩn bị triển khai theo thứ tự

Công cụ: Markdown, `rg`, bộ đọc YAML, Python và các lệnh kiểm tra hiện có của dự án. Đây là bản bàn giao thiết kế, chưa phải sự cho phép triển khai.

1. [ ] Xem xét và giải quyết các quyết định còn mở bên dưới.
2. [ ] Kiểm kê các phương pháp, biến thể, profile, điểm bắt đầu chương trình (entry point), checkpoint và nơi đọc tệp kết quả cần giữ lại.
3. [ ] Xác định đầy đủ trường con của từng phương pháp và profile dữ liệu tổng hợp mà không tự suy đoán ý nghĩa còn thiếu.
4. [ ] Xác định cách chọn checkpoint, tệp hiệu chỉnh và vị trí tệp cho run độc lập.
5. [ ] Xác định trường đánh giá và quy tắc kiểm tra protocol nào áp dụng cho từng phase.
6. [ ] Xác định đầy đủ quy ước cấu hình run đã xử lý, giá trị mặc định, thông báo lỗi và nguồn gốc profile.
7. [ ] Ánh xạ từng cấu hình và quy ước tệp kết quả cũ cần giữ sang vị trí mới, hoặc đánh dấu chưa giải quyết.
8. [ ] Chuẩn bị cách thay thế và khôi phục an toàn trước khi đổi loader hoặc profile hiện tại.
9. [ ] Chỉ triển khai loader và YAML mới sau khi thiết kế và kế hoạch chuyển đổi được duyệt.
10. [ ] Kiểm chứng các luồng đại diện: run độc lập, offline, online, baseline, resume và calibration bằng đầu ra tham chiếu đã thống nhất trước khi bỏ đường dẫn cũ.

## Các quyết định còn mở

1. Chưa xác định đủ schema trường con của MUSTARD, từng baseline và từng nhóm bất thường tổng hợp. Các bảng trên chưa phải đặc tả triển khai hoàn chỉnh.
2. Xác nhận cách `offline_run` chỉ đúng checkpoint và tệp ngưỡng, giữ nguồn gốc và tránh chọn tự động mơ hồ.
3. Quyết định vị trí YAML cho run độc lập và có cần liệt kê profile riêng cho từng biến thể trong cây hay không.
4. Xác định chính xác trường đánh giá bắt buộc theo phase và phương pháp, gồm cả phương pháp không cần calibration.
5. Quyết định ghi rõ đường dẫn đầu ra như đề xuất hay tạo từ danh tính run; đối chiếu đường dẫn kết quả hiện tại trước khi triển khai.
6. Cây trước đó gọi cấu hình experiment là “experiment set”, chưa khớp với định nghĩa đã thống nhất: một experiment tự nó gồm nhiều run. Kế hoạch này dùng `experiment_id` và không thêm cấp experiment set.

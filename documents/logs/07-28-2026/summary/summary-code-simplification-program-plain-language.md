# Tóm tắt chương trình tối giản hóa codebase

## 1. Câu chuyện tổng quát

Codebase hiện có nhiều phần đã hoạt động, nhưng một số trách nhiệm bị lặp
lại hoặc nằm rải rác ở nhiều nơi. Cùng một khái niệm có thể xuất hiện ở một
facade công khai, một module triển khai bên trong, các helper và một đường dẫn
legacy (đường dẫn cũ để giữ tương thích). Điều này làm người đọc khó biết nơi
nào là nơi sở hữu logic chính.

Mục tiêu của chương trình là làm code dễ đọc và dễ bảo trì hơn mà không đổi
hành vi của hệ thống. Chương trình không xem việc giảm số dòng code là mục
tiêu chính. Một thay đổi chỉ có ý nghĩa khi nó làm rõ nơi sở hữu logic, giảm
số đường chạy trùng nhau hoặc làm cho luồng chính dễ theo dõi hơn.

Kế hoạch chia công việc thành 33 phase (pha). Mỗi phase xử lý đúng một
khu vực. Bảy phase đầu có độ ưu tiên cao nhất và tạo thành đợt triển khai đầu
tiên. Các phase sau phụ thuộc vào kết quả của những phase trước.

Mỗi phase đi qua năm stage (giai đoạn):

1. **Dựng hàng rào hành vi (behavioral fence):** ghi lại đầu vào, đầu ra,
   thứ tự xử lý, tác dụng phụ, lỗi, caller (nơi gọi) và test liên quan.
2. **Lập bản đồ sở hữu (ownership map):** xác định implementation (phần
   triển khai) chính, consumer tương thích, logic bị lặp và ranh giới chỉnh
   sửa nhỏ nhất.
3. **Tối giản từng phần:** thực hiện một thay đổi nhỏ, có thể kiểm tra và
   không đổi hành vi.
4. **Kiểm tra tương đương (parity verification):** chạy test tập trung,
   kiểm tra import/compile/lint và một smoke flow nhỏ phù hợp.
5. **Đóng phase:** ghi lại owner cuối cùng, interface được giữ nguyên, rủi ro
   còn lại và điều kiện để bắt đầu phase tiếp theo.

Nói ngắn gọn, quy trình đi theo chuỗi sau:

```text
Hiểu hành vi hiện tại
        ↓
Xác định nơi sở hữu logic
        ↓
Đổi một phần nhỏ
        ↓
So sánh trước và sau
        ↓
Ghi nhận kết quả rồi mới sang phase tiếp theo
```

## 2. Những hợp đồng phải giữ nguyên

Trong tài liệu này, “hợp đồng” là những điều mà các module khác đang dựa vào.
Code có thể được sắp xếp lại, nhưng các hợp đồng sau phải giữ nguyên:

- Batch dữ liệu vẫn là dictionary. Tensor đầu vào vẫn có dạng
  `x=[B,L,D]`, trong đó `B` là số mẫu, `L` là độ dài cửa sổ và `D` là số đặc
  trưng.
- Encoder vẫn trả hidden state có dạng `hidden=[B,L,H]`, trong đó `H` là số
  chiều biểu diễn ẩn.
- Thesis model vẫn trả reconstruction, classification, point-score và các
  trường auxiliary. Các trường này có thể bao gồm uncertainty và thông tin
  chẩn đoán.
- Thứ tự update khi train và khi chạy online vẫn giữ nguyên. Phase gate,
  nhóm parameter của optimizer và state trong checkpoint cũng phải giữ
  nguyên.
- Threshold, nguồn gốc artifact, output path, tên thí nghiệm, tên field và
  alias tương thích vẫn giữ nguyên nếu phase chưa chứng minh được rằng đường
  cũ đã không còn người dùng.
- Luồng online vẫn giữ tính nhân quả (causal): hệ thống không được dùng dữ
  liệu tương lai khi xử lý cửa sổ hiện tại.
- Luồng benchmark offline hai stage, vai trò của checkpoint và provenance
  (thông tin cho biết artifact được tạo từ đâu) vẫn giữ nguyên.

## 3. Bảy phase ưu tiên cao nhất

### Phase 01 — CFG-01: Configuration orchestration

Configuration orchestration là luồng điều phối việc đọc và chuẩn bị config.
Mục tiêu là làm cho luồng `load_experiment_config(path)` dễ đọc hơn. Luồng
này vẫn phải giữ thứ tự resolve path, merge section, normalize alias, resolve
window, validate, ghi log và xử lý lỗi.

Em sẽ không tạo một config framework mới. Em chỉ tách những trách nhiệm đã rõ
thành các helper có tên cụ thể, rồi giữ một hàm điều phối tuyến tính.

```python
def load_experiment_config(path):
    raw = read_config(path)
    resolved = resolve_references(raw)
    merged = merge_sections(resolved)
    normalized = normalize_aliases(merged)
    validate_experiment_config(normalized)
    return normalized
```

### Phase 02 — CFG-02: Configuration validation ownership

Một số luật kiểm tra config đang nằm ở nhiều module. Phase này phân loại từng
luật thành kiểm tra schema, kiểm tra liên kết giữa các section, kiểm tra runtime,
kiểm tra alias hoặc hành vi logging. Sau đó mỗi luật có một owner rõ ràng.

Hệ thống vẫn phải từ chối config sai, giữ alias cũ và giữ loại lỗi cũng như thứ
tự báo lỗi nếu caller đang phụ thuộc vào chúng.

### Phase 03 — MOD-01: Thesis model lifecycle và mixin graph

Thesis model có constructor, `forward`, `training_step`, phase hook, memory,
serialization và checkpoint loading. Các trách nhiệm này hiện đi qua nhiều
mixin và helper nên khó theo dõi.

Phase này lần theo toàn bộ vòng đời model từ `ThesisMultitaskModel`. Sau đó,
phase làm rõ ownership bằng composition hoặc helper được đặt tên rõ ràng nếu
việc đó vẫn giữ method order, `super()`, parameter, buffer, attribute và
state-dict key.

Không được xóa mixin chỉ vì muốn rút ngắn cây kế thừa. Trước hết phải chứng
minh mixin đó không phải là public contract độc lập.

### Phase 04 — EVAL-01: Trainer/evaluator metric ownership

Trainer và evaluator có một số logic gần giống nhau về window stitching,
threshold, point-score, VUS-PR, uncertainty và output key. Phase này so sánh
logic ở mức hợp đồng, không chỉ so sánh tên hàm.

Chỉ phần giống nhau thật sự mới được đưa vào một helper chung. Logic riêng của
trainer, chẳng hạn checkpoint monitoring, và logic riêng của evaluator, chẳng
hạn report assembly, vẫn ở nơi phù hợp.

Rủi ro lớn nhất là đổi nguồn threshold hoặc làm metric bị tính trùng.

### Phase 05 — ONLINE-01: Canonical online runtime path

Repository có nhiều đường dẫn liên quan đến online TTA, `online_loop`, online
adaptation và baseline online. Trước khi gộp hoặc xóa, phase này lập ma trận
caller, config và test.

Mỗi đường dẫn được đánh dấu là active, compatibility hoặc unverified. Đường
dẫn unverified không bị xóa. Chỉ routing bị trùng và đã được xác nhận mới được
tối giản.

Các test quan trọng phải kiểm tra A0/A1/A2, thứ tự update, state replay,
calibration, giới hạn số bước và tính toàn vẹn của artifact.

### Phase 06 — RUNTIME-01: Runtime registration lifecycle

Phase này làm rõ nơi dataset và model được register, clear và build trong các
luồng offline, evaluation, online, benchmark và test.

Registry hiện tại chỉ được giữ như một ranh giới khởi tạo thật sự cần thiết.
Không tạo thêm factory layer nếu nó không thêm context mới. Chỉ xóa wrapper
không có trách nhiệm riêng sau khi test import và re-registration pass.

### Phase 07 — COMPAT-01: Facade và compatibility boundaries

Phase này xử lý facade, wildcard export, alias tên baseline, script wrapper,
`sys.modules` replacement và các điểm monkeypatch.

Mỗi facade phải có một implementation owner và một lý do tương thích được ghi
rõ. Thay đổi được thực hiện từng adapter nhỏ, đồng thời giữ import identity
và hành vi CLI.

## 4. Hai mươi sáu phase còn lại

Các phase dưới đây có ưu tiên thấp hơn vì chúng phụ thuộc vào bảy phase đầu
hoặc vào các hợp đồng đã được làm rõ trước đó.

| Phase | Mã | Khu vực cần tối giản |
|---:|---|---|
| 08 | CFG-03 | Quyền sở hữu field config và alias |
| 09 | CFG-04 | Khác biệt giữa default công khai và override benchmark |
| 10 | MOD-02 | Routing trong `forward` của thesis model |
| 11 | MOD-03 | Vòng đời state và memory của thesis model |
| 12 | MOD-04 | Vòng đời loss và training step |
| 13 | EVAL-02 | Hợp đồng threshold, uncertainty và artifact đánh giá |
| 14 | ONLINE-02 | Context của online runtime |
| 15 | ONLINE-03 | Dispatch giữa các biến thể online |
| 16 | ONLINE-04 | Quyền sở hữu stream, cursor, batcher và persistence |
| 17 | DATA-01 | API dữ liệu công khai bị lặp |
| 18 | DATA-02 | Post-processing sau khi parser dataset chạy |
| 19 | AUG-01 | Owner của synthetic anomaly injector |
| 20 | RED-01 | Owner của RedLamp baseline và delegator |
| 21 | RUN-01 | Thesis offline benchmark runner |
| 22 | RUN-02 | Script tạo benchmark config |
| 23 | RUN-03 | Điều phối command theo stage và variant |
| 24 | REPORT-01 | Trích xuất dữ liệu cho report chỉ đọc |
| 25 | REPORT-02 | Tách re-evaluation khỏi thao tác pruning |
| 26 | CFG-05 | Ma trận config được sinh tự động |
| 27 | MODEL-05 | Quyền sở hữu component và config của thesis model |
| 28 | EVAL-03 | Adapter cho metric toán học |
| 29 | CLI-01 | Namespace của comparative runner |
| 30 | COMPAT-02 | Legacy alias và CLI flag |
| 31 | STATIC-01 | Dọn import và cải thiện readability |
| 32 | DEMO-01 | Demo và replay entrypoint |
| 33 | DOC-01 | Path tài liệu và thuật ngữ bị lệch |

Các phase 24 và 25 cần chú ý đặc biệt. Reporting phải giữ schema và provenance.
Pruning là hành động có thể xóa artifact, nên trước hết chỉ được lập manifest
và chạy dry run. Chỉ thực hiện xóa khi có yêu cầu rõ ràng với target cụ thể.

## 5. Cách đánh giá một phase đã hoàn thành

Một phase chỉ được xem là hoàn thành khi:

- test tập trung pass mà không cần làm yếu test;
- `.venv/bin/python` pass các kiểm tra compile/import cần thiết;
- không có Ruff finding mới;
- smoke flow liên quan pass;
- diff chỉ nằm trong phạm vi phase;
- so sánh trước và sau cho thấy contract và artifact không đổi;
- các lỗi có sẵn trước khi sửa được ghi riêng, không gán nhầm cho refactor.

Mốc test đã được ghi nhận trước đó là `442 passed, 1 skipped, 10 failed`.
Trước mỗi wave triển khai, cần chạy lại baseline để xác nhận mốc này còn đúng.
Mười lỗi có sẵn phải được tách khỏi lỗi do thay đổi mới.

```text
baseline trước sửa
        ↓
thay đổi nhỏ trong đúng phase
        ↓
test tập trung
        ↓
compile/import/lint
        ↓
smoke flow
        ↓
so sánh output, state, artifact và lỗi
```

Không được sửa test chỉ để test chấp nhận một hành vi mới. Nếu chưa hiểu rõ
hành vi cũ, phase phải dừng ở bước nghiên cứu.

## 6. Những quyết định chưa được chốt

Kế hoạch sơ bộ chưa tự quyết định các vấn đề sau:

- `max_online_steps: null` có phải là giá trị có chủ ý hay không;
- implementation online nào là canonical cho từng nhóm config;
- facade legacy có còn consumer bên ngoài hay không;
- mixin nào có thể xóa mà không làm đổi contract;
- path và alias nào thực sự đã chết.

Các câu hỏi này cần được kiểm tra từ code đang chạy, config hiện hành, test,
caller và git history khi bắt đầu từng phase. Không nên đoán dựa trên tên file.

## 7. Tóm tắt bằng mã giả

```python
for phase in phases_in_priority_order:
    baseline = capture_behavior(phase)
    ownership = map_logic_owners(phase)

    if not behavior_is_understood(baseline, ownership):
        stop_and_record_uncertainty(phase)
        continue

    change = make_one_small_refactor(phase, ownership)
    result = run_focused_tests_and_smoke(phase)

    if result.breaks_protected_contract:
        revert_or_rework(change)
        record_blocker(phase, result)
    else:
        compare_before_after(baseline, result)
        close_phase(phase)
```

Mã giả trên nhấn mạnh ba điều: hiểu trước khi sửa, mỗi lần chỉ sửa một phần
nhỏ, và chỉ đóng phase sau khi có bằng chứng trước/sau.

## 8. Câu hỏi MCQ để tự kiểm tra

Mỗi câu có một đáp án đúng. Các phương án sai được viết gần giống đáp án đúng
để kiểm tra xem người đọc có hiểu điều kiện và lý do của kế hoạch hay không.

### Câu 1

Mục tiêu chính của chương trình tối giản hóa codebase là gì?

A. Giảm số dòng code nhiều nhất có thể.
B. Chia mọi file lớn thành nhiều file nhỏ.
C. Làm rõ ownership và giảm đường chạy trùng nhưng giữ nguyên hành vi.
D. Thay toàn bộ kiến trúc cũ bằng framework mới.

### Câu 2

Trong Phase 01, vì sao chưa nên tạo một config framework mới?

A. Vì config không cần validation.
B. Vì mục tiêu là làm rõ luồng loader hiện tại, không tạo thêm đường xử lý.
C. Vì chỉ benchmark mới dùng config.
D. Vì helper không được phép tồn tại trong codebase.

### Câu 3

Một validator kiểm tra cả kiểu dữ liệu, alias và quan hệ giữa hai section. Bước
nào phù hợp nhất trước khi chỉnh sửa validator?

A. Xóa các kiểm tra trùng tên.
B. Chuyển toàn bộ kiểm tra vào một class mới.
C. Lập rule inventory và phân loại từng kiểm tra theo trách nhiệm.
D. Sửa test để chấp nhận mọi config hiện tại.

### Câu 4

Khi đơn giản hóa mixin graph của thesis model, hành động nào nguy hiểm nhất?

A. Ghi lại thứ tự constructor và phase hook.
B. Kiểm tra state-dict key và registered buffer.
C. Dùng composition nếu vẫn giữ method order và state contract.
D. Xóa mixin chỉ vì cây kế thừa dài.

### Câu 5

Trainer và evaluator cùng có đoạn code tính threshold. Khi nào có thể đưa đoạn
code đó vào helper chung?

A. Khi hai đoạn có tên hàm giống nhau.
B. Khi helper mới làm giảm số dòng code.
C. Khi hai đoạn có cùng hành vi, cùng nguồn dữ liệu và cùng contract đầu ra.
D. Khi trainer không còn cần checkpoint monitoring.

### Câu 6

Trong Phase 05, một online path chưa có test và chưa có config active nào dùng.
Kế hoạch yêu cầu xử lý path này thế nào?

A. Xóa ngay vì không có test.
B. Đổi tên thành canonical path.
C. Đánh dấu unverified và chưa xóa.
D. Gộp vào path có nhiều dòng code hơn.

### Câu 7

Vì sao phải giữ thứ tự update trong online A0/A1/A2?

A. Vì thứ tự update chỉ ảnh hưởng log.
B. Vì thay đổi thứ tự có thể làm thay đổi tính nhân quả và kết quả thích nghi.
C. Vì mọi model đều phải dùng cùng một thứ tự.
D. Vì checkpoint không lưu state.

### Câu 8

Một wrapper registry chỉ gọi lại hàm đăng ký hiện tại và không thêm context.
Khi nào có thể xem xét loại wrapper đó?

A. Ngay khi wrapper làm code dài hơn.
B. Sau khi test import, clear và re-registration chứng minh contract vẫn giữ.
C. Chỉ sau khi xóa toàn bộ registry.
D. Chỉ khi đổi tên dataset và model.

### Câu 9

Một benchmark generator đang tạo filename, output path và experiment identity.
Khi tối giản generator, điều gì phải được bảo vệ?

A. Chỉ nội dung metric cuối cùng.
B. Chỉ tên biến trong Python.
C. Filename, path, key, identity và semantics của experiment.
D. Không cần bảo vệ gì nếu test vẫn pass.

### Câu 10

Phase REPORT-02 liên quan đến re-evaluation và pruning. Cách làm an toàn nhất là
gì?

A. Re-evaluate xong thì xóa ngay mọi artifact không được dùng.
B. Trộn lập kế hoạch xóa vào script report để giảm số file.
C. Tách phần audit/manifest khỏi phần xóa và chạy dry run trước.
D. Xóa toàn bộ output cũ rồi chạy lại benchmark.

### Câu 11

Nếu baseline test có lỗi trước khi bắt đầu refactor, nhóm nên làm gì?

A. Sửa test trước để toàn bộ test pass.
B. Ghi nhận lỗi baseline và tách nó khỏi lỗi do refactor.
C. Bỏ qua toàn bộ test liên quan.
D. Gán lỗi đó cho thay đổi mới để dễ theo dõi.

### Câu 12

Vì sao bảy phase đầu được làm trước 26 phase còn lại?

A. Vì bảy phase đầu luôn dễ hơn mọi phase khác.
B. Vì bảy phase đầu làm rõ contract và ownership mà nhiều phase sau phụ thuộc.
C. Vì các phase sau không cần test.
D. Vì chỉ bảy phase đầu ảnh hưởng runtime.

## 9. Đáp án và giải thích ngắn

1. **C.** Chương trình ưu tiên làm rõ ownership và bỏ đường chạy trùng, nhưng
   không đổi hành vi.
2. **B.** Phase 01 chỉ làm cho loader hiện tại dễ đọc và dễ kiểm tra hơn.
3. **C.** Phải biết mỗi rule thuộc trách nhiệm nào trước khi gộp hoặc tách.
4. **D.** Cây kế thừa dài không tự chứng minh rằng mixin đã không còn cần thiết.
5. **C.** Tên hàm hoặc số dòng không đủ chứng minh hai logic có thể dùng chung.
6. **C.** Chưa có bằng chứng thì phải giữ path và đánh dấu trạng thái chưa xác
   minh.
7. **B.** Thứ tự update có thể thay đổi dữ liệu được phép nhìn thấy và trạng
   thái model.
8. **B.** Test phải chứng minh wrapper không giữ một contract ẩn trước khi xóa.
9. **C.** Các giá trị này quyết định nơi lưu, cách nhận diện và cách diễn giải
   kết quả benchmark.
10. **C.** Manifest và dry run giúp kiểm tra target trước khi có hành động xóa.
11. **B.** Lỗi có sẵn cần được ghi riêng để không kết luận sai về refactor.
12. **B.** Bảy phase đầu tạo nền tảng contract và ownership cho các phase sau.

## 10. Tài liệu nguồn

- Nghiên cứu các hotspot: `documents/logs/07-26-2026/research/research-code-simplification-hotspots.md`
- Tầm nhìn chương trình: `documents/logs/07-26-2026/plan/vision_code-simplification-program.md`
- Kế hoạch lập trình sơ bộ: `documents/logs/07-26-2026/plan/plan-code-simplification-program.md`
- Cấu trúc phase và stage: `documents/logs/07-26-2026/structure/structure-code-simplification-program.md`
- Chi tiết stage sơ bộ: `documents/logs/07-26-2026/detail/detail-code-simplification-program.md`
- Plain Language Guide được dùng để biên tập tài liệu này: `/Users/conquerormikrokosmos/.codex/attachments/fcb31a8f-c370-42c2-9c70-29a6da98c575/pasted-text.txt`

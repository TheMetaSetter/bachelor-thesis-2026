---
date: 2026-09-01
researcher: TheMetaSetter
repository: bachelor-thesis-2026
status: proposed-plan
baseline_revision: 0428f3742f6c08a6f8a1650325932972369acfdd
source_research: local pytest collection and codebase_preferences.md
external_research: Exa, 2026-09-01
---

# Kế hoạch rút gọn test suite theo mức độ quan trọng

## Quyết định chính

Phân loại ở mức **test case được Pytest thu thập** (`nodeid`), không phân loại
chỉ theo file. Với snapshot hiện tại, repo có 144 file test, 482 hàm test và
513 test case sau khi mở rộng parameterization. Vì 513 chia hết cho 3, mục tiêu
là ba nhóm bằng nhau:

| Nhóm | Số case | Trạng thái sau migration |
| --- | ---: | --- |
| `high` | 171 | Giữ trong `tests/` và luôn được thu thập |
| `medium` | 171 | Chuyển vào `tests_archive/` |
| `low` | 171 | Chuyển vào `tests_archive/` |

Đây là mục tiêu cho snapshot nêu trên. Khi số case thay đổi, dùng
`divmod(N, 3)`: chia đều trước, phần dư phân bổ lần lượt cho `high`, rồi
`medium`. Nếu tập test bắt buộc lớn hơn quota `high`, dừng migration và ghi
ngoại lệ; không hạ mức một test bảo vệ contract để đạt con số đẹp.

`tests_archive/` hiện chưa tồn tại, dù `tests/README.md` đã mô tả nó. Archive
là di chuyển có thể khôi phục bằng `git mv`, không phải xoá vĩnh viễn.

## Cơ sở chuyên gia

Workflow này tách ba việc thường bị trộn lẫn:

1. **Minimization:** bỏ test obsolete hoặc redundant khỏi active suite.
2. **Selection:** chọn test liên quan tới phần code/config vừa đổi.
3. **Prioritization:** chạy test có khả năng phát hiện lỗi sớm trước.

Phân biệt này theo survey của Yoo và Harman giúp archive là quyết định
minimization; nó không thay thế selection theo thay đổi hay thứ tự chạy.
[Yoo and Harman, *Regression testing minimization, selection and prioritization*](https://doi.org/10.1002/stvr.430)

ISTQB CTAL-TA yêu cầu đánh giá rủi ro theo likelihood và impact, duy trì
traceability từ test tới test basis, áp dụng ưu tiên xuyên suốt thiết kế và
thực thi, rồi đánh giá lại ở các test cycle sau. [ISTQB CTAL-TA v3.1.2
Syllabus](https://istqb-main-web-prod.s3.amazonaws.com/media/documents/ISTQB_CTAL-TA_Syllabus_v3.1.2.pdf)

Rothermel và cộng sự cho thấy coverage, coverage phần code chưa được kiểm tra
và khả năng phát hiện fault là các tín hiệu hợp lý để ưu tiên thứ tự chạy.
[Rothermel et al., *Prioritizing test cases for regression testing*](https://doi.org/10.1109/32.962562)

Mutation-based reduction xem test suite như bài toán set cover. Tuy nhiên,
nghiên cứu cũng cho thấy giảm số test có thể làm mất fault detection; vì vậy
mutation score chỉ là bước xác nhận bổ sung, không phải lý do tự động xoá test.
[Jatana et al., *Test Suite Reduction by Mutation Testing Mapped to Set Cover Problem*](https://doi.org/10.1145/2905055.2905094)

Một case study công nghiệp của Renault cũng liệt kê code coverage, code-change
coverage, criticality, similarity, cost và historical effectiveness là các
tiêu chí có thể dùng để so sánh test. [Meyer, Waeselynck and Cuesta, *A Case
Study on the “Jungle” Search for Industry-Relevant Regression Testing*](https://doi.org/10.1109/qrs60937.2023.00045)

Các nguồn trên là tài liệu peer-reviewed hoặc chuẩn nghề nghiệp. Exa đã trả về
56 kết quả qua 13 lượt tìm kiếm; note chỉ giữ các nguồn có nội dung trực tiếp
cho quyết định này. Các bài thuật toán mới không được dùng để suy ra ngưỡng
cho repo này.

## Scoring schema

Chấm mỗi `nodeid` bằng năm trường. Hai trường đầu mô tả risk chính; ba trường
sau mô tả giá trị bảo vệ và độ trùng lặp. Mỗi điểm phải có bằng chứng trong
source, config, test history hoặc tài liệu `documents/`.

| Trường | Khoảng | Điểm cao nghĩa là | Bằng chứng cần đọc |
| --- | ---: | --- | --- |
| `I` — impact | 1–5 | Fail làm sai kết quả nghiên cứu, causal contract, checkpoint, metric hoặc flow chính | `documents/spec/`, output schema, caller thật |
| `L` — likelihood | 1–5 | Code/config liên quan có churn, nhiều dependency, lịch sử failure hoặc logic phức tạp | `git log`, failure history, import/caller graph |
| `U` — uniqueness | 0–5 | Test là guard duy nhất cho một hành vi hoặc edge case | coverage/contract map và test assertions |
| `A` — active-path relevance | 0–5 | Test chạy trên public entrypoint, active config hoặc benchmark hiện hành | runner, config generator, `pytest --collect-only` |
| `O` — oracle strength | 0–5 | Assertion quyết định, deterministic, chỉ rõ output/state/artifact sai | assertion, fixture, expected schema |

Điểm tổng:

```text
R = I × L
S = 4R + 2U + A + O
```

`R` giữ risk likelihood × impact ở vị trí chính. `U`, `A` và `O` chỉ bổ sung
để ưu tiên test duy nhất, đang chạy thật và có oracle rõ. Không dùng số dòng,
thời gian chạy hoặc tên thư mục làm điểm importance trực tiếp; chúng chỉ được
ghi lại để tối ưu execution order sau này.

### Anchor chấm điểm

#### Impact `I`

- `5`: sai causal online score/threshold, checkpoint contract, metric chính,
  split hoặc protocol làm kết quả luận văn không đáng tin.
- `4`: sai shape/batch, model forward/loss, loader, state hoặc artifact schema
  dùng bởi nhiều workflow.
- `3`: sai một public runner, config generator, baseline hoặc reporting flow.
- `2`: sai helper chẩn đoán, log hoặc report phụ nhưng có cách kiểm tra khác.
- `1`: cosmetic, wording hoặc nội bộ không ảnh hưởng runtime và artifact.

#### Likelihood `L`

- `5`: đang được sửa thường xuyên, vừa từng fail, hoặc nằm trên nhánh nhiều
  state/dependency.
- `4`: shared runtime/config path với nhiều caller hoặc logic khó quan sát.
- `3`: code active nhưng ít thay đổi và có một số caller.
- `2`: code ổn định, ít caller, ít lịch sử lỗi.
- `1`: historical, unreachable hoặc không còn nằm trong active config.

#### Uniqueness `U`

- `5`: test duy nhất bảo vệ contract quan trọng.
- `4`: edge case/state transition khác biệt rõ.
- `3`: có overlap nhưng vẫn có input hoặc oracle khác.
- `2`: gần như trùng với test khác.
- `1`: assertion trùng hoặc chỉ lặp cùng fixture.
- `0`: không có oracle có ý nghĩa, test obsolete hoặc chỉ kiểm tra implementation
  detail không còn tồn tại.

#### Active-path relevance `A`

- `5`: active offline/online benchmark hoặc public API hiện tại.
- `4`: public runtime path được gọi bởi active tests/configs.
- `3`: reachable nhưng không có active config hiện tại.
- `2`: demo, one-shot ops hoặc compatibility path có kiểm soát.
- `1`: historical config/legacy path.
- `0`: không có caller/config/reference nội bộ sau khi đã kiểm tra.

#### Oracle strength `O`

- `5`: deterministic exact value/schema/state assertion.
- `4`: artifact/checkpoint/state round-trip assertion rõ.
- `3`: integration assertion qua nhiều owner nhưng failure vẫn định vị được.
- `2`: chỉ kiểm tra side effect/log hoặc assertion rộng.
- `1`: flaky, phụ thuộc môi trường hoặc khó tái hiện.
- `0`: không chứng minh behavior nào.

## Hard-pin vào `high`

Trước khi xếp hạng, tạo tập `H_required`. Các test sau luôn ở `high`, bất kể
điểm `S`:

- config YAML: load, normalize và reject invalid/unknown active fields;
- loader: batch size, `x=[B,L,D]`, labels và split/window boundary;
- một forward và backward trên một batch nhỏ của model active;
- save/load checkpoint, strict reload và metadata role;
- synthetic anomaly injection: label/span correctness và một visualization
  representative theo yêu cầu trong `codebase_preferences.md`;
- offline threshold/calibration/metric contract hiện hành;
- online causal stream, EWMA/threshold, triage, verification buffer, state
  round-trip và variant update contract;
- ít nhất một smoke/runner contract cho mỗi public workflow đang active;
- compliance test bảo vệ các giới hạn kiến trúc đang được dùng làm gate.

Nếu hai test cùng bảo vệ một contract, chỉ test có oracle rõ hơn và ít trùng
hơn được hard-pin. Test còn lại vẫn chấm bình thường; không hard-pin toàn bộ
file theo tên thư mục.

## Workflow thực hiện tuần tự

### Phase 1 — Freeze baseline

1. Ghi revision, `git status --short` và danh sách file chưa được track.
2. Thu snapshot bằng:

   ```bash
   .venv/bin/python -m pytest --collect-only -q
   ```

3. Lưu 513 `nodeid` hiện tại, gồm cả từng parameterized case.
4. Chạy baseline focused suite và ghi pass/fail/skip; failure có trước migration
   là baseline fence, không được che bằng cách sửa assertion.

### Phase 2 — Build evidence matrix

Tạo một bảng tạm, mỗi dòng một `nodeid`, với các cột:

```text
nodeid, file, owner, parameter, I, L, U, A, O, R, S, evidence, group, reason
```

Các bước không được gộp:

1. Map test tới source owner và public workflow.
2. Map assertion tới contract, artifact hoặc failure mode.
3. Tìm duplicate/overlap bằng assertion, fixture và input; không chỉ bằng tên.
4. Ghi điểm và URL/path bằng chứng.
5. Gắn `H_required` trước khi xếp hạng.

### Phase 3 — Rank và tạo ba nhóm cân bằng

1. Kiểm tra mọi dòng có đủ điểm và bằng chứng; dòng thiếu evidence bị gắn
   `review`, chưa được archive.
2. Sắp xếp deterministic theo `(-S, -R, -U, -A, -O, nodeid)`.
3. Đặt tất cả `H_required` vào `high`.
4. Điền các vị trí còn lại của `high` từ đầu danh sách.
5. Gán 171 dòng tiếp theo vào `medium`, 171 dòng cuối vào `low`.
6. Kiểm tra coverage floor: `high` phải còn ít nhất một test cho từng owner
   active và từng workflow hard-pin. Nếu thiếu, thay test `high` không bắt buộc
   có điểm thấp nhất bằng test bị thiếu, rồi ghi lý do.
7. Xuất bảng tổng kết theo thư mục, owner, workflow và loại contract; cân bằng
   chỉ được chấp nhận khi vừa đủ quota vừa không làm mất coverage floor.

### Phase 4 — Review trước archive

1. Review từng dòng `high` và từng candidate `medium/low` có impact `I=5`.
2. Chọn một mẫu nhỏ ở ranh giới `high/medium` và `medium/low` để kiểm tra tie-break.
3. Nếu test nào không chứng minh được behavior hoặc source path không còn tồn
   tại, ghi `obsolete-candidate`; chưa archive cho tới khi caller search và
   artifact search đều âm tính.
4. Nếu test có impact cao nhưng active relevance thấp, giữ lại trong `high` khi
   nó bảo vệ historical result hoặc protocol audit; ghi rõ lý do override.

### Phase 5 — Archive có thể khôi phục

1. Tạo `tests_archive/` với cây thư mục mirror `tests/` và README ghi ngày,
   revision, manifest, lý do archive và lệnh restore.
2. File chỉ có `medium/low` case được di chuyển nguyên file bằng `git mv`.
3. File mixed phải tách test case trước: giữ `high` trong `tests/`, chuyển phần
   `medium/low` sang path tương ứng trong `tests_archive/`. Giữ fixture dùng
   chung ở nơi duy nhất; không copy implementation chỉ để giữ test cũ.
4. Không thay đổi source behavior, expected result hoặc contract để làm test
   pass sau khi archive.
5. Cập nhật `tests/README.md` để số liệu và quy tắc archive khớp thực tế.
6. Giữ `pytest.ini:norecursedirs = tests_archive` và kiểm tra Pytest không thu
   archive. Không thêm marker hoặc registry mới chỉ để lưu importance.

### Phase 6 — Verification và rollback gate

Chạy lần lượt:

```bash
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/core tests/data tests/models tests/engine
.venv/bin/python -m pytest -q tests/evaluation tests/online tests/benchmarks tests/runtime
```

Kỳ vọng sau migration:

- tổng collection là 171 active cases;
- mọi `high` nodeid còn được thu thập;
- không `medium/low` nodeid nào còn được thu thập từ `tests/`;
- các hard-pin contract vẫn chạy;
- baseline failure nếu còn phải được ghi riêng, không đổi thành pass bằng cách
  nới assertion;
- smoke/full-flow phù hợp với active path vẫn có kết quả và artifact schema
  không đổi.

Nếu collection count, coverage floor hoặc contract smoke không đạt, dừng ở
trạng thái review và khôi phục đúng các file vừa di chuyển bằng `git mv`. Không
xoá `tests_archive/` và không xoá test vĩnh viễn trong cùng migration.

Mutation testing hoặc coverage comparison chỉ là bước tăng độ tin cậy khi công
cụ và thời gian cho phép. Nếu dùng, so sánh baseline với active suite bằng cùng
config; không thêm dependency mới chỉ vì scoring workflow.

## Quy tắc duy trì sau migration

- Test mới mặc định vào `review`, không tự động vào `high`.
- Chấm lại khi source owner, active config, protocol, checkpoint schema hoặc
  online semantics đổi; và sau mỗi failure/incident quan trọng.
- Mỗi cycle có thể đổi `L` và `A`; `I` thường ổn định hơn nhưng phải sửa khi
  protocol hoặc mục tiêu luận văn đổi.
- Test archive được restore khi một risk quay lại active path hoặc khi audit
  cần bằng chứng. Sau đó chấm lại bằng cùng schema.
- Không dùng tổng số test, coverage phần trăm hoặc thời gian chạy đơn lẻ làm
  bằng chứng rằng suite đã đủ. Quyết định phải giữ được traceability tới
  contract và risk.

## Tiêu chí hoàn thành kế hoạch

Kế hoạch chỉ được xem là đã thực thi khi có manifest 513 dòng, ba nhóm đạt quota
hoặc có ngoại lệ được ghi, `high` giữ đủ hard-pin/coverage floor, archive mirror
khôi phục được, và verification log chỉ rõ baseline failure khác với failure do
migration. Note này tự nó chưa thực hiện phân loại hay di chuyển test.

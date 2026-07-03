---
date: 2026-06-30 15:23:35 +0700
author: Codex
topic: "Open questions for a future sequence-level synthetic augmentation spec"
status: pending
---

# Open Questions For Future Sequence-Level Synthetic Augmentation Spec

Muc dich cua note nay la giu lai cac cau hoi low-level can chot truoc khi viet spec hoan chinh cho huong:

- tach `train`, `val`, `test` theo thoi gian truoc
- fit `standard scaler` tren clean train
- transform `train`, `val`, `test`
- tao sequence synthetic truoc windowization
- `test` giu nguyen

Hien tai codebase van dang chay theo huong **window-level augmentation**. Note nay chi de danh dau cac diem mo ho can quay lai tra loi sau.

## 1. Split va scaler

1. `train` va `val` co phai luon duoc cat tu qua khu, con `test` la mot doan tuong lai khong?
2. `standard scaler` co dung la phai `fit` tren clean train only, hay duoc phep `fit` tren `train + val clean`?
3. `val` co bat buoc phai hoan toan clean goc khong, hay co the la contaminated nhung chua gan nhan?

## 2. Sequence-level synthetic augmentation

4. Tu moi sequence train goc va moi sequence val goc, co muon tao dung:
   - `1 clean sequence`
   - `11 synthetic sequences`
   dung khong?
5. 11 synthetic sequence do co phai la moi sequence chi chua dung 1 anomaly family khong?
6. Co can tao nhieu ban cho moi family voi cac ratio `1%`, `3%`, `5%`, hay moi family chi chon 1 ratio duy nhat?
7. Neu phai dung ca `1%`, `3%`, `5%`, co muon:
   - moi family co 3 ban rieng
   - hay chia family ra, moi family chi gan 1 trong 3 muc do?

## 3. Span injection

8. Moi synthetic sequence co duoc phep chua nhieu anomaly span hay chi 1 span duy nhat?
9. Span length duoc tinh theo:
   - dung bang `round(ratio * sequence_length)`
   - hay lay ngau nhien trong mot khoang quanh ratio do?
10. Vi tri span co duoc dat o bat ky cho nao trong sequence mien khong tran bien, hay can tranh qua sat dau/cuoi chuoi?
11. Neu sequence qua ngan khien `1%` ra 0 time-step, co muon:
   - ep toi thieu 1 time-step
   - hay bo khong inject sequence do o muc 1%?

## 4. Window label sau khi windowize

12. Window label multiclass se duoc gan the nao?
   - `normal` neu khong overlap
   - overlap thi nhan family tuong ung
   - hay can them rule theo ti le overlap toi thieu?
13. `point_labels` trong window co phai la mask pointwise that su cua span injected khong?
14. Neu mot window overlap rat it, vi du chi 1 point anomaly, no van bi xem la anomalous window dung khong?

## 5. Can bang du lieu

15. Dieu kien "can bang 12 class" muon ap dung o dau?
   - chi tren `train`
   - hay ca `train` va `val_synth`
16. "Can bang" o day muon can bang theo:
   - so luong sequence
   - so luong window
   - hay so luong anomaly point?
17. Neu giu `1 clean + 11 synthetic` cho moi sequence goc, thi sequence-level da can bang, nhung window-level thuong se khong can bang tuyet doi. Co chap nhan khong?

## 6. Determinism va reproducibility

18. `train_synth` co chot la co dinh y het trong suot mot run khong?
19. `val_synth` cung co dinh y het trong suot mot run luon dung khong?
20. Co muon synthetic artifact duoc materialize san o data layer de lan sau load lai dung y chang, hay chi can deterministic tu seed la du?

## 7. Runtime / scope

21. Ban dau tien muon ap dung unified cho:
   - chi `SMD` va `SWaT`
   - hay toan bo dataset active sau nay theo cung contract?
22. Voi dataset co nhieu entity nhu `SMD`, synthetic augmentation se chay doc lap cho tung entity, dung khong, khong bao gio tron entity voi nhau?

## Current Temporary Decision

Tam thoi de kip tien do benchmark:

- giu nguyen **window-level augmentation** nhu codebase hien tai
- khong chuyen sang sequence-level augmentation truoc khi da chot xong 22 cau hoi nay
- khi quay lai huong nay, dung note nay lam danh sach cau hoi de khoa spec

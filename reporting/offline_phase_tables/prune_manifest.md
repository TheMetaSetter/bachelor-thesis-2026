# Prune manifest and post-cleanup verification

Đây là danh sách các artifact đã được phân loại trước cleanup và kết quả hậu kiểm
sau khi thực hiện các entry có `action=delete`.

- Tạo lúc (UTC): `2026-07-25T09:13:59.983447+00:00`
- Remote root: `/root/bachelor-thesis-2026`
- Số entry: `5511`
- `keep`: `667`
- `review`: `4628`
- `delete` candidate: `216`
- Dung lượng `delete` candidate: `6.888 GiB`
- Protected artifact bị đánh dấu delete: `0`
- Hậu kiểm raw trace: `0` file thuộc cả ba loại cần xóa
- Hậu kiểm retention: `36` manifest, `36` summary, `233` checksum khớp
- Hậu kiểm lỗi retention: `0`

## Quy tắc và hậu kiểm

- Chỉ raw trace có canonical report row hợp lệ mới được đánh dấu `delete`.
- Summary, protocol, threshold, retention manifest/summary và checkpoint đều phải `keep`.
- `offline_metrics.json` giữ ở `review` cho tới khi chốt source discrepancy.
- `review` không phải lệnh xóa; cần quyết định riêng ở phase sau.
- Post-cleanup được kiểm tra ở chế độ read-only trên host `unstoppable-puma` ngày `2026-07-26`.

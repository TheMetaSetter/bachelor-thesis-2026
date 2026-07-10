# Test archive

Pytest không thu thập thư mục này. Mỗi nhóm chỉ lưu bằng chứng lịch sử cho một
luồng đã rời khỏi codebase hiện hành:

- `legacy_three_stage/`: orchestration ba stage cũ.
- `legacy_w100/`: runner ablation theo contract window 100 cũ.
- `broken_legacy/`: test đã mất dependency hoặc config nguồn.
- `stale_config_contracts/`: test trỏ tới YAML và validation surface đã bị xoá.
- `stale_contracts/`: test có assertion tự mâu thuẫn hoặc contract không còn
  được runtime hiện hành sở hữu.

Không chuyển test trở lại `tests/` nếu chưa xác định được public contract hiện
hành và file cấu hình đang tồn tại mà test đó bảo vệ.

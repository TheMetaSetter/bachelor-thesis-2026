# Bản đồ test cho học sinh phổ thông

Mỗi thư mục trả lời một câu hỏi đơn giản:

- `data/`: Dữ liệu có được đọc, làm sạch và cắt thành cửa sổ đúng không?
- `models/`: Mô hình có nhận đầu vào, tính loss và lưu state đúng không?
- `online/`: Luồng thích nghi online có giữ đúng threshold, buffer và projector không?
- `evaluation/`: Điểm bất thường và metric có được tính đúng không?
- `runtime/`: Config, checkpoint, logging và trainer có chạy đúng không?
- `benchmarks/`: Các script tạo và kiểm tra ma trận thí nghiệm có đúng không?
- `demo/`: Demo có phát dữ liệu tuần tự và vẽ kết quả đúng không?
- `compliance/`: Code có giữ các contract kiến trúc và giới hạn readability không?

Các test legacy hoặc test phụ thuộc config cũ được giữ ngoài `tests/` trong
`tests_archive/`. Pytest chỉ thu những test trong các thư mục trên.

1. Tối ưu EasyOCR hiện tại (không train lại)
Đây là bước nên làm ngay vì đơn giản, không cần training.

Dùng allowlist để chỉ cho phép [A–Z0–9] (và dấu gạch nếu cần): reader = easyocr.Reader(['en'], gpu=True, recog_network='english_g2') rồi khi gọi readtext thêm allowlist="ABCDEFGHIKLMNPQRSTUVXYZ0123456789" để loại ký tự lạ.​

Kết hợp hậu xử lý theo format biển số VN (regex, kiểm tra độ dài 8–9 ký tự, cấu trúc 2 số – 1–2 chữ – 4–5 số, v.v.) để sửa hoặc loại các dự đoán sai.​

Cách này đã giúp nhiều hệ thống biển số tăng đáng kể accuracy mà chưa cần fine-tune OCR.​


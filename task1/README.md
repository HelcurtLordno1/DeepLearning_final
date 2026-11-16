# Task 1 — dự báo một thời điểm

Input `[batch, 48, features]`; output `[batch, 1]` là `OT` đúng hai giờ sau quan sát cuối.

`ForecastRNN` dùng RNN hoặc LSTM. Attention lấy trạng thái cuối làm query và nhìn lại 48 trạng thái. Residual dự báo mức thay đổi rồi cộng nhiệt độ cuối đã quan sát. Không dùng nhiệt độ tương lai làm input.

Kiểm tra máy tại root: `powershell -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1`. Khi preflight cho phép, chạy một lượt ngắn bằng `powershell -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1 -Mode Train -Task task1 -Epochs 2 -Minutes 5`.

`task1.ipynb` có cùng code cốt lõi, mặc định `TRAIN=False` để xem kết quả đã lưu khi Run All. Đọc `CONFIG` ở đầu file và `selected_config.json` sau thực nghiệm. Run và metric lưu dưới `run/task1/`; có baseline persistence và daily seasonal. Ba seed benchmark cũ được giữ nguyên; lượt recovery có tên riêng và chỉ đánh giá validation.

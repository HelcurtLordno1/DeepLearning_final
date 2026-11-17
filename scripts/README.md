# Các công cụ còn dùng

Notebook và model nằm ở `task1/`, `task2/`, `task3/`. Thư mục này chỉ giữ công cụ chuẩn bị, vận hành và kiểm tra; không cần chạy tất cả các file.

| File | Dùng khi nào? |
| --- | --- |
| `bootstrap.py` | Các script import để tìm đúng thư mục project và `src/`. |
| `prepare_data.py` | Chuẩn bị dữ liệu từ archive trên máy mới; giữ manifest đã kiểm tra. |
| `check_environment.py` | Kiểm tra native Windows Python và các thư viện. |
| `presentation_notebooks.py` | Tạo lại bố cục notebook từ model thật, có backup output cũ. |
| `verify_presentation.py` | Chạy notebook với `TRAIN=False`, kiểm tra output và tạo HTML xem nhanh. |
| `verify_results.py` | Đối chiếu dự đoán, nhãn, metric, hash và code trong notebook. |
| `verify_layout.py` | Kiểm tra import, shape và tính tương thích của model bằng CPU. |
| `verify_safety.py` | Kiểm tra lock, mất monitor và lưu checkpoint an toàn bằng CPU. |
| `summarize_results.py` | Tổng hợp bảng kết quả từ các run đã khóa. |
| `create_task3_report.py` | Tạo Word, Markdown, bảng và hình Task 3 từ kết quả thật. |
| `verify_task3_report.py` | Kiểm tra số liệu và hình trong báo cáo Task 3. |
| `write_review_report.py` | Tạo bản review notebook và benchmark trong local `run/`. |
| `evaluate_task3.py` | Đánh giá checkpoint đã khóa, được launcher có monitor gọi. |

Các file PowerShell còn lại:

- `setup_environment.ps1`: tạo/cài môi trường Windows.
- `run_experiments.ps1`: mặc định kiểm tra tài nguyên; chỉ chạy một phiên khi chọn mode rõ ràng.
- `start_session.ps1`, `session_host.ps1`: mở và giám sát phiên hữu hạn.
- `verify_safety.ps1`, `verify_exit_code.ps1`: kiểm tra guard và mã thoát tiến trình, không train.

Trong `src/`, cả `utils.py` và `notebook_support.py` đều cần cho notebook. Các script một lần để chuyển thư mục, đọc đề, phục hồi sự cố, diagnostic cũ và sweep đã vô hiệu hóa đã được dọn khỏi thư mục này. Bản sao local nằm trong `run/cleanup_backup/`; lịch sử Git cũng giữ phiên bản trước.

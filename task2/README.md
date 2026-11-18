# Task 2 — dự báo năm thời điểm

Input `[batch, 48, features]`; output `[batch, 5]` là năm giá trị `OT` tại `t+1 … t+5`. Chiều scalar cuối được bỏ để tính MSE thuận tiện.

LSTM encoder tạo memory và hidden/cell states. Lặp context năm lần, truyền qua **LSTM decoder thật**, rồi dùng cùng một Linear cho từng bước. Attention cho mỗi bước decoder nhìn lại memory của encoder. Không đưa nhãn tương lai vào decoder.

Kiểm tra máy tại root: `powershell -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1`. Khi preflight cho phép, chạy một lượt ngắn bằng `powershell -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1 -Mode Train -Task task2 -Epochs 2 -Minutes 5`.

`task2.ipynb` có cùng code cốt lõi, mặc định `TRAIN=False` để xem artifact thay vì tự retrain. Metric có từng horizon, macro trung bình đều và pooled riêng. Run lưu dưới `run/task2/`. Ba seed benchmark cũ giữ nguyên; recovery chỉ đánh giá validation.

## Revised submission context

Task 2 is self-contained under the lecturer's updated scope. The notebook retains the original model, optimizer and training cells. Two short training-only views show the 48-hour input/five-target window and persistence errors across horizons. A decision table links these observations to the recorded validation comparisons; it does not claim that unrun lookbacks are optimal.

Macro test R²/NSE is 0.874910 and macro RMSE is 0.982309 ± 0.017116 over three seeds. The t+1 R² of about 0.9481 is a single-horizon regression score. The current English report covers Tasks 2 and 3 only.

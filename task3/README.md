# Task 3 — nhận dạng biển số

Mở `task3.ipynb`, chọn kernel **Final Project (Windows .venv)** rồi Run All. Mặc định `TRAIN=False` để xem dữ liệu, kiến trúc và kết quả đã lưu. Các thao tác PowerShell và quản lý checkpoint nằm ngoài notebook.

## Model đi qua những bước nào?

1. Ảnh crop RGB có shape `[B,3,64,192]`.
2. Ba block Conv–ReLU–Pool tạo `[B,128,8,48]`.
3. Nhánh hai Conv nhỏ bổ sung chi tiết, cộng vào feature với hệ số `0.1`.
4. Ghép tám hàng đặc trưng rồi dùng Linear `1024→128`, giữ chi tiết trên/dưới thay vì chỉ lấy trung bình. Cộng vector vị trí để phân biệt 48 cột trái/phải.
5. Pool thành bảy vùng ảnh. Mỗi bước decoder nhận `0.5 × vùng tương ứng + 0.5 × context toàn ảnh`.
6. GRU chạy bảy bước; dot attention cho mỗi bước nhìn lại 48 cột ảnh.
7. Ghép decoder và context attention, Linear `256→67`. Argmax rồi tra vocabulary tạo đủ bảy ký tự, gồm ký tự tỉnh.

Đây là CNN encoder + recurrent decoder theo yêu cầu đề. Model cuối có **642.051 tham số**; không dùng dịch vụ OCR ngoài, Transformer lớn hoặc nhãn thật làm input decoder.

## Code và dữ liệu

- `code/refined_model.py`: class model cuối, hiện cùng định nghĩa trong notebook.
- `code/task3.py`: model baseline/aligned và hàm đánh giá.
- `code/train_advanced.py`: optimizer và training theo phiên có checkpoint/resume.
- `config/refined.json`: cấu hình phát triển của model cuối.
- `../src/utils.py`: đọc dữ liệu, metric, train step và vẽ hình dùng chung.
- `../data/raw/ccpd/`: ảnh gốc; `../data/splits/`: manifest và vocabulary cố định.

Train/validation/test có **35.043 / 7.475 / 7.482 ảnh**, chia theo toàn chuỗi biển số. Không đổi holdout hoặc bỏ ảnh khó. Crop cache `uint8` ở `data/processed/` được đọc theo batch; không nạp toàn bộ ảnh thành float vào RAM.

## Huấn luyện và đánh giá

Cross-entropy nhận logits `[B,7,67]` và nhãn class `[B,7]`. Full-plate accuracy yêu cầu đúng cả bảy ký tự; character accuracy tính theo vị trí; CER tính Levenshtein edit distance / tổng ký tự thật. Bốn metric one-hot trong phụ lục được ghi rõ là phụ trợ, không hồi quy trên mã ký tự.

Warm start theo attention baseline → aligned → refined. Chọn checkpoint bằng validation accuracy, tie-break CER. Đây là **một seed fine-tune (42)**; không báo mean ± std qua ba lần train. Bảng so sánh thể hiện cả kiến trúc và thời lượng fine-tuning, không tách riêng tác dụng từng layer.

Run, source snapshot, checkpoint, history, prediction CSV và hình nằm trong `../run/task3/`. Đã chốt sau **12 epoch refined**, giữ checkpoint epoch 12 với **95,44% validation full-plate**. Cấu hình/hash checkpoint ở `final_selection.json` được khóa trước test. Các model đã khóa không được tiếp tục train bằng launcher phát triển.

Kiểm tra tài nguyên từ PowerShell tại root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1
```

Các phiên thật dùng native Windows `.venv`, batch tối đa 16, zero worker, CPU hai luồng, nghỉ giữa batch và monitor tài nguyên. Không chạy sweep dài hay tự retry safety stop. Trial vị trí lúc 22:14 ngày 16/11 bị mất monitor, chưa hoàn tất validation; artifact được giữ nguyên, không dùng làm bằng chứng tăng accuracy.

Phạm vi là nhận dạng **crop đã biết vị trí biển**, chưa đánh giá detector từ ảnh cả xe hoặc biển có độ dài khác. Các số cuối và hình minh họa được xuất từ dự đoán thật vào `../report/task3/`.

## Kết quả test đã khóa

| Model | Đúng toàn biển | Đúng ký tự | CER |
| --- | ---: | ---: | ---: |
| Attention baseline | 62,764% | 92,743% | 7,236% |
| Aligned | 80,540% | 96,353% | 3,637% |
| Refined cuối | **95,322%** | **99,148%** | **0,846%** |

Model cuối đọc đúng **7.132 / 7.482 biển**; 350 ảnh còn sai ít nhất một ký tự. Tăng **32,56 điểm phần trăm** so baseline, trên cùng tập test. Các số này là test độc lập; không dùng chúng để chọn lại model.

[Báo cáo Word](../report/task3/task3_report.docx), [Markdown](../report/task3/task3_report.md), [CSV benchmark](../report/task3/test_benchmark.csv).

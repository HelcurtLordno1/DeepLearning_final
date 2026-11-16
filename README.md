# Attention-Based Time Series Forecasting and License Plate Recognition

Deep Learning Final Project gồm ba bài: dự báo nhiệt độ dầu máy biến áp một bước, dự báo nhiều bước và nhận dạng biển số. Các notebook trình bày dữ liệu, model và kết quả thực nghiệm theo từng bước dễ đọc.

## Project Description

1. **Single-Step Time Series Forecasting:** Develop an LSTM model with Attention and residual forecasting to predict transformer oil temperature (`OT`) two hours ahead, using the previous 48 hourly observations of oil temperature and six electrical load variables from the ETTh1 dataset.
2. **Multi-Step Time Series Forecasting:** Implement an Encoder–Decoder LSTM with Attention and residual forecasting to predict transformer oil temperature for the next five consecutive hours, using the same ETTh1 dataset, input window, and target variable.
3. **License Plate Recognition:** Develop a CNN Encoder with a GRU Decoder and visual Attention to recognize complete seven-character Chinese license plates, including the province character, from annotation-cropped images using a fixed 50,000-image CCPD-Base subset.

Nội dung ba cột để điền sheet nằm trong [project_information.csv](report/project_information.csv).

## Link to Download Data

| Task | Dataset | Link |
| --- | --- | --- |
| 1 & 2 | ETTh1 — hourly electricity transformer data | [Download CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv) · [Authors' repository](https://github.com/zhouhaoyi/ETDataset) |
| 3 | CCPD-Base — fixed 50,000-image subset | [Official source and download instructions](https://github.com/detectRecog/CCPD) · [Pinned image mirror: train.zip](https://huggingface.co/datasets/okita-souji/ccpd2019train/resolve/230a8d1092b7c8b0113541efe9861527275b5c04/train.zip?download=true) |

ETTh1 chứa `OT` và sáu biến tải điện; Task 1–2 dùng cùng dữ liệu và target. File `train.zip` là nguồn ảnh gốc, **không phải tập train riêng của project**. Dùng danh sách 50.000 ảnh và grouped split đã lưu trong `data/splits/`; xem [hướng dẫn dữ liệu](data/raw/README.md).

## Kết quả test

| Task | Bài toán | Model | Kết quả test |
| --- | --- | --- | --- |
| 1 | ETTh1: OT đúng `t+2` | LSTM + attention + residual | RMSE **0,846787 ± 0,007900**, 3 seed |
| 2 | Cùng ETTh1: OT `t+1 … t+5` | LSTM Encoder–Decoder + attention | Macro RMSE **0,982309 ± 0,017116**, 3 seed |
| 3 | Ảnh biển số → đủ 7 ký tự | Refined CNN + GRU decoder + attention | Đúng toàn biển **95,32%**, 1 seed fine-tune |

Task 1–2 giảm RMSE **10,38% / 13,55%** so persistence. Task 3 trên **7.482 ảnh test**: đúng **7.132 biển**, character accuracy **99,15%**, CER **0,85%**. Baseline attention đạt **62,76%** toàn biển; model mới tăng **32,56 điểm phần trăm** trên cùng test. Không báo std qua nhiều seed cho Task 3.

## Mở notebook

Mở `task1/task1.ipynb`, `task2/task2.ipynb` hoặc `task3/task3.ipynb`; chọn kernel **Final Project (Windows .venv)** rồi Run All.

Mặc định `TRAIN=False`: xem EDA, nạp checkpoint CPU, forward mẫu và hiển thị benchmark đã lưu. Không tự train lại để tạo output. Model, attention, optimizer, train step và shape hiện trong notebook; quản lý file và PowerShell nằm ngoài notebook. Bản HTML xem nhanh ở `run/notebook_review/`.

**Khi clone từ GitHub:** output đã lưu trong notebook và báo cáo có thể xem ngay. Chạy lại toàn bộ notebook cần dữ liệu gốc cùng checkpoint/bảng kết quả trong `run/` trên máy cục bộ. Repository không kèm `.venv`, ảnh gốc, crop cache hoặc thư mục `run/`; chỉ tải dữ liệu chưa đủ để chạy chế độ xem checkpoint. Các bảng benchmark và kiểm tra đã được xuất sang `report/` để đọc trên GitHub.

## Thư mục

```text
task1/ task2/ task3/   Notebook và README từng bài
  code/               Model và code Python tương ứng
src/                  Helper dữ liệu, metric và notebook
scripts/              PowerShell, training launcher, kiểm tra, tạo report
environment/          requirements.txt và requirements-cuda.txt
data/raw/             Dữ liệu gốc, không sửa
data/splits/          Manifest và vocabulary cố định
data/processed/       Crop uint8 đọc theo batch
run/                  Checkpoint, source snapshot, history, prediction và log
report/task3/         Word, Markdown, bảng CSV và hình của Task 3
report/benchmark/     Bảng kết quả cả ba task để xem trên GitHub
report/verification/  Bằng chứng kiểm tra notebook và metric
assignments/          Đề và tài liệu riêng, chỉ giữ local
lessons_slides/       Bài giảng riêng, chỉ giữ local
```

Không còn `.py`, `.ps1` hoặc `.txt` rải ở root. `.venv`, `__pycache__` và `__init__.py` được ẩn trong Explorer của VS Code; đường dẫn môi trường và kernel giữ nguyên. Code/artifact trước khi chuyển có bản sao trong `run/layout_backup/`.

## Windows PowerShell

Mở PowerShell tại root. Máy hiện tại đã có `.venv`; trên máy mới, chạy `scripts/setup_environment.ps1` trước. Sau đó:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python .\scripts\check_environment.py
```

Nếu cần tạo/cài lại, dùng `scripts/setup_environment.ps1`. Python là `.venv\Scripts\python.exe`, chạy native Windows; không dùng môi trường Linux/WSL. GPU thực nghiệm: RTX A4500 Laptop 16 GiB.

Kiểm tra notebook và kết quả khi không có training:

```powershell
.\.venv\Scripts\python.exe .\scripts\verify_presentation.py
.\.venv\Scripts\python.exe .\scripts\verify_results.py
```

Kiểm tra tài nguyên mà không train:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_experiments.ps1
```

## Thực nghiệm Task 3

Chuỗi đã chạy: attention baseline → aligned (4 epoch) → refined (12 epoch). Bản mới giữ chi tiết theo chiều cao, cung cấp bảy vùng trái–phải cho decoder, thêm hai Conv residual nhỏ và vị trí cột; tổng **642.051 tham số**. Không dùng OCR ngoài. Code ở `task3/code/`, config ở `task3/config/`.

Checkpoint epoch 12 đạt **95,44% validation** và được khóa **trước khi mở test**. Hash và lựa chọn ở `run/task3/final_selection.json`. Đây là một seed fine-tune từ checkpoint đã học trên train; bảng so sánh phản ánh cả kiến trúc lẫn thời lượng fine-tuning. Các model đã khóa được bảo vệ khỏi tiếp tục train bằng launcher.

Các phiên thật dùng `scripts/start_session.ps1` và monitor Windows: batch ≤16, 0 loader worker, CPU 2 luồng, nghỉ giữa batch, checkpoint/resume và kiểm tra RAM/commit/SSD/GPU/log phần cứng. Mỗi phiên hữu hạn; không tự retry safety stop. Lịch sử sự cố cũ và trial bị ngắt được giữ tại `run/safety/`. Không bật train hàng loạt từ notebook.

## Dữ liệu và metric

ETTh1 chia thời gian 70/15/15; scaler chỉ fit train. CCPD có **35.043 / 7.475 / 7.482 ảnh**, chia theo toàn chuỗi biển số, không trùng plate giữa các tập. Crop từ annotation, chưa đánh giá detector ảnh cả xe. Chi tiết: [data/raw/README.md](data/raw/README.md).

Task 1–2 báo R², NSE, MAE và RMSE ở thang OT gốc; R²/NSE bằng nhau theo công thức residual. Task 2 phân biệt macro với pooled và có đủ năm horizon. Task 3 dùng full-plate accuracy, character accuracy và CER Levenshtein; bốn metric decoded one-hot được ghi rõ là phụ trợ, không hồi quy trên mã ký tự.

## Báo cáo và bằng chứng

- [Báo cáo Task 3 Word](report/task3/task3_report.docx) và [Markdown](report/task3/task3_report.md).
- [Bảng test trước/sau](report/task3/test_benchmark.csv), [hình và bảng chi tiết](report/task3/).
- [Kết quả cả ba task](report/benchmark/README.md), [bảng test](report/benchmark/all_test_metrics.csv).
- [Đối chiếu dự đoán với nhãn gốc và metric](report/verification/results_audit.json), [kiểm tra output notebook](report/verification/presentation_checks.json).

Các thư mục `assignments/`, `lessons_slides/` và hai file `AGENTS.md`, `project_replan.md` được giữ local và loại khỏi Git theo yêu cầu. Dữ liệu lớn, môi trường và experiment artifacts tiếp tục dùng quy tắc ignore sẵn có.

Template Word và PDF đề gốc không bị sửa. Nội dung Task 3 mới có thể đưa vào báo cáo chung; tên nhóm và thông tin bìa cần điền theo nhóm của bạn.

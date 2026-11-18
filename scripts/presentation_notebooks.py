"""Build classroom notebooks. Back up real outputs before replacing a notebook."""

import bootstrap  # Resolve src/ and the project root.
from pathlib import Path
from datetime import datetime
import ast
import shutil
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def definition(filename, name):
    source = (ROOT/filename).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])


def build(task):
    cells = []
    def md(text):
        cells.append(nb.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text, role=None):
        source = textwrap.dedent(text).strip()
        compile(source, task, 'exec')
        metadata = {'scrolled': False}
        if role:
            metadata['presentation_role'] = role
        cells.append(nb.v4.new_code_cell(source, metadata=metadata))

    forecast = task != 'task3'
    architecture = 'legacy'
    final_plate_test = False
    if not forecast:
        from notebook_support import plate_runs, read_json
        candidates = plate_runs()
        chosen = max(candidates, key=lambda row: (row['plate_accuracy'], -row['CER']))['path']
        selection_file = ROOT/'run/task3/final_selection.json'
        if selection_file.exists():
            chosen = ROOT/read_json(selection_file)['selected_run']
            final_plate_test = (chosen/'test_metrics.json').exists()
        architecture = read_json(chosen/'config.json').get('architecture','legacy')
    title = {'task1': 'Dự báo nhiệt độ dầu trước 2 giờ',
             'task2': 'Dự báo 5 giờ bằng Encoder–Decoder',
             'task3': 'Nhận dạng chuỗi 7 ký tự biển số'}[task]
    objective = {'task1': 'Dùng 48 giờ quan sát để dự đoán một giá trị OT tại **t+2**.',
                 'task2': 'Dùng 48 giờ ETTh1 với target OT, dự đoán **t+1 đến t+5** bằng Encoder–Decoder LSTM có attention và residual.',
                 'task3': 'Từ ảnh biển số đã crop, dự đoán đủ **7 ký tự**, gồm ký tự tỉnh.'}[task]
    status = ('Đã có test độc lập cho cấu hình cuối với **3 seed: 42, 123, 2026**.' if forecast else
              ('Đã khóa cấu hình bằng validation và đánh giá test độc lập. Task 3 hiện có **một seed fine-tune**, không trình bày thành ba lần train độc lập.' if final_plate_test else
               '**Chưa có benchmark test cuối.** Các số hiện tại là validation, một seed; bài toán còn cần cải thiện.'))
    md(f'# {task.replace("task", "Task ")} — {title}\n\n{objective}\n\n{status}\n\nNotebook trình bày dữ liệu, kiến trúc và kết quả đã đo. `TRAIN=False` xem lại kết quả, không tự huấn luyện lại.')
    md('## Chuẩn bị\n\nDùng PyTorch cho model; pandas và matplotlib để xem dữ liệu. Các hàm đọc kết quả và vận hành được để trong helper như notebook M1.')
    code(f'''
        from pathlib import Path
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from IPython.display import display

        ROOT = Path.cwd() if (Path.cwd()/"src/utils.py").exists() else Path.cwd().parent
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT/"src"))
        from notebook_support import (
            load_run, setup_plots, layer_shapes, load_weights, training_settings,
            validation_comparison, prepare_training, train_presented_model
        )

        setup_plots()
        torch.set_num_threads(2)
        TRAIN = False
        TASK = "{task}"
        run = load_run(TASK, architecture="{architecture}")
        settings = run["config"]
        torch.manual_seed(settings["seed"])
    ''', 'setup')

    md('## 1. Đọc và kiểm tra dữ liệu')
    if forecast:
        md('**ETTh1:** 17.420 quan sát theo giờ, 7 biến số. `OT` là biến cần dự đoán; sáu biến còn lại là các phép đo tải điện. Dữ liệu được giữ đúng thứ tự thời gian.')
        code('''
            data = pd.read_csv(ROOT/"data/raw/etth1/ETTh1.csv", parse_dates=["date"])
            display(data.head())
            print("Số quan sát:", len(data))
            print("Khoảng thời gian:", data.date.min(), "→", data.date.max())
            print("Giá trị thiếu:", data.isna().sum().sum())
            print("Timestamp trùng:", data.date.duplicated().sum())
            print("Khoảng cách thời gian:", data.date.diff().dropna().unique())
        ''', 'data')
        md('## 2. EDA trên tập train\n\nChỉ dùng 70% đầu để tìm hiểu phân bố, tương quan và độ trễ. Các nhận xét này tạo giả thuyết; quyết định giữ cải tiến phải dựa trên validation.')
        code('''
            train_end = int(len(data) * 0.70)
            validation_end = int(len(data) * 0.85)
            train_data = data.iloc[:train_end]
            display(train_data.drop(columns="date").describe().round(2))

            fig, axes = plt.subplots(1, 2, figsize=(11, 3))
            axes[0].plot(train_data.date, train_data.OT, linewidth=0.6)
            axes[0].set(title="OT in training data", xlabel="Time", ylabel="OT")
            axes[1].hist(train_data.OT, bins=40, color="steelblue")
            axes[1].set(title="Training distribution", xlabel="OT", ylabel="Count")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()
        ''', 'eda_distribution')
        code('''
            correlations = train_data.drop(columns="date").corr()["OT"].drop("OT")
            lags = [1, 2, 5, 24, 48, 72]
            autocorrelation = [train_data.OT.autocorr(lag) for lag in lags]

            fig, axes = plt.subplots(1, 2, figsize=(11, 3))
            correlations.plot.barh(ax=axes[0], color="steelblue")
            axes[0].set(title="Correlation with OT (train)", xlabel="Pearson correlation")
            axes[1].bar([str(lag) for lag in lags], autocorrelation, color="darkorange")
            axes[1].set(title="OT autocorrelation (train)", xlabel="Lag in hours", ylim=(0, 1))
            plt.tight_layout()
            plt.show()
            display(pd.DataFrame({"Lag (giờ)": lags, "Autocorrelation": autocorrelation}).round(3))
        ''', 'eda_lags')
        if task == 'task2':
            md('### A concrete five-hour forecast\n\nThe last observed hour is t. All five targets occur after t. This example uses only training data; the model will learn to predict the orange points from the blue history.')
            code('''
                example = train_data.iloc[:53]
                plt.figure(figsize=(10, 3))
                plt.plot(example.date.iloc[:48], example.OT.iloc[:48], label="Observed: 48 hours")
                plt.plot(example.date.iloc[48:], example.OT.iloc[48:], "o--", label="Targets: next 5 hours")
                plt.axvline(example.date.iloc[47], color="grey", linestyle=":", label="Last observed hour t")
                plt.xlabel("Time")
                plt.ylabel("OT")
                plt.legend()
                plt.tight_layout()
                plt.show()
            ''', 'eda_window')
            md('### Why compare with persistence and learn a correction?\n\nFor each horizon, calculate future OT minus the last observed OT on training data. Larger corrections make a constant forecast harder. This motivates a residual hypothesis; validation measurements decide whether to retain it.')
            code('''
                changes = pd.DataFrame({f"t+{h}": train_data.OT.shift(-h) - train_data.OT
                                        for h in range(1, 6)}).dropna()
                training_baseline = pd.DataFrame({"MAE": changes.abs().mean(),
                                                   "RMSE": np.sqrt(changes.pow(2).mean())})
                display(training_baseline.round(4))
                training_baseline.plot.bar(figsize=(8, 3), rot=0)
                plt.title("Persistence errors on training data only")
                plt.xlabel("Forecast horizon")
                plt.ylabel("Error in original OT scale")
                plt.tight_layout()
                plt.show()
            ''', 'eda_persistence')
        md('**Ý nghĩa cho mô hình:** OT gần đây là baseline mạnh, vì vậy cần so với persistence. Tự tương quan ở độ trễ dài có thể do cả xu hướng chậm; chưa đủ để kết luận có chu kỳ 24 giờ. Các hướng kiểm tra là input 24/48/72 giờ, OT đơn lẻ so với nhiều biến, và dự báo phần thay đổi (residual). Không coi các hướng chưa chạy là đã cải thiện điểm.')
        md('## 3. Chia dữ liệu và tạo cửa sổ\n\nChia 70% / 15% / 15% theo thời gian. Scaler chỉ fit train; mỗi cửa sổ nằm trọn trong một tập. Input có 48 giờ × 7 biến. `t` là timestamp cuối đã quan sát.')
        code('''
            from utils import prepare_forecast
            parts, scalers, horizons, identity = prepare_forecast(TASK, settings)
            partitions = [data.iloc[:train_end], data.iloc[train_end:validation_end],
                          data.iloc[validation_end:]]
            split_table = []
            for split, frame in zip(["train", "validation", "test"], partitions):
                split_table.append({"Tập": split, "Dòng gốc": len(frame),
                                    "Từ": frame.date.iloc[0], "Đến": frame.date.iloc[-1],
                                    "X shape": str(parts[split]["X"].shape),
                                    "y shape": str(parts[split]["y"].shape)})
            display(pd.DataFrame(split_table))
            origin = pd.Timestamp(parts["train"]["origins"][0])
            display(pd.DataFrame({"Horizon": horizons,
                                  "Quan sát cuối t": origin,
                                  "Target timestamp": parts["train"]["targets"][0]}))
        ''', 'split')
        md(('Ví dụ: input kết thúc tại 23:00 thì năm target là 00:00–04:00 ngày tiếp theo. ' if task=='task2' else 'Target là t+2. ')+'Nhãn dự báo không xuất hiện trong input. StandardScaler chỉ fit train; metric cuối được đưa về thang OT gốc.')
    else:
        md('**CCPD-Base:** 50.000 ảnh trong cohort đã chọn, crop theo bounding box annotation. Đây là nhận dạng khi đã biết vị trí biển số; chưa đánh giá detector từ ảnh cả xe.')
        code('''
            data = pd.read_csv(ROOT/"data/splits/ccpd_subset.csv")
            # Vocabulary dùng cho model lấy từ thực nghiệm để giữ đúng thứ tự class.
            from notebook_support import read_json, plate_examples, plate_batch
            vocabulary = read_json(ROOT/"data/splits/ccpd_vocabulary.json")
            display(data[["text", "split"]].head().rename(columns={"text": "Nhãn", "split": "Tập"}))
            display(data.groupby("split").agg(Images=("text", "size"), Plates=("text", "nunique")))
            print("Số class:", len(vocabulary))
            print("Độ dài nhãn:", sorted(data.text.str.len().unique()))
        ''', 'data')
        md('## 2. Xem ảnh và nhãn\n\nSáu ảnh đầu của train, không chọn thủ công ảnh đẹp. Giữ đủ ký tự tỉnh và sáu vị trí tiếp theo. Chia theo toàn chuỗi biển số để không trùng danh tính giữa các tập.')
        code('''
            plate_examples(data)
            groups = {split: set(data.loc[data.split.eq(split), "text"])
                      for split in ["train", "validation", "test"]}
            print("Plate trùng train/validation:", len(groups["train"] & groups["validation"]))
            print("Plate trùng train/test:", len(groups["train"] & groups["test"]))
            print("Plate trùng validation/test:", len(groups["validation"] & groups["test"]))
            province_counts = data.loc[data.split.eq("train"), "text"].str[0].value_counts()
            display(pd.DataFrame({"Ký tự tỉnh (train)": province_counts.index,
                                  "Số ảnh": province_counts.values,
                                  "Tỷ lệ (%)": 100*province_counts.values/province_counts.sum()}).head().round(2))
        ''', 'eda_images')
        md('## 3. Tiền xử lý và tensor đầu vào\n\nCrop → RGB → resize 192×64 → chia 255 → đổi sang `[B,3,64,192]`. Nhãn là chỉ số class `[B,7]`. Augmentation, nếu bật trong cấu hình, chỉ áp dụng train; validation và test giữ nguyên. Bảng thông số cho biết cấu hình thực tế.')
        code('''
            X_example, y_example = plate_batch(data, vocabulary)
            print("X:", tuple(X_example.shape), "range:", float(X_example.min()), "→", float(X_example.max()))
            print("y:", tuple(y_example.shape))
            display(pd.DataFrame({"Nhãn gốc": data[data.split.eq("train")].head(2).text.tolist(),
                                  "Class IDs": y_example.tolist()}))
        ''', 'split')

    md('## 4. Xây dựng mô hình\n\n`B` là batch size, `H` là hidden size. Dùng cùng dot-product attention: query so với các trạng thái đầu vào, softmax tạo trọng số, rồi lấy tổng có trọng số. Đây là một attention head, không phải MultiHeadAttention.')
    code(definition('src/utils.py', 'dot_attention'), 'attention')
    model_name = {'task1': 'ForecastRNN', 'task2': 'ForecastSeq2Seq', 'task3':
                  {'aligned':'AlignedPlateSeq2Seq','refined':'RefinedPlateSeq2Seq'}.get(architecture,'PlateSeq2Seq')}[task]
    model_source = 'task3/code/refined_model.py' if architecture=='refined' else task+'/code/'+task+'.py'
    if task == 'task1':
        md('**Luồng:** `[B,48,7]` → LSTM → memory `[B,48,128]` → attention từ trạng thái cuối → concat `[B,256]` → Linear → cộng OT cuối → `[B,1]`. Residual học mức thay đổi, không dùng OT tương lai.')
    elif task == 'task2':
        md('**Luồng:** encoder LSTM nhận `[B,48,7]` → memory và hidden/cell → lặp context 5 lần → decoder LSTM `[B,5,128]` → cross-attention → Linear dùng chung → cộng OT cuối → `[B,5]`. Decoder nhận context và state của encoder, không nhận nhãn tương lai.')
    else:
        if architecture in ['aligned','refined']:
            md('**Luồng mới:** `[B,3,64,192]` → CNN `[B,128,8,48]` → học tổ hợp theo chiều cao `[B,48,128]` → pooling thành **7 vùng trái–phải** → trộn local region với global context → GRU decoder 7 bước → attention nhìn lại 48 cột → Linear → `[B,7,67]`.\n\nBản cũ lặp cùng context bảy lần và lấy trung bình theo chiều cao. Bản mới giữ được nét ở trên/dưới và cho mỗi bước decoder một vùng ảnh tương ứng. Khởi tạo từ CNN–GRU attention đã train trên **train**, rồi fine-tune; không phải train từ đầu.')
            if architecture=='refined':
                md('**Hai phần bổ sung:** nhánh residual gồm hai Conv 3×3 hiệu chỉnh nét từ CNN; vector vị trí cho 48 cột giúp attention phân biệt trái/phải. Nhánh mới và vector vị trí bắt đầu bằng 0 để giữ nguyên dự đoán của checkpoint khởi tạo. Mô hình vẫn là CNN encoder → GRU decoder → attention, không dùng OCR ngoài.')
                md('**Chuỗi học thật:** checkpoint baseline attention → aligned → refined. Refined nạp trọng số aligned, không nạp trực tiếp baseline cũ. Mỗi giai đoạn có history riêng; learning curve bên dưới chỉ là giai đoạn refined.')
                md('**Why these additions?**\n\n- The height projection preserves upper/lower character detail.\n- Seven local regions give the GRU an ordered spatial cue; column positions distinguish left from right.\n- The small residual branch adjusts CNN features while preserving the warm-start prediction at initialization.\n\nThe measured improvement combines architecture changes and extra training. It is not an isolated ablation of each component, and the final model has one fine-tuning seed.')
        else:
            md('**Luồng baseline:** `[B,3,64,192]` → CNN `[B,128,8,48]` → trung bình chiều cao → 48 cột `[B,48,128]` → GRU 7 bước → attention → Linear → logits `[B,7,67]`. Argmax ở mỗi vị trí tạo chuỗi.')
    code(definition(model_source, model_name), 'model')
    if forecast:
        code(f'''
            model = {model_name}(len(scalers["features"]), settings)
            if not TRAIN:
                load_weights(model, run)
            X_example = torch.from_numpy(parts["validation"]["X"][:2])
            last_example = torch.from_numpy(parts["validation"]["last"][:2])
            layers, prediction = layer_shapes(model, X_example, last_example)
            print(model)
            display(layers)
            print("Output:", tuple(prediction.shape))
            print("Số tham số:", sum(p.numel() for p in model.parameters()))
        ''', 'architecture_output')
    else:
        code(f'''
            model = {model_name}(len(vocabulary), settings)
            if not TRAIN:
                load_weights(model, run)
            layers, logits = layer_shapes(model, X_example)
            print(model)
            display(layers)
            print("Logits:", tuple(logits.shape))
            predicted_ids = logits.argmax(dim=-1)
            display(pd.DataFrame({{"Nhãn": data[data.split.eq("train")].head(2).text.tolist(),
                                  "Dự đoán": ["".join(vocabulary[i] for i in row) for row in predicted_ids.tolist()]}}))
        ''', 'architecture_output')
    md('Bảng trên là shape đi qua **layer thật** trên hai mẫu, không phải bảng shape viết tay. Attention, phép lấy trung bình và cộng residual là phép toán trong `forward`, nên cần đọc cùng sơ đồ và code phía trên.')

    md('## 5. Loss và huấn luyện\n\nThông số bên dưới thuộc run đã đo, không phải cấu hình recovery mới. Checkpoint được chọn bằng validation, không lấy epoch cuối một cách mặc định.')
    code('display(training_settings(run))', 'training_settings')
    explanation = ('MSE trên OT đã chuẩn hóa. Khi đánh giá, inverse scale để tính sai số ở đơn vị gốc.' if forecast else
                   'Model trả logits `[B,7,67]`; bước train transpose thành `[B,67,7]` cho cross-entropy, cùng target `[B,7]`. Không softmax trước loss; chỉ argmax khi dự đoán ký tự.')
    md(explanation+'\n\nHàm dưới đây là một bước cập nhật **được dùng trong vòng lặp train thật** của project. Ô này chỉ định nghĩa hàm, không tự train.')
    step = 'forecast_train_step' if forecast else 'plate_train_step'
    code(definition('src/utils.py', step), 'training_step')
    if not forecast and architecture in ['aligned','refined']:
        md('Các lớp đã học dùng learning rate nhỏ; lớp mới dùng learning rate lớn hơn. Đây là cùng hàm tạo Adam được dùng trong script fine-tune.')
        code(definition('task3/code/train_advanced.py','make_optimizer'), 'optimizer_factory')
        code('''
            if TRAIN:
                model = prepare_training(model, settings["seed"])
            optimizer = make_optimizer(model, settings)
            display(pd.DataFrame({"Nhóm": range(1,len(optimizer.param_groups)+1),
                                  "Learning rate": [group["lr"] for group in optimizer.param_groups],
                                  "Số tham số": [sum(p.numel() for p in group["params"]) for group in optimizer.param_groups]}))
        ''', 'optimizer')
    else:
        code('''
            if TRAIN:
                model = prepare_training(model, settings["seed"])
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
            print("Adam — learning rate ban đầu:", settings["learning_rate"])
        ''', 'optimizer')
    call = ('parts=parts, scalers=scalers' if forecast else '')
    if not forecast and architecture in ['aligned','refined']:
        md('Kết quả đã đo dùng chuỗi warm-start ở trên. Đổi `TRAIN=True` tạo trial mới từ đầu để học code; không tái tạo nguyên trạng benchmark fine-tune. Cách chạy chuỗi thực nghiệm được ghi trong README.')
    code(f'''
        if TRAIN:
            run = train_presented_model(TASK, model, optimizer, settings, {call})
            display(training_settings(run))
        history = run["history"]

        history_columns = {(['epoch', 'train_loss', 'val_RMSE', 'lr'] if forecast else ['epoch', 'train_loss', 'val_loss', 'val_plate_accuracy', 'lr'])!r}
        display(history[history_columns].tail(5).round(4))
    ''', 'training_call')
    md('## 6. Validation và lựa chọn cấu hình\n\nSo sánh ở seed 42 trước, chọn bằng validation. Chỉ cấu hình cuối mới được lặp thêm seed; không chọn seed theo điểm test. Learning curve dưới đây là run được phân tích ở trên.')
    code('''
        best_epoch = run["metadata"]["best_epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3))
        axes[0].plot(history.epoch, history.train_loss, marker="o", markersize=3, label="Train")
        axes[0].set(xlabel="Epoch", ylabel="Loss")
    ''' + ('''
        axes[0].plot(history.epoch, history.val_loss, marker="o", markersize=3, label="Validation")
        axes[0].set_title("Cross-entropy")
        axes[1].plot(history.epoch, history.val_plate_accuracy * 100, marker="o", color="darkorange")
        axes[1].set(title="Validation full-plate accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    ''' if not forecast else '''
        axes[0].set_title("Train MSE (scaled OT)")
        axes[1].plot(history.epoch, history.val_RMSE, color="darkorange")
        axes[1].set(title="Validation RMSE (original OT)", xlabel="Epoch", ylabel="RMSE")
    ''') + '''
        axes[0].legend()
        axes[1].axvline(best_epoch, linestyle="--", color="grey", label="Best checkpoint")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        display(validation_comparison(TASK))
    ''', 'learning_curves')
    if task == 'task2':
        md('### How the evidence leads to the final model\n\n| Evidence | Decision |\n| --- | --- |\n| Ordered hourly observations and short-term continuity | Use an LSTM and retain persistence as a strong baseline. |\n| Five future values are required | Use a real five-step recurrent decoder. |\n| Training changes motivate corrections | Test residual forecasting; retain it based on validation. |\n| Attention helps some configurations but not all | Compare measured validation RMSE, rather than assume a benefit. |\n| seq128_attention_residual has the lowest measured validation RMSE | Freeze that configuration, then report three seeds on test. |\n\nThe 48-hour input is a design choice. Lookbacks 24/72 were not fully benchmarked, so EDA does not establish that 48 is optimal. New training-only EDA explains the saved experiment; it is not a new model-selection experiment.')
    if forecast:
        md('MSE train và RMSE validation có thang đo khác nhau nên vẽ hai trục riêng. Bảng so sánh cho thấy attention không luôn giúp: ở Task 2, `seq64_attention_residual` kém `seq64_residual`; cấu hình H=128 mới cho validation tốt nhất. Vì vậy không quy mọi cải thiện cho attention.' if task=='task2' else
           'MSE train và RMSE validation có thang đo khác nhau nên vẽ hai trục riêng. So sánh RNN, LSTM, attention và residual giúp thấy phần nào tạo cải thiện. Cấu hình cuối có H=128; không dùng test để chọn lại lookback hoặc model.')
        md('## 7. Benchmark test cuối\n\n**R²/NSE càng cao càng tốt; MAE/RMSE càng thấp càng tốt.** R² là hệ số xác định, không phải phần trăm accuracy. R² và NSE dùng cùng công thức residual nên bằng nhau. Bảng chính dùng mean ± sample std của đúng ba seed; baseline là công thức xác định, không phải ba model đã train.')
        if task=='task2':
            md('**Macro:** tính metric riêng cho từng horizon rồi trung bình đều. **Pooled:** gộp mọi dự đoán rồi tính một lần. RMSE macro và pooled khác nhau; benchmark chính dùng macro. Mỗi horizon có 2.561 cặp test.')
        code('''
            from notebook_support import forecast_benchmark, benchmark_tables
            benchmark = forecast_benchmark(TASK)
            final_table, seed_table, improvement = benchmark_tables(benchmark)
            display(final_table)
            display(seed_table)
            print(f"Giảm RMSE so với persistence: {improvement:.2f}%")
            print("Mục tiêu trong plan: giảm ít nhất 5% —", "ĐẠT" if improvement >= 5 else "CHƯA ĐẠT")
        ''', 'benchmark')
        if task=='task2':
            code('''
                per_horizon = benchmark[benchmark.horizon.str.startswith("t+")]
                network = per_horizon[per_horizon.model.eq("model")]
                means = network.groupby("horizon")[["R2", "NSE", "MAE", "RMSE"]].mean()
                means["RMSE std"] = network.groupby("horizon").RMSE.std()
                persistence = per_horizon[per_horizon.model.eq("persistence")].groupby("horizon").RMSE.first()
                means["Persistence RMSE"] = persistence
                means["Giảm RMSE (%)"] = 100 * (1-means.RMSE/persistence)
                display(means.round(4))
                pooled = benchmark[benchmark.model.eq("model") & benchmark.horizon.eq("pooled")]
                print(f"Pooled RMSE, 3 seed: {pooled.RMSE.mean():.4f} ± {pooled.RMSE.std():.4f}")
            ''', 'horizon_benchmark')
        md('## 8. Xem dự đoán và sai số\n\nHình dùng 240 dự đoán test đầu tiên theo thời gian của seed 42, không chọn các đoạn đẹp. Bảng nhỏ hiển thị target timestamp và sai số thực tế.')
        code('''
            predictions = run["predictions"].copy()
            predictions["target_time"] = pd.to_datetime(predictions.target_time)
            predictions["absolute_error"] = abs(predictions.actual-predictions.predicted)
            columns = ["target_time", "horizon", "actual", "predicted", "absolute_error"]
            display(predictions[columns].head().round({"actual": 4, "predicted": 4, "absolute_error": 4}))
        ''' + ('''
            fig, axes = plt.subplots(1, 2, figsize=(11, 3))
            values = predictions.head(240)
            axes[0].plot(values.target_time, values.actual, label="Actual")
            axes[0].plot(values.target_time, values.predicted, label="Model")
            axes[0].plot(values.target_time, values.persistence, label="Persistence", alpha=.5)
            axes[0].set(title="t+2: first 240 test predictions", ylabel="OT")
            axes[0].legend()
            axes[1].hist(predictions.absolute_error, bins=40)
            axes[1].set(title="Absolute errors on all test pairs", xlabel="Absolute error", ylabel="Count")
            fig.autofmt_xdate()
        ''' if task=='task1' else '''
            fig, axes = plt.subplots(2, 1, figsize=(11, 5))
            for ax, horizon in zip(axes, [1, 5]):
                values = predictions[predictions.horizon.eq(horizon)].head(240)
                ax.plot(values.target_time, values.actual, label="Actual")
                ax.plot(values.target_time, values.predicted, label="Model")
                ax.plot(values.target_time, values.persistence, label="Persistence", alpha=.5)
                ax.set(title=f"t+{horizon}: first 240 test predictions", ylabel="OT")
                ax.legend()
            fig.autofmt_xdate()
        ''') + '''
            plt.tight_layout()
            plt.show()
        ''', 'predictions')
        md('## 9. Đánh giá kết quả\n\n'+('Test **R²=0,9113**, **RMSE=0,8468 ± 0,0079**; giảm RMSE **10,38%** so với persistence. Kết quả đạt mục tiêu cải thiện 5% của plan, không phải điểm thấp trong phép so sánh này. Điều đó không khẳng định model tốt nhất trên mọi cách chia ETTh1.' if task=='task1' else
            'Test macro **R²=0,8749**, **RMSE=0,9823 ± 0,0171**; giảm RMSE **13,55%** so với persistence, đạt mục tiêu 5% của plan. t+5 khó hơn: RMSE trung bình **1,2581**, R² **0,8043**. Đây là hạn chế thực tế cần trình bày, không giấu trong một số trung bình.')+'\n\nCác hướng chưa chạy như lookback 24/72 giờ vẫn là đề xuất. Những thay đổi tiếp theo phải chọn bằng validation mới; không tinh chỉnh theo test đã xem.')
    else:
        md(('## 7. Benchmark test độc lập' if final_plate_test else '## 7. Kết quả hiện tại — chỉ validation')+'\n\nFull-plate accuracy đo đúng **toàn bộ chuỗi**; character accuracy đo đúng từng vị trí; CER dùng edit distance / số ký tự thật. Không dùng character accuracy thay cho accuracy toàn biển. '+('Các checkpoint đã khóa trước khi xem test; chỉ báo số seed thực sự có.' if final_plate_test else 'Các kết quả này chưa phải test cuối.'))
        code('''
            scores = run["scores"]
            display(pd.DataFrame([{
                "Full-plate (%)": 100*scores["plate_accuracy"],
                "Character (%)": 100*scores["character_accuracy"],
                "CER (%)": 100*scores["CER"],
                "Ảnh đánh giá": len(run["predictions"]),
                "Seed": settings["seed"]
            }]).round(3))
            print(f"Biển số còn sai: {100*(1-scores['plate_accuracy']):.2f}%")
            print("Mục tiêu trong plan: 90% full-plate —",
                  "ĐẠT" if scores["plate_accuracy"] >= .90 else "CHƯA ĐẠT")
        ''', 'benchmark')
        if final_plate_test:
            md('So sánh trên **cùng tập test đã khóa**. Đây là các giai đoạn warm-start với thời lượng học khác nhau, không phải ablation tách biệt tác dụng từng layer.')
            code('''
                from notebook_support import plate_test_comparison
                test_comparison = plate_test_comparison()
                display(test_comparison)
            ''', 'test_comparison')
        md('Run baseline attention cũ có một file metric validation được phục hồi từ CSV đã đối chiếu manifest; các run mới tính metric từ dự đoán thực tế. Phụ lục PDF dùng decoded one-hot, không hồi quy trên mã số ký tự:')
        code('''
            auxiliary = {key.removeprefix("aux_one_hot_"): value
                         for key, value in scores.items() if key.startswith("aux_one_hot_")}
            display(pd.DataFrame([auxiliary]).round(5))
        ''', 'auxiliary')
        md('## 8. Phân tích lỗi\n\nBa biển đúng đầu tiên và ba biển sai đầu tiên trong tập đánh giá đang báo cáo. Mũi tên là nhãn thật → dự đoán; không dùng ảnh test để chọn lại model.')
        code('''
            predictions = run["predictions"]
            plate_examples(data, predictions)
            position_accuracy = [np.mean([a[i] == b[i] for a, b in
                                 zip(predictions.true_text, predictions.predicted_text)]) for i in range(7)]
            display(pd.DataFrame({"Vị trí ký tự": range(1,8),
                                  "Accuracy (%)": np.array(position_accuracy)*100}).round(2))
            errors = predictions.edit_distance.value_counts().sort_index()
            display(pd.DataFrame({"Số ký tự chỉnh sửa": errors.index,
                                  "Số biển": errors.values,
                                  "Tỷ lệ (%)": 100*errors.values/len(predictions)}).round(2))
        ''', 'error_analysis')
        md('Attention cho biết decoder đang chú ý cột nào. Bản đồ sáng không tự chứng minh dự đoán đúng; cần đọc cùng ảnh và nhãn. Vị trí tỉnh/thành ít đa dạng có thể làm character accuracy trông cao, trong khi nhiều biển vẫn sai ở các ký tự khác.')
        code('''
            from IPython.display import Image
            attention_figure = run["_folder"]/(run["split"]+"_attention.png")
            if attention_figure.exists():
                display(Image(filename=str(attention_figure)))
        ''', 'attention_output')
        if final_plate_test:
            final=read_json(chosen/'test_metrics.json')
            frozen=read_json(selection_file)
            baseline_folder=ROOT/next(item['run'] for item in frozen['checkpoints'] if item['role']=='baseline')
            baseline_score=read_json(baseline_folder/'test_metrics.json')['plate_accuracy']
            md(f'## 9. Kết luận\n\nTrên **7.482 ảnh test**, model cuối đạt **{100*final["plate_accuracy"]:.2f}% đúng toàn biển**, **{100*final["character_accuracy"]:.2f}% đúng ký tự**, CER **{100*final["CER"]:.2f}%**. So với baseline attention trên cùng test, tăng **{100*(final["plate_accuracy"]-baseline_score):.2f} điểm phần trăm**. '+('Đạt' if final['plate_accuracy']>=.9 else 'Chưa đạt')+' mục tiêu 90% của plan ban đầu.\n\nCấu hình được chọn bằng validation trước khi mở test. Cải thiện là tổ hợp kiến trúc và fine-tuning, không khẳng định tác dụng riêng từng thành phần. Model cuối có **một seed fine-tune**, không có std qua ba seed để báo cáo.\n\n**Giới hạn:** biển cố định 7 ký tự, crop từ annotation, dữ liệu/split riêng. Ảnh mờ, nghiêng và ký tự giống nhau vẫn có lỗi; chưa đánh giá detector hoặc biển có độ dài khác. Không so trực tiếp con số này với bài báo dùng cohort/split khác.')
        else:
            md('## 9. Kết luận tạm thời\n\nBaseline attention ban đầu đạt 62,03% validation. Các model cải tiến giữ chi tiết chiều cao, thêm vùng trái–phải, residual CNN và vị trí cột; bảng phía trên ghi toàn bộ trial đã hoàn thành. Điểm hiện tại vẫn chỉ là **validation**, không phải benchmark test cuối.\n\nCần chọn checkpoint bằng validation và khóa trước test. Không gọi một seed fine-tune là ba lần train độc lập. Phạm vi là biển 7 ký tự với crop từ annotation, chưa có detector cả xe.')

    if forecast:
        md('### Nguồn\n\n[ETDataset — tác giả công bố ETTh1](https://github.com/zhouhaoyi/ETDataset). Notebook dùng cách chia 70/15/15 đã ghi trong plan; đây là benchmark nội bộ của project, không tự nhận là cùng giao thức của mọi bài báo. Chi tiết dữ liệu: [data/raw/README.md](../data/raw/README.md).')
    else:
        md('### Nguồn\n\n[CCPD — repository của tác giả](https://github.com/detectRecog/CCPD). Cohort 50.000 ảnh, nguồn mirror cố định và cách chia theo chuỗi được ghi tại [data/raw/README.md](../data/raw/README.md).')
    return nb.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Final Project (Windows .venv)', 'language': 'python', 'name': 'final-project-windows'},
        'language_info': {'name': 'python', 'version': '3.12.10'},
        'report_model_class': model_name,
        'report_model_source': model_source,
        'presentation_version': 2})


def main():
    backup = ROOT/'run/notebook_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True)
    for task in ['task2', 'task3']:
        path = ROOT/task/(task+'.ipynb')
        shutil.copy2(path, backup/path.name)
        nb.write(build(task), path)
        print('Updated', task)


if __name__ == '__main__':
    main()

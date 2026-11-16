"""Create report tables and a Word/Markdown draft only from completed run artifacts."""
import bootstrap
from pathlib import Path
import json
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bootstrap import ROOT
from utils import file_hash
from docx import Document
from docx.shared import Inches, Pt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    output=ROOT/'report/task3'
    figures=output/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    selection=read(ROOT/'run/task3/final_selection.json')
    data=pd.read_csv(ROOT/'data/splits/ccpd_subset.csv')
    benchmark=[]
    for item in selection['checkpoints']:
        folder=ROOT/item['run']
        assert file_hash(folder/'best.pt')==item['checkpoint_sha256']
        scores=read(folder/'test_metrics.json')
        meta=read(folder/'metadata.json')
        cfg=read(folder/'config.json')
        label={'legacy':'CNN–GRU attention','aligned':'Aligned CNN–GRU','refined':'Refined spatial CNN–GRU'}[cfg.get('architecture','legacy')]
        benchmark.append({'Model':label,'Role':item['role'],
                          'Seed':cfg['seed'], 'Full-plate (%)':100*scores['plate_accuracy'],
                          'Character (%)':100*scores['character_accuracy'], 'CER (%)':100*scores['CER'],
                          'Parameters':meta['parameters'],'Best epoch':meta['best_epoch'],
                          'Train seconds (run only)':meta['training_seconds'],
                          **{key:value for key,value in scores.items() if key.startswith('aux_')}})
    table=pd.DataFrame(benchmark)
    table.to_csv(output/'test_benchmark.csv',index=False,encoding='utf-8-sig')
    selected=ROOT/selection['selected_run']
    cfg=read(selected/'config.json')
    meta=read(selected/'metadata.json')
    scores=read(selected/'test_metrics.json')
    predictions=pd.read_csv(selected/'test_predictions.csv')
    history=pd.read_csv(selected/'history.csv')
    assert len(predictions)==7482 and predictions.image_path.is_unique
    correct=int(predictions.exact_match.sum())
    correct_text=f'{correct:,}'.replace(',','.')
    layers=[
        ('Input RGB','[B,3,64,192]'),
        ('Conv 3→32, ReLU, Pool 2×2','[B,32,32,96]'),
        ('Conv 32→64, ReLU, Pool 2×2','[B,64,16,48]'),
        ('Conv 64→128, ReLU, Pool 2×1','[B,128,8,48]'),
        ('Residual: Conv–ReLU–Conv, cộng hệ số 0,1','[B,128,8,48]'),
        ('Ghép chiều cao + Linear 1024→128 + vị trí cột','[B,48,128]'),
        ('AdaptiveAvgPool1d(7), trộn 0,5 local + 0,5 global','[B,7,128]'),
        ('GRU decoder, hidden ban đầu = global context','[B,7,128]'),
        ('Dot attention: query decoder, memory 48 cột','[B,7,128]'),
        ('Ghép decoder/context, Linear 256→67','[B,7,67]'),
        ('Argmax, tra vocabulary','B chuỗi × 7 ký tự'),
    ]
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    x=np.arange(len(table))
    axes[0].bar(x,table['Full-plate (%)'],color=['#8196a8','#5799b3','#277a8a'][:len(table)])
    axes[0].set(xticks=x,xticklabels=table.Model.str.replace(' CNN–GRU','',regex=False),ylabel='Full-plate accuracy (%)',ylim=(0,100),title='Locked test: 7,482 images')
    for i,value in enumerate(table['Full-plate (%)']): axes[0].text(i,value+1,f'{value:.2f}%',ha='center')
    axes[1].plot(history.epoch,100*history.val_plate_accuracy,marker='o',markersize=3)
    axes[1].axvline(meta['best_epoch'],color='grey',linestyle='--')
    axes[1].set(xlabel='Fine-tuning epoch',ylabel='Validation full-plate (%)',title='Selection used validation only')
    fig.tight_layout();fig.savefig(figures/'benchmark_and_validation.png',dpi=170);plt.close(fig)
    for name in ['test_examples.png','test_attention.png','learning_curves.png']:
        shutil.copy2(selected/name,figures/name)
    positions=pd.read_csv(selected/'test_positions.csv')
    positions.to_csv(output/'position_accuracy.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,3))
    ax.bar(positions.position,positions.accuracy*100,color='#277a8a')
    ax.set(xlabel='Character position',ylabel='Accuracy (%)',ylim=(0,100),xticks=range(1,8),title='Accuracy at each of the seven output positions')
    fig.tight_layout();fig.savefig(figures/'position_accuracy.png',dpi=170);plt.close(fig)
    improvement=table[table.Role.eq('selected_model')].iloc[0]['Full-plate (%)']-table.iloc[0]['Full-plate (%)']
    split_table=data.groupby('split').agg(images=('text','size'),plate_groups=('text','nunique')).reindex(['train','validation','test'])
    sections=[
        ('1. Bài toán và dữ liệu',
         'Nhận dạng toàn bộ chuỗi bảy ký tự từ ảnh biển số. Dataset là cohort 50.000 ảnh CCPD-Base, gồm ký tự tỉnh, một chữ cái và năm chữ/số tiếp theo. Crop dùng bounding box annotation; không tuyên bố đã xây detector từ ảnh cả xe. Chia theo toàn chuỗi biển số, không trùng plate giữa train/validation/test. Đây là benchmark nội bộ của cohort đã chọn, không phải giao thức test chính thức để so trực tiếp với mọi bài báo.'),
        ('2. Tiền xử lý và kiến trúc',
         'Ảnh crop được chuyển RGB, resize 192×64 và chia 255. CNN có ba block Conv–ReLU–Pool, tạo [B,128,8,48]. Bản baseline lấy trung bình chiều cao và lặp global context bảy lần. Bản aligned dùng Linear(8×128,128) giữ thông tin trên/dưới của ký tự; AdaptiveAvgPool1d(7) tạo bảy vùng trái–phải. Bản refined thêm nhánh hai Conv 3×3 residual (hệ số 0,1) và vector vị trí cho 48 cột. Nhánh residual và vị trí khởi tạo bằng 0, giữ nguyên hàm dự đoán của checkpoint aligned trước fine-tune. Mỗi bước GRU nhận hỗn hợp local/global (hệ số 0,5), attention trên 48 cột và Linear(256,67). Output logits [B,7,67], argmax tạo đủ bảy ký tự. Không dùng nhãn tương lai hoặc dịch vụ OCR ngoài.'),
        ('3. Thiết lập huấn luyện',
         f'Khởi tạo theo chuỗi baseline attention → aligned (4 epoch) → refined, tất cả chỉ học train; không khởi tạo từ test. Seed {cfg["seed"]}; batch {cfg["batch_size"]}; Adam với CNN lr={cfg["cnn_learning_rate"]}, projection lr={cfg["projection_learning_rate"]}, decoder/classifier lr={cfg["learning_rate"]}, residual lr={cfg.get("refinement_learning_rate","N/A")}, vị trí lr={cfg.get("position_learning_rate","N/A")}; cross-entropy, gradient clipping 1.0. Scheduler giảm LR theo validation CE. Ngân sách thực tế là {meta["last_completed_epoch"]} epoch fine-tune, giới hạn cấu hình {cfg["epochs"]}; chốt sau review validation và giữ checkpoint epoch {meta["best_epoch"]} theo full-plate accuracy, tie-break CER. Một seed fine-tune, không phải ba lần train độc lập. Thời gian bảng chỉ thuộc từng run; warm start còn dùng công huấn luyện baseline trước đó. Batch nhỏ và nghỉ giữa batch làm thời gian không thể so như benchmark tốc độ thuần.'),
        ('4. Kết quả test độc lập',
         f'Cấu hình và hash checkpoint được khóa trước khi mở test. Test gồm 7.482 ảnh, giữ nguyên toàn bộ ảnh khó. Mô hình mới đọc đúng hoàn toàn {correct_text}/7.482 ảnh, còn {len(predictions)-correct:,} ảnh sai ít nhất một ký tự; full-plate accuracy {100*scores["plate_accuracy"]:.2f}%, character accuracy {100*scores["character_accuracy"]:.2f}% và CER {100*scores["CER"]:.2f}%. So với baseline trên cùng tập test, tăng {improvement:.2f} điểm phần trăm full-plate accuracy. Không chọn lại model bằng các điểm test này.'),
        ('5. Định nghĩa metric và yêu cầu PDF',
         'Full-plate accuracy là tỷ lệ đúng cả bảy ký tự. Character accuracy tính theo vị trí. CER là tổng Levenshtein edit distance chia tổng ký tự thật. R²/NSE/MAE/RMSE trong phụ lục được tính trên decoded binary one-hot đã flatten, không hồi quy trên character ID. R² và NSE bằng nhau theo công thức residual. Các metric one-hot chỉ là chẩn đoán phân loại phụ, không có ý nghĩa khoảng cách số học giữa biển số và không thay cho accuracy toàn biển.'),
        ('6. Phân tích và giới hạn',
         'Kết quả cải thiện là tổ hợp thay đổi kiến trúc và fine-tuning, không phải một ablation tách riêng tác dụng từng thành phần. Dữ liệu cố định bảy ký tự, crop bằng annotation và đa số cùng dạng biển; cần thận trọng khi suy rộng sang quốc gia, biển biến độ dài hoặc detector thực tế. Ảnh mờ, nghiêng và ký tự gần hình vẫn gây lỗi. Chỉ có một seed fine-tune cho model mới nên chưa ước lượng được phương sai giữa các lần train. Hình minh họa lấy các trường hợp đầu theo thứ tự cố định, không chọn thủ công chỉ ảnh đẹp.'),
        ('7. Nguồn và khả năng tái lập',
         'Yêu cầu: assignments/Final Project Requirements 2026.pdf. Dataset: https://github.com/detectRecog/CCPD . Adaptive pooling: https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html . Code: task3/code/refined_model.py, task3.py và train_advanced.py; config: task3/config/refined.json. Manifest/vocabulary ở data/splits. Checkpoint, source snapshot, history, prediction CSV, metric và hash khóa test ở run/task3. Không sửa assignment PDF hoặc template Word gốc.')
    ]
    lines=['# Task 3 — CNN encoder + GRU decoder nhận dạng biển số','',f'Bản nháp từ artifact thật, tạo {datetime.now().isoformat(timespec="seconds")}.','']
    for heading,text in sections:
        lines.extend(['## '+heading,'',text,''])
        if heading.startswith('1.'):
            lines += ['| Tập | Ảnh | Plate groups |','| --- | ---: | ---: |']
            lines += [f'| {index} | {int(row.images)} | {int(row.plate_groups)} |' for index,row in split_table.iterrows()]
            lines += ['']
        if heading.startswith('4.'):
            lines += ['| Model | Seed | Full-plate % | Character % | CER % |','| --- | ---: | ---: | ---: | ---: |']
            lines += [f'| {row["Model"]} | {row["Seed"]} | {row["Full-plate (%)"]:.3f} | {row["Character (%)"]:.3f} | {row["CER (%)"]:.3f} |' for _,row in table.iterrows()]
            lines += ['', '![Benchmark và validation](figures/benchmark_and_validation.png)','', '![Dự đoán đúng và sai](figures/test_examples.png)','']
        if heading.startswith('2.'):
            lines += ['| Layer / phép biến đổi | Shape |','| --- | --- |']
            lines += [f'| {name} | `{shape}` |' for name,shape in layers]
            lines += ['']
        if heading.startswith('5.'):
            aux=[key for key in table if key.startswith('aux_')]
            table[['Model']+aux].to_csv(output/'auxiliary_one_hot_metrics.csv',index=False,encoding='utf-8-sig')
            lines += ['| Model | R² phụ | NSE phụ | MAE phụ | RMSE phụ |','| --- | ---: | ---: | ---: | ---: |']
            for _,row in table.iterrows():
                values=' | '.join(f'{row["aux_one_hot_"+key]:.5f}' for key in ['R2','NSE','MAE','RMSE'])
                lines += [f'| {row["Model"]} | {values} |']
            lines += ['']
        if heading.startswith('6.'):
            lines += ['![Accuracy theo vị trí](figures/position_accuracy.png)','',
                      '![Attention trên 48 cột ảnh](figures/test_attention.png)','']
    (output/'task3_report.md').write_text('\n'.join(lines),encoding='utf-8')
    doc=Document()
    doc.styles['Normal'].font.name='Calibri'
    doc.styles['Normal'].font.size=Pt(11)
    doc.add_heading('TASK 3 — NHẬN DẠNG BIỂN SỐ',0)
    doc.add_paragraph('CNN Encoder – GRU Decoder – Attention\nBản nháp dựa trên kết quả thực nghiệm đã khóa.')
    for heading,text in sections:
        doc.add_heading(heading,level=1)
        doc.add_paragraph(text)
        if heading.startswith('1.'):
            t=doc.add_table(rows=1,cols=3);t.style='Light Shading Accent 1'
            for cell,value in zip(t.rows[0].cells,['Tập','Ảnh','Plate groups']):cell.text=value
            for index,row in split_table.iterrows():
                for cell,value in zip(t.add_row().cells,[index,int(row.images),int(row.plate_groups)]):cell.text=str(value)
        if heading.startswith('2.'):
            t=doc.add_table(rows=1,cols=2);t.style='Light Shading Accent 1'
            for cell,value in zip(t.rows[0].cells,['Layer / phép biến đổi','Shape']):cell.text=value
            for values in layers:
                for cell,value in zip(t.add_row().cells,values):cell.text=value
        if heading.startswith('4.'):
            t=doc.add_table(rows=1,cols=4);t.style='Light Shading Accent 1'
            for cell,value in zip(t.rows[0].cells,['Model','Full-plate %','Character %','CER %']):cell.text=value
            for _,row in table.iterrows():
                values=[row['Model']]+[f'{row[key]:.3f}' for key in ['Full-plate (%)','Character (%)','CER (%)']]
                for cell,value in zip(t.add_row().cells,values):cell.text=str(value)
            for name in ['benchmark_and_validation.png','test_examples.png','position_accuracy.png','test_attention.png']:
                doc.add_picture(str(figures/name),width=Inches(6.2))
        if heading.startswith('5.'):
            t=doc.add_table(rows=1,cols=5);t.style='Light Shading Accent 1'
            for cell,value in zip(t.rows[0].cells,['Model','R²','NSE','MAE','RMSE']):cell.text=value
            for _,row in table.iterrows():
                values=[row['Model']]+[f'{row["aux_one_hot_"+key]:.5f}' for key in ['R2','NSE','MAE','RMSE']]
                for cell,value in zip(t.add_row().cells,values):cell.text=str(value)
    doc.save(output/'task3_report.docx')
    print('Created report/task3/task3_report.md, .docx, CSV tables and figures from locked test artifacts.')


if __name__=='__main__':main()

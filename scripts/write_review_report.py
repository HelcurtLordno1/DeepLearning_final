"""Review the current notebooks using measured results; do not retrain."""
import bootstrap
from datetime import datetime
import pandas as pd
from notebook_support import (
    ROOT, read_json, forecast_benchmark, benchmark_tables, load_run,
)


def main():
    folder = ROOT/'run/notebook_review'
    checks = read_json(folder/'presentation_checks.json')
    rows = []
    for task in ['task1', 'task2']:
        table = forecast_benchmark(task)
        _, seeds, gain = benchmark_tables(table)
        assert sorted(seeds.index.tolist()) == [42, 123, 2026]
        values = table[table.model.eq('model') & table.horizon.eq('macro')]
        rows.append({
            'Task': task, 'Split': 'test', 'Seeds': len(values),
            **{key+'_mean': values[key].mean() for key in ['R2','NSE','MAE','RMSE']},
            'RMSE_sample_std': values.RMSE.std(ddof=1),
            'Persistence_RMSE': table[table.model.eq('persistence') & table.horizon.eq('macro')].iloc[0].RMSE,
            'RMSE_reduction_percent': gain, 'Plan_goal_met': gain >= 5,
        })
    selection = read_json(ROOT/'run/task3/final_selection.json')
    config = read_json(ROOT/selection['selected_run']/'config.json')
    run = load_run('task3', architecture=config.get('architecture','legacy'))
    assert run['split'] == 'test', 'Complete the locked test before writing the final review.'
    scores = run['scores']
    baseline = read_json(ROOT/selection['checkpoints'][0]['run']/'test_metrics.json')
    gain = 100*(scores['plate_accuracy']-baseline['plate_accuracy'])
    rows.append({
        'Task': 'task3', 'Split': 'test', 'Seeds': len(selection['final_seeds']),
        'Full_plate_percent': 100*scores['plate_accuracy'],
        'Character_percent': 100*scores['character_accuracy'],
        'CER_percent': 100*scores['CER'], 'Full_plate_gain_points': gain,
        'Plan_goal_met': scores['plate_accuracy'] >= .9,
    })
    pd.DataFrame(rows).to_csv(folder/'benchmark_assessment.csv', index=False)
    lines = [
        '# Kiểm tra notebook và benchmark', '',
        f'Cập nhật {datetime.now().isoformat(timespec="seconds")}.', '',
        '## Trình bày đã kiểm tra', '',
        'Đối chiếu school_work6/M1.ipynb: section đánh số, EDA train, model hiện từng layer, shape, loss/optimizer, đường học, validation và kết quả test. PowerShell và quản lý artifact nằm ngoài notebook.', '',
        '| Notebook | Ô code | Ô Markdown | Hình inline | Warning/error | Ô code dài nhất |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for check in checks:
        lines.append(f"| {check['task']} | {check['code_cells']} | {check['markdown_cells']} | {check['inline_figures']} | {check['stderr_outputs']} / {check['errors']} | {check['largest_code_cell_lines']} dòng |")
    lines += [
        '', 'Đã chạy Run All bằng kernel Windows mới với TRAIN=False: EDA, đọc benchmark, nạp checkpoint CPU và forward hai mẫu; không gọi đây là train mới. Model, attention, training step và optimizer factory được đối chiếu AST với code thực. Không có Python code fence bị đặt nhầm trong Markdown.', '',
        'EDA Task 1–2 chỉ dùng train: chuỗi OT, phân bố, tương quan, autocorrelation và split/timestamp. Task 3 có crop mẫu, nhãn đủ bảy vị trí, split theo plate, mất cân bằng tỉnh, lỗi theo vị trí và edit distance. Không suy diễn mùa vụ chỉ từ autocorrelation cao.', '',
        'Preview: [Task 1](task1.html), [Task 2](task2.html), [Task 3](task3.html). Các ảnh inline ở figures/.', '',
        '## Benchmark test thật', '',
        '| Task | Kết quả | So baseline |', '| --- | --- | --- |',
    ]
    for row in rows[:2]:
        lines.append(f"| {row['Task']} | Test, 3 seed: R² {row['R2_mean']:.4f}; RMSE {row['RMSE_mean']:.4f} ± {row['RMSE_sample_std']:.4f} | Giảm RMSE {row['RMSE_reduction_percent']:.2f}% so persistence |")
    lines += [
        f"| Task 3 | Test, 1 seed fine-tune: toàn biển {100*scores['plate_accuracy']:.2f}%; ký tự {100*scores['character_accuracy']:.2f}%; CER {100*scores['CER']:.2f}% | Tăng {gain:.2f} điểm phần trăm so CNN–GRU attention |", '',
        'Task 1–2 giữ nguyên benchmark ba seed. R² không phải classification accuracy. R²/NSE bằng nhau theo công thức residual. Task 2 phân biệt macro với pooled và hiển thị đủ năm horizon.', '',
        'Task 3 đã khóa cấu hình/hash checkpoint trước khi mở test. CNN–GRU mới giữ chi tiết chiều cao, thêm bảy vùng ảnh cho decoder, nhánh CNN residual nhỏ và vị trí cột. Warm start theo baseline → aligned → refined, không phải các lần train độc lập. Tăng điểm là kết quả của cả kiến trúc và fine-tuning, chưa phải ablation tách từng thành phần.', '',
        'Chỉ có một seed fine-tune ở Task 3 nên không báo ± std. Protocol là cropped CCPD-Base 50.000 ảnh, grouped split theo chuỗi biển; không phải detector ảnh cả xe hoặc bảng SOTA chung. CER dùng Levenshtein; bốn metric one-hot trong phụ lục được ghi rõ là phụ trợ.', '',
        '## Bố cục và lịch sử thực nghiệm', '',
        'Notebook vẫn ở taskX/; Python tương ứng ở taskX/code/, helper ở src/, launcher/kiểm tra ở scripts/, requirements ở environment/. Không còn .py/.ps1/.txt rải tại root; kernel Windows .venv giữ nguyên.', '',
        'Trial vị trí lúc 22:14 ngày 16/11 bị ngắt do mất monitor, chưa hoàn tất validation; giữ nguyên trạng thái và artifact. Các phiên phát triển sau là phiên ngắn có checkpoint/resume và kiểm tra tài nguyên mới. Không tự lặp safety stop; không suy ra sự cố phần cứng cũ đã được sửa.', '',
        'Bản review trước được giữ ở REVIEW_before_task3_development.md. Báo cáo mới: [Word](../../report/task3/task3_report.docx), [Markdown](../../report/task3/task3_report.md).', '',
        'Bằng chứng: [kiểm tra output](presentation_checks.json), [đối chiếu source](source_agreement.json), [bảng đánh giá](benchmark_assessment.csv), [audit số liệu](../verification/results_audit.json).',
    ]
    (folder/'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Updated review from completed locked test results.')


if __name__ == '__main__':
    main()

"""Inspect notebook structure and the actual saved benchmark, without training."""

import bootstrap  # Resolve src/ and the project root.
import json
from pathlib import Path
import pandas as pd
import nbformat
ROOT = Path(__file__).resolve().parents[1]
for relative in ['assignments/simple_examples_code/school_work6/M1.ipynb', 'task1/task1.ipynb', 'task2/task2.ipynb', 'task3/task3.ipynb']:
    book = nbformat.read(ROOT/relative, as_version=4)
    print('\nNOTEBOOK:', relative)
    for index, cell in enumerate(book.cells):
        print(index, cell.cell_type, 'lines='+str(len(cell.source.splitlines())), cell.source.splitlines()[0][:100] if cell.source else '')
        if cell.cell_type == 'markdown' and '```python' in cell.source:
            print('  CODE FENCE IN MARKDOWN:', cell.source[:200])
        if cell.cell_type == 'code':
            compile(cell.source, relative+str(index), 'exec')
            print('  outputs:', [(o.output_type, len(o.get('text','')), list(o.get('data',{}))) for o in cell.outputs])
            for output in cell.outputs:
                if output.output_type=='stream' and output.name=='stderr': print('  STDERR:', output.text[:200])
    if relative.startswith('assignments'):
        print('REFERENCE MARKDOWN:\n', '\n'.join(c.source for c in book.cells if c.cell_type=='markdown'))
metrics = pd.read_csv(ROOT/'run/summary/all_test_metrics.csv')
choices = json.loads((ROOT/'run/selected_configs.json').read_text())
for task, choice in choices.items():
    selected=metrics[(metrics.task==task)&(metrics.name==choice['config']['name'])]
    print('\nBENCHMARK',task)
    print(selected[selected.horizon=='macro'][['model','seed','R2','MAE','RMSE']].to_string(index=False))
    model=selected[(selected.model=='model')&(selected.horizon=='macro')]
    base=selected[(selected.model=='persistence')&(selected.horizon=='macro')].iloc[0].RMSE
    print('RMSE improvement %:',100*(1-model.RMSE.mean()/base))
    print(selected[selected.model=='model'].groupby('horizon')[['R2','MAE','RMSE']].mean().round(6))

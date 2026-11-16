"""Execute presentation notebooks on Windows CPU and inspect the real outputs."""

import bootstrap  # Resolve src/ and the project root.
from pathlib import Path
import asyncio
import ast
import base64
import json
import re
import sys
import importlib.util
import html
import mistune
import nbformat
from nbclient import NotebookClient
from utils import ROOT, save_json, file_hash


def export_review(book, path):
    # A self-contained HTML preview; no browser, GPU or extra template install.
    markdown = mistune.create_markdown(plugins=['table'])
    blocks = []
    for cell in book.cells:
        if cell.cell_type == 'markdown':
            blocks.append('<section>'+markdown(cell.source)+'</section>')
            continue
        blocks.append('<pre class="code">'+html.escape(cell.source)+'</pre>')
        for output in cell.outputs:
            data = output.get('data', {})
            if 'image/png' in data:
                blocks.append('<img alt="Notebook figure" src="data:image/png;base64,'+data['image/png']+'">')
            elif 'text/html' in data:
                blocks.append('<div class="table">'+data['text/html']+'</div>')
            elif output.output_type == 'stream':
                blocks.append('<pre class="output">'+html.escape(output.text)+'</pre>')
            elif 'text/plain' in data:
                blocks.append('<pre class="output">'+html.escape(data['text/plain'])+'</pre>')
    css = '''body{max-width:1080px;margin:36px auto;padding:0 24px;color:#222;background:white;
    font:16px/1.65 system-ui,Segoe UI,sans-serif}h1{font-size:29px}h2{font-size:23px;margin-top:42px}
    pre{font:14px/1.6 Consolas,monospace;white-space:pre;overflow:auto;padding:16px}
    .code{background:#f5f7f9;border-left:3px solid #507b99;border-radius:4px}
    .output{background:#fff;border:1px solid #e2e6e9}img{max-width:100%;height:auto;display:block;margin:18px auto}
    .table{overflow:auto;margin:16px 0 30px}table{border-collapse:collapse;font-size:14px}
    th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}th{background:#f1f4f6}
    tbody tr:nth-child(even){background:#f9fafb}code{font-family:Consolas,monospace}'''
    path.write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Notebook review</title>'
                    '<style>'+css+'</style><body>'+''.join(blocks)+'</body></html>', encoding='utf-8')


def main():
    assert sys.platform == 'win32'
    assert Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    folder = ROOT/'run/notebook_review'
    (folder/'figures').mkdir(parents=True, exist_ok=True)
    records, checks = [], []
    for task in ['task1', 'task2', 'task3']:
        path = ROOT/task/(task+'.ipynb')
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(path), 'exec')
                assert 'argparse' not in cell.source and '__main__' not in cell.source
            else:
                assert '```python' not in cell.source, 'Executable-looking code in Markdown'
        probe = nbformat.v4.new_code_cell('''
import json
assert TRAIN is False
assert sys.platform == "win32"
assert Path(sys.executable).resolve() == (ROOT/".venv/Scripts/python.exe").resolve()
assert not torch.cuda.is_initialized()
print(json.dumps({"platform": sys.platform, "cuda_initialized": torch.cuda.is_initialized()}))
''')
        book.cells.append(probe)
        print('Checking', task, flush=True)
        try:
            NotebookClient(book, timeout=180, kernel_name='final-project-windows',
                           resources={'metadata': {'path': str(ROOT/task)}}).execute()
        except Exception:
            nbformat.write(book, folder/(task+'_failed.ipynb'))
            raise
        proof = json.loads(book.cells.pop().outputs[0].text)
        figures, warnings = 0, []
        for index, cell in enumerate(book.cells):
            if cell.cell_type != 'code':
                continue
            for output in cell.outputs:
                assert output.output_type != 'error'
                if output.output_type == 'stream' and output.name == 'stderr':
                    warnings.append(output.text)
                if 'image/png' in output.get('data', {}):
                    figures += 1
                    image = output.data['image/png']
                    (folder/'figures'/f'{task}_{index:02d}_{figures}.png').write_bytes(base64.b64decode(image))
        assert figures >= 4, (task, 'Missing EDA/learning/prediction figures', figures)
        assert not warnings, (task, warnings)
        nbformat.write(book, path)
        export_review(book, folder/(task+'.html'))
        records.append({'task': task, 'notebook_sha256': file_hash(path),
                        'executed_code_cells': sum(c.cell_type=='code' for c in book.cells),
                        'kernel': 'final-project-windows', 'full_train_default': False,
                        **proof, 'note': 'EDA, saved benchmark, loaded checkpoints and tiny CPU forward passes. No retraining.'})
        checks.append({'task': task, 'code_cells': records[-1]['executed_code_cells'],
                       'markdown_cells': sum(c.cell_type=='markdown' for c in book.cells),
                       'inline_figures': figures, 'stderr_outputs': 0, 'errors': 0,
                       'python_in_markdown': False,
                       'largest_code_cell_lines': max(len(c.source.splitlines()) for c in book.cells if c.cell_type=='code')})
        save_json(ROOT/'run/notebook_execution.json', records)
        save_json(folder/'presentation_checks.json', checks)
        print(task, 'PASS;', figures, 'inline figures; no warnings; CUDA not initialized.', flush=True)
    from verify_results import audit_notebooks
    save_json(folder/'source_agreement.json', audit_notebooks())


if __name__ == '__main__':
    main()

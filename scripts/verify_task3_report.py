"""Compare the delivered Word/CSV report with locked test measurements."""
import bootstrap
import io
import zipfile
import pandas as pd
from docx import Document
from PIL import Image
from notebook_support import ROOT, read_json
from utils import save_json

selection = read_json(ROOT/'run/task3/final_selection.json')
folder = ROOT/'report/task3'
benchmark = pd.read_csv(folder/'test_benchmark.csv')
assert len(benchmark) == len(selection['checkpoints']) == 3
for (_, row), item in zip(benchmark.iterrows(), selection['checkpoints']):
    scores = read_json(ROOT/item['run']/'test_metrics.json')
    for column, metric in [('Full-plate (%)','plate_accuracy'),
                           ('Character (%)','character_accuracy'),('CER (%)','CER')]:
        assert abs(row[column]-100*scores[metric]) < 1e-9

doc = Document(folder/'task3_report.docx')
assert len(doc.tables) == 4 and len(doc.inline_shapes) == 4
for cells, (_, row) in zip(doc.tables[2].rows[1:], benchmark.iterrows()):
    assert [cell.text for cell in cells.cells] == [row['Model']] + [
        f'{row[key]:.3f}' for key in ['Full-plate (%)','Character (%)','CER (%)']]
assert '7.132/7.482' in '\n'.join(p.text for p in doc.paragraphs)
with zipfile.ZipFile(folder/'task3_report.docx') as archive:
    assert archive.testzip() is None
    images = [name for name in archive.namelist() if name.startswith('word/media/')]
    assert len(images) == 4
    for name in images:
        with Image.open(io.BytesIO(archive.read(name))) as image:
            image.verify()
assert len(pd.read_csv(folder/'position_accuracy.csv')) == 7
save_json(ROOT/'run/verification/task3_report.json', {
    'word_tables_match_test_csv': True, 'test_csv_matches_locked_metrics': True,
    'word_tables': len(doc.tables), 'embedded_figures': len(images),
    'readable_embedded_images': True, 'completed_fine_tuning_seeds': [42],
})
print('Word/CSV report: PASS; 4 tables, 4 verified images, locked test numbers match.')

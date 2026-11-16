
import bootstrap  # Resolve src/ and the project root.
from pathlib import Path
import importlib.util
root=Path(__file__).resolve().parents[1]
pdf=root/'assignments/Final Project Requirements 2026.pdf'
for module in ['pypdf','fitz','PyPDF2']:
    if importlib.util.find_spec(module):
        if module=='fitz':
            import fitz
            text='\n'.join(page.get_text() for page in fitz.open(pdf))
        else:
            lib=__import__(module)
            text='\n'.join(page.extract_text() for page in lib.PdfReader(pdf).pages)
        print(text)
        break
else:
    raise RuntimeError('No PDF reader installed')

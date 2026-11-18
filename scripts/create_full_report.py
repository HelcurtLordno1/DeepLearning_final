"""Build the English Task 2/3 submission from reviewed text and saved figures.

Requires the lecturer's local template and saved notebook output for the new EDA.
No model training or test inference is performed.
"""
import bootstrap
from pathlib import Path
import json,re,hashlib,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from bootstrap import ROOT

FOLDER=ROOT/'report/full_project'
FIGURES=FOLDER/'figures'
TEMPLATE=ROOT/'assignments/Final Major Assignment Report 2026.docx'
OUTPUT=FOLDER/'Group6_Final_Major_Assignment_Report_2026_EN.docx'


def main():
    blocks=json.loads((FOLDER/'report_content.json').read_text(encoding='utf-8'))
    for role in ['eda_window','eda_persistence']:
        shutil.copy2(ROOT/'run/notebook_review/figures'/f'task2_{role}.png',FIGURES/f'task2_{role}.png')
    fig,ax=plt.subplots(figsize=(8.5,3.7));ax.axis('off')
    stages=[
        'RGB plate [B,3,64,192] → 3 CNN blocks → [B,128,8,48]',
        'Residual CNN + height projection + column positions → memory [B,48,128]',
        '7 local/global region inputs → GRU decoder → states [B,7,128]',
        'Attention over 48 columns + states → Linear(256,67) → 7 characters']
    for i,text in enumerate(stages):
        y=.88-i*.25
        ax.text(.5,y,text,ha='center',va='center',fontsize=11,
                bbox=dict(boxstyle='round,pad=.55',fc='#E7EEF5',ec='#397b99'))
        if i<3:ax.annotate('',xy=(.5,y-.17),xytext=(.5,y-.07),arrowprops=dict(arrowstyle='->',color='#397b99'))
    fig.tight_layout();fig.savefig(FIGURES/'task3_architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
    doc=Document(TEMPLATE)
    for p in list(doc.paragraphs)[16:]:p._p.getparent().remove(p._p)
    for t in list(doc.tables):t._tbl.getparent().remove(t._tbl)
    for n in ['Normal','Body Text']:
        style=doc.styles[n];style.font.name='Times New Roman';style.font.size=Pt(13)
        style.paragraph_format.line_spacing=1.12;style.paragraph_format.space_after=Pt(6)
    for level,size in [(1,15),(2,14),(3,13),(4,13)]:
        style=doc.styles[f'Heading {level}'];style.font.name='Times New Roman';style.font.size=Pt(size)
        style.font.bold=True;style.font.color.rgb=RGBColor.from_string('17365D')
        style.paragraph_format.keep_with_next=True
    cover=doc.paragraphs
    texts={5:'TOPIC:',6:'ATTENTION-BASED MULTI-STEP FORECASTING\nAND LICENSE PLATE RECOGNITION',7:'Instructor: Nguyễn Kim Ngân',8:'Class: DSEB 65A | Group 6',9:'11230548 — Đinh Nam Khánh (Group leader)',10:'11230565 — Hồ Đức Mạnh',11:'11230567 — Khổng Gia Minh',12:'11230527 — Đỗ Tuấn Đạt',13:'',14:''}
    for i,text in texts.items():
        p=cover[i];p.clear();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.left_indent=Inches(0);p.paragraph_format.right_indent=Inches(0)
        p.paragraph_format.space_after=Pt(5)
        r=p.add_run(text);r.font.name='Times New Roman';r.font.size=Pt(15 if i==6 else 13);r.bold=i<9
    p=cover[15]
    for r in list(p.runs):p._p.remove(r._r)
    r=p.add_run('Hanoi, November 2026');r.font.name='Times New Roman';r.font.size=Pt(14);r.bold=True
    table_number=figure_number=0
    for block_index,block in enumerate(blocks):
        kind=block['kind']
        if kind=='heading':
            p=doc.add_heading(block['text'],block['level'])
            p.paragraph_format.page_break_before=block.get('newpage',False)
            for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(15 if block['level']==1 else 13);r.bold=True
        elif kind=='paragraph':
            p=doc.add_paragraph(block['text']);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
            if 'https://' in block['text']:p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            if block['text'].startswith('hₜ =') or block['text'].startswith(('aⱼᵢ =','ŷₜ₊ₕ =','MAE =','R² =','Full-plate accuracy =','Character accuracy =','CER =')):
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif kind=='image':
            figure_number+=1
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(FIGURES/block['file']),width=Inches(5.8))
            p=doc.add_paragraph(f'Figure {figure_number}. '+block['caption']);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:r.font.size=Pt(11);r.italic=True
        elif kind=='table':
            table_number+=1;p=doc.add_paragraph(f'Table {table_number}. '+block['caption']);p.paragraph_format.keep_with_next=True
            for r in p.runs:r.font.size=Pt(11);r.bold=True
            t=doc.add_table(rows=1,cols=len(block['headers']));t.style='Table Grid'
            for c,v in zip(t.rows[0].cells,block['headers']):c.text=v
            t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for row in block['rows']:
                for c,v in zip(t.add_row().cells,row):c.text=v
            for i,row in enumerate(t.rows):
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                for c in row.cells:
                    for p in c.paragraphs:
                        p.paragraph_format.line_spacing=1;p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                        if len(t.rows)<=8:p.paragraph_format.keep_with_next=(i<len(t.rows)-1)
                        for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(10.5);r.bold=i==0
                    if i==0:
                        shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EEF5');c._tc.get_or_add_tcPr().append(shade)
        # Add a new Word TOC after the executive summary, before Part 1.
        next_index=block_index+1
        if next_index<len(blocks) and blocks[next_index].get('text')=='Part 1: Theoretical Background':
            p=doc.add_paragraph('Report Contents');p.paragraph_format.page_break_before=True
            for r in p.runs:r.bold=True;r.font.size=Pt(14)
            p=doc.add_paragraph();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'TOC \\o "1-4" \\h \\z \\u');p._p.append(field)
    for p in doc.paragraphs:
        previous=p._p.getprevious()
        if previous is not None and previous.tag.endswith('}tbl'):p.paragraph_format.space_before=Pt(6)
    for s in list(doc.sections)[1:]:
        s.footer.is_linked_to_previous=False
        for child in list(s.footer._element):s.footer._element.remove(child)
        p=s.footer.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.add_run('Group 6 • Deep Learning | ')
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');p._p.append(field)
        for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(10)
    doc.core_properties.title='Attention-Based Multi-Step Forecasting and License Plate Recognition'
    doc.core_properties.author='Group 6 — DSEB 65A';doc.core_properties.language='en-US'
    doc.save(OUTPUT)
    (FOLDER/'build_audit.json').write_text(json.dumps({'active_tasks':['task2','task3'],'tables':table_number,'figures':figure_number,'template_sha256':hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),'training_started':False,'source':'reviewed English content and saved experiment artifacts'},indent=2),encoding='utf-8')
    print('Created English Task 2/3 report:',table_number,'tables;',figure_number,'figures.')


if __name__=='__main__':main()

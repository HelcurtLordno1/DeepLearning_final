"""Cheap path/import checks after moving support files."""
import bootstrap
from pathlib import Path
import ast
import json
import sys
from bootstrap import ROOT
for folder in ['src','scripts','task2/code','task3/code']:
    for path in (ROOT/folder).glob('*.py'):
        compile(path.read_text(encoding='utf-8'),str(path),'exec')
from utils import ROOT as data_root
assert data_root == ROOT
from task2.code.task2 import ForecastSeq2Seq
from task3.code.task3 import PlateSeq2Seq, AlignedPlateSeq2Seq
from task3.code.refined_model import RefinedPlateSeq2Seq
import torch
torch.set_num_threads(2)
config=json.loads((ROOT/'task3/config/aligned.json').read_text())
old_config={**config,'position':False,'flat_context':False,'alignment':0.0}
old=PlateSeq2Seq(67,old_config)
new=AlignedPlateSeq2Seq(67,old_config)
saved=torch.load(ROOT/config['initial_checkpoint'],map_location='cpu',weights_only=True)
old.load_state_dict(saved['model_state'])
missing,unexpected=new.load_state_dict(saved['model_state'],strict=False)
assert missing == ['height_projection.weight','height_projection.bias'] and not unexpected
# Double precision distinguishes a wiring error from float32 reduction order.
torch.manual_seed(42)
old.double()
new.double()
X=torch.rand(2,3,64,192,dtype=torch.float64)
with torch.no_grad():
    baseline=old(X)
    equivalent=new(X)
torch.testing.assert_close(baseline,equivalent,atol=1e-9,rtol=1e-9)
new.alignment=.5
with torch.no_grad():
    logits,attention=new(X,return_attention=True)
assert logits.shape==(2,7,67) and attention.shape==(2,7,48)
assert torch.isfinite(logits).all()
refined_config=json.loads((ROOT/'task3/config/refined.json').read_text())
aligned_saved=torch.load(ROOT/refined_config['initial_checkpoint'],map_location='cpu',weights_only=True)
reference=AlignedPlateSeq2Seq(67,aligned_saved['config'])
reference.load_state_dict(aligned_saved['model_state'])
refined=RefinedPlateSeq2Seq(67,refined_config)
missing,unexpected=refined.load_state_dict(aligned_saved['model_state'],strict=False)
assert set(missing)=={'column_positions','refinement.0.weight','refinement.0.bias','refinement.2.weight','refinement.2.bias'} and not unexpected
reference.double();refined.double()
with torch.no_grad():
    torch.testing.assert_close(reference(X),refined(X),atol=1e-9,rtol=1e-9)
assert sum(p.numel() for p in refined.parameters())==642051
assert not torch.cuda.is_initialized()
print('Layout/import/model shape checks PASS; refined starts identically to its aligned warm start; CUDA not initialized.')

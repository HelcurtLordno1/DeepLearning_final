"""Freeze validation-selected checkpoints before either test score is inspected."""
import bootstrap
import argparse
from datetime import datetime
import json
from bootstrap import ROOT
from utils import save_json, file_hash
parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
args=parser.parse_args()
folder=(ROOT/args.run).resolve()
config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
assert metadata['validation_evaluated'] and not metadata['test_evaluated']
assert metadata['last_completed_epoch']>=4
assert not (folder/'test_metrics.json').exists()
baseline=ROOT/'run/task3/cnn_gru_attention/42/20261116_203037_103880'
assert metadata['data_id']==json.loads((baseline/'metadata.json').read_text(encoding='utf-8'))['data_id']
entries=[]
intermediate=ROOT/'run/task3/aligned_cnn_gru/42/20261116_235328_983621'
refined=ROOT/'run/task3/refined_spatial_cnn_gru/42/20261117_003013_172780'
candidates=[(baseline,'baseline'),(intermediate,'aligned_intermediate')]
if refined.resolve()!=folder:
    candidates.append((refined,'refined_intermediate'))
candidates.append((folder,'selected_model'))
for path,role in candidates:
    if any((ROOT/item['run']).resolve()==path.resolve() for item in entries):
        continue
    entries.append({'run':str(path.relative_to(ROOT)),'role':role,
                    'checkpoint_sha256':file_hash(path/'best.pt'),
                    'config_sha256':file_hash(path/'config.json')})
selection={'selected_run':str(folder.relative_to(ROOT)), 'selected_on':'validation only',
           'frozen_at':datetime.now().isoformat(), 'checkpoints':entries,
           'validation_plate_accuracy':metadata['validation_score'],
           'validation_CER':metadata['validation_CER'],
           'last_completed_epoch':metadata['last_completed_epoch'], 'best_epoch':metadata['best_epoch'],
           'final_seeds':[config['seed']], 'initialization':'shared training-only baseline checkpoint; fine-tuned model, not from-scratch independent runs'}
lock=ROOT/'run/task3/final_selection.json'
assert not lock.exists(), 'Preserve the existing locked selection; do not reselect using test.'
save_json(lock,selection)
save_json(ROOT/'run/task3/locked_test_runs.json',[entry['run'] for entry in entries])
save_json(ROOT/'task3/selected_config.json',config)
choices=json.loads((ROOT/'run/selected_configs.json').read_text(encoding='utf-8'))
choices['task3']={'config':config,'seed42_run':str(folder.relative_to(ROOT)),
                  'selection_metric':'plate_accuracy_then_CER','selection_uses':'validation only',
                  'final_seeds':[config['seed']]}
save_json(ROOT/'run/selected_configs.json',choices)
metadata.update({'training_complete':True,'stop_reason':'frozen_after_validation_review',
                 'frozen_for_test':True,'frozen_at':selection['frozen_at']})
save_json(folder/'metadata.json',metadata)
print('Frozen baseline and selected model before test. Final fine-tuning seeds:',selection['final_seeds'])

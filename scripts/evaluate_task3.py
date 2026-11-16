"""Evaluate only checkpoint hashes already frozen before viewing test scores."""
import bootstrap
import argparse
import json
from bootstrap import ROOT
from utils import file_hash
from task3.code.task3 import evaluate_run
parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
args=parser.parse_args()
folder=(ROOT/args.run).resolve()
selection=json.loads((ROOT/'run/task3/final_selection.json').read_text(encoding='utf-8'))
entry=next(item for item in selection['checkpoints'] if (ROOT/item['run']).resolve()==folder)
assert file_hash(folder/'best.pt')==entry['checkpoint_sha256']
assert file_hash(folder/'config.json')==entry['config_sha256']
if (folder/'test_metrics.json').exists():
    print('Locked checkpoint already evaluated; keeping its original test artifacts.')
else:
    evaluate_run(folder,'test')

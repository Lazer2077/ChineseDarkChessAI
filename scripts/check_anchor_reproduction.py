"""Require a full selected-checkpoint reproduction before a weighted ablation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--expected',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    folder=ROOT/a.run;out=ROOT/a.out
    if out.exists():raise ValueError('Refuse to overwrite reproduction check')
    state=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text())
    if state['status']!='completed' or state['entropy_weight_mode']!='constant-unit':raise ValueError('Require completed constant control')
    if sha(folder/'best.npz')!=a.expected or best['npz_sha256']!=a.expected:
        raise ValueError('Constant entropy-wrapper control differs from original unit anchor; stop ablation')
    save_json(out,{'schema':1,'status':'completed','scope':'Same-seed technical control reproduction, not new strength evidence.',
        'model_sha256':a.expected,'epochs':state['epochs_completed'],'updates':state['updates'],
        'all_selected_arrays_reproduced_bitwise':True})


if __name__=='__main__':main()

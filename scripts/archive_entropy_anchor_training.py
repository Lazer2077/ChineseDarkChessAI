"""Extend the existing completed-training archive with required weight metadata."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import archive_anchored_training as original
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--training-plan',required=True);a=p.parse_args()
    folder=ROOT/a.run;path=ROOT/a.training_plan;plan=json.loads(path.read_text())
    state=json.loads((folder/'manifest.json').read_text())
    if state['status']!='completed' or state['plan_sha256']!=sha(path):raise ValueError('Wrong completed training recipe')
    mode=state['entropy_weight_mode']
    if mode!=plan['entropy_weight_mode']:raise ValueError('Weight mode mismatch')
    weight_sha=None
    if mode=='mean-matched-observer-entropy':
        file=Path(plan['uncertainty_data'])/'diagnostic.npz';weight_sha=sha(ROOT/file)
        if plan['sha256'].get(str(file))!=weight_sha:raise ValueError('Observable weights not pinned in training plan')
    sys.argv=[sys.argv[0],'--run',a.run,'--out',a.out];original.main()
    out=ROOT/a.out;report=json.loads(out.read_text())
    report.update(entropy_weight_mode=mode,coefficient_mean=state['entropy_weight_mean'],
        coefficient_min=state['entropy_weight_min'],coefficient_max=state['entropy_weight_max'],
        uncertainty_arrays_sha256=weight_sha,
        coefficient_definition='One global TRAIN-corpus normalization; no per-minibatch renormalization. No test/selection inputs used for coefficients.')
    save_json(out,report)


if __name__=='__main__':main()

"""Archive a fixed-final-epoch policy-target intervention, without promotion."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    p.add_argument('--expected-original');a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());folder=ROOT/plan['out'];out=ROOT/a.out
    if out.exists():raise ValueError('Refuse to overwrite training report')
    for name,h in plan['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen training input changed: '+name)
    state=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text())
    if state['status']!='completed' or state['plan_sha256']!=sha(path):raise ValueError('Training not complete/frozen')
    if state['checkpoint_selection']!='fixed-final-epoch' or best['epoch']!=plan['epochs']:
        raise ValueError('Not the prespecified final epoch')
    if state['policy_target']!=plan['policy_target']:raise ValueError('Wrong policy target')
    n=sum(r['positions'] for r in state['training']);v=sum(r['positions'] for r in state['selection_validation'])
    if state['updates']!=math.ceil(n/plan['batch'])*plan['epochs']:raise ValueError('Unexpected update count')
    model_sha=sha(folder/'best.npz')
    if best['npz_sha256']!=model_sha or state['best_npz_sha256']!=model_sha:raise ValueError('Selected model digest differs')
    if a.expected_original and (plan['policy_target']!='original-search-visits' or model_sha!=a.expected_original):
        raise ValueError('Original-policy control does not reproduce the reference; stop intervention')
    save_json(out,{'schema':1,'status':'completed','scope':
        'Fixed-final-epoch policy-target comparison; same observer input and learner terminal-return values. Profile-specific losses are descriptive, not comparable chess-strength estimates. No native terminal equivalence, privileged actor, DAgger convergence or promotion claim.',
        'research_stage':plan['research_stage'],'training_plan_sha256':sha(path),'policy_target':state['policy_target'],
        'checkpoint_selection':state['checkpoint_selection'],'initial_model_sha256':state['initial_network_sha256'],
        'model_sha256':model_sha,'training_games':len(state['training']),'training_positions':n,
        'selection_games':len(state['selection_validation']),'selection_positions':v,
        'epochs':state['epochs_completed'],'updates':state['updates'],'selected_epoch':best['epoch'],
        'initial_validation':json.loads((folder/'initial-validation.json').read_text()),
        'training_curve':json.loads((folder/'training-log.json').read_text()),
        'original_control_reproduced_bitwise':bool(a.expected_original),
        'value_target':'Unchanged source learner self-play terminal return; no expert cp/mate input.'})
    print(json.dumps({'status':'archived','policy_target':state['policy_target'],'model_sha256':model_sha,'updates':state['updates']}),flush=True)


if __name__=='__main__':main()

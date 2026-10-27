"""Archive a fixed-duration supervision factorial with explicit value semantics."""
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
    p.add_argument('--expected-terminal-control');a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());folder=ROOT/plan['out'];out=ROOT/a.out
    if out.exists():raise ValueError('Refuse to overwrite factorial training report')
    for name,h in plan['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen factorial input changed: '+name)
    state=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text())
    if state['status']!='completed' or state['plan_sha256']!=sha(path):raise ValueError('Training not complete/frozen')
    if state['checkpoint_selection']!='fixed-final-epoch' or best['epoch']!=plan['epochs']:raise ValueError('Not prespecified final epoch')
    for key in ['policy_target','value_target','value_score_scale']:
        if state[key]!=plan[key]:raise ValueError('Wrong factorial target: '+key)
    n=sum(r['positions'] for r in state['training']);v=sum(r['positions'] for r in state['selection_validation'])
    if state['updates']!=math.ceil(n/plan['batch'])*plan['epochs']:raise ValueError('Unexpected update count')
    model_sha=sha(folder/'best.npz')
    if best['npz_sha256']!=model_sha or state['best_npz_sha256']!=model_sha:raise ValueError('Model digest differs')
    if a.expected_terminal_control and (plan['value_target']!='original-learner-terminal' or model_sha!=a.expected_terminal_control):
        raise ValueError('Unchanged-terminal control failed exact reproduction; stop intervention')
    save_json(out,{'schema':1,'status':'completed','scope':
        'Matched policy/value target factorial, fixed final epoch and one training seed. Expert scores are uncalibrated observer-only search proxies, not true returns or an unbiased asymmetric critic. Losses across target definitions are not chess-strength evidence. No new native match, promotion or final70%claim.',
        'research_stage':plan['research_stage'],'training_plan_sha256':sha(path),
        'policy_target':state['policy_target'],'value_target':state['value_target'],'value_score_scale':state['value_score_scale'],
        'value_definition':'Original learner terminal return' if state['value_target']=='original-learner-terminal' else 'Observer-expert cp -> tanh(cp/860), nonzero mate -> sign; fixed proxy only.',
        'checkpoint_selection':state['checkpoint_selection'],'initial_model_sha256':state['initial_network_sha256'],
        'model_sha256':model_sha,'training_games':len(state['training']),'training_positions':n,
        'selection_games':len(state['selection_validation']),'selection_positions':v,
        'epochs':state['epochs_completed'],'updates':state['updates'],'selected_epoch':best['epoch'],
        'initial_validation':json.loads((folder/'initial-validation.json').read_text()),
        'training_curve':json.loads((folder/'training-log.json').read_text()),
        'terminal_control_reproduced_bitwise':bool(a.expected_terminal_control)})
    print(json.dumps({'status':'archived','policy_target':state['policy_target'],'value_target':state['value_target'],'model_sha256':model_sha,'updates':state['updates']}),flush=True)


if __name__=='__main__':main()

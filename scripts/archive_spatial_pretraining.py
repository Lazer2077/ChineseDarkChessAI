"""Archive frozen supervised decoder comparisons; no strength inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json
from research.spatial_policy_value import POLICY_MAPPING_SHA256


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());folder=ROOT/a.run;out=ROOT/a.out
    if out.exists():raise ValueError('Refuse to overwrite decoder training report')
    for name,h in plan['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen decoder input changed: '+name)
    state=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text());curve=json.loads((folder/'training-log.json').read_text())
    if state['status']!='completed' or state['plan_sha256']!=sha(path):raise ValueError('Decoder training not completed/frozen')
    if state['policy_head']!=plan['policy_head'] or best['policy_head']!=plan['policy_head']:raise ValueError('Wrong policy decoder')
    if state['policy_mapping_sha256']!=POLICY_MAPPING_SHA256 or best['policy_mapping_sha256']!=POLICY_MAPPING_SHA256:raise ValueError('Wrong decoder geometry')
    if state['epochs_completed']!=plan['training']['epochs'] or curve[-1]['updates']!=state['updates_planned']:raise ValueError('Incomplete optimization budget')
    model_sha=sha(folder/'best.npz')
    if state['best_npz_sha256']!=model_sha or best['npz_sha256']!=model_sha:raise ValueError('Selected decoder artifact differs')
    save_json(out,{'schema':1,'status':'completed','scope':plan['scope'],'training_plan_sha256':sha(path),
        'research_stage':plan['research_stage'],'policy_head':state['policy_head'],'policy_mapping_sha256':POLICY_MAPPING_SHA256,
        'model_sha256':model_sha,'initial_sha256':state['initial_sha256'],'parameters':state['parameters'],
        'channels':state['channels'],'blocks':state['blocks'],'training':state['training'],'selection_validation':state['selection_validation'],
        'epochs':state['epochs_completed'],'updates':state['updates_planned'],'selected_epoch':best['epoch'],
        'initial_validation':json.loads((folder/'initial-validation.json').read_text()),'training_curve':curve,
        'checkpoint_selection':'Minimum common selection policy CE + original0.7score/0.3outcome MSE; no match-based choice.',
        'limitations':'Supervised search-action imitation, not current-model self-play RL, an information-state sufficiency proof or a playing-strength/promotion result.'})
    print(json.dumps({'status':'archived','head':state['policy_head'],'model_sha256':model_sha,'updates':state['updates_planned']}),flush=True)


if __name__=='__main__':main()

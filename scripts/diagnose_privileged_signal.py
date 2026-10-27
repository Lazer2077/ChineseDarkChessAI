"""Post-hoc training-only feature-use diagnostic, with frozen selected critics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research.neural_replay import sha,save_json
from research.neural_search import load_model
from research.policy_value import load_npz
from research.privileged_value import PrivilegedValueNet
from train_privileged_value import load_games,prediction


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    path=Path(args.plan);plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Signal plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Signal input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corpus=json.loads((ROOT/plan['corpus_plan']).read_text())
    manifest=json.loads((ROOT/corpus['out']/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Incomplete corpus')
    entries={x['index']:x for x in manifest['episodes']}
    raw,truth,z,groups,meta=load_games(entries,corpus['splits']['training'])
    shuffled=truth.copy();board=raw[:,:90].reshape(-1,10,9)[:,::-1].reshape(-1,90)
    colors=(board>>4)^1;rng=np.random.default_rng(plan['permutation_seed'])
    for i in range(len(truth)):
        for color in (0,1):
            cells=np.flatnonzero((truth[i]>0)&(colors[i]==color))
            shuffled[i,cells]=rng.permutation(truth[i,cells])
    dev=torch.device(plan['device']);warm,_=load_model(ROOT/plan['initial_model'],dev)
    net=PrivilegedValueNet(warm).to(dev);load_npz(net,ROOT/plan['critic']/'best.npz')
    actual=prediction(net,raw,truth,True,dev,plan['batch'])
    zero=prediction(net,raw,truth,False,dev,plan['batch'])
    permuted=prediction(net,raw,shuffled,True,dev,plan['batch'])
    full=~(truth>0).any(1);delta=actual.astype(np.float64)-zero
    extra=net.network.conv.weight[:,238:].detach().float().cpu().numpy()
    base=net.network.conv.weight[:,:238].detach().float().cpu().numpy()
    report={'schema':1,'status':'completed','scope':plan['scope'],'plan_sha256':digest,
        'training_games':len(meta),'training_positions':len(raw),'fully_public_positions':int(full.sum()),
        'selected_critic_sha256':sha(ROOT/plan['critic']/'best.npz'),
        'extra_input_weight_l2':float(np.linalg.norm(extra)),
        'original_input_weight_l2':float(np.linalg.norm(base)),
        'mean_abs_value_change_truth_vs_zero':float(np.abs(delta).mean()),
        'max_abs_value_change_truth_vs_zero':float(np.abs(delta).max()),
        'public_max_abs_change_truth_vs_zero':float(np.abs(delta[full]).max()) if full.any() else None,
        'training_position_mse':{k:float(((v.astype(np.float64)-z)**2).mean())
            for k,v in [('true_labels',actual),('zero_labels',zero),('within_color_permuted_labels',permuted)]},
        'permutation_seed':plan['permutation_seed'],
        'permutation_changes_positions':int((truth!=shuffled).any(1).sum()),
        'training_episode_hashes':meta}
    np.savez_compressed(out/'predictions.npz',actual=actual,zero=zero,permuted=permuted,returns=z,groups=groups)
    report['prediction_arrays_sha256']=sha(out/'predictions.npz');verify();save_json(out/'report.json',report)


if __name__=='__main__':main()

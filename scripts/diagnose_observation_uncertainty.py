"""Training-only finite-inventory uncertainty and search-target entropy study."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research.neural_replay import sha,save_json
from research.observation_uncertainty import record_uncertainty,INITIAL_TWO_SIDE_ENTROPY
from train_neural_selfplay import load_split


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    path=Path(args.plan);plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Uncertainty plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Uncertainty input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    folder=ROOT/plan['episodes'];manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Incomplete fixed generation')
    raw,pi,_,meta=load_split(folder,plan['indices'],manifest)
    uncertainty=record_uncertainty(raw)
    masks=ROOT/plan['observer_masks'];mask_report=json.loads((masks/'report.json').read_text())
    if sha(masks/'masks.npz')!=mask_report['mask_arrays_sha256']:raise ValueError('Masks changed')
    with np.load(masks/'masks.npz',allow_pickle=False) as a:
        if list(a['episode_indices'])!=plan['indices'] or len(a['mask_bits'])!=len(raw):raise ValueError('Mask alignment differs')
        legal=np.unpackbits(a['mask_bits'],axis=1,count=2550,bitorder='little').sum(1)
    target_entropy=-(pi*np.log(np.maximum(pi,np.finfo(np.float64).tiny))).sum(1)
    relative=np.divide(target_entropy,np.log(np.maximum(legal,1)),out=np.zeros(len(raw)),where=legal>1)
    if (relative>1+1e-6).any():raise ValueError('Target entropy exceeds legal-set maximum')
    normalized=uncertainty['normalized_entropy'];mean=float(normalized.mean())
    if mean<=0:raise ValueError('No partially observed training coverage')
    weights=normalized/mean
    report={'schema':1,'status':'completed','scope':plan['scope'],'plan_sha256':digest,
        'games':len(meta),'positions':len(raw),'initial_two_side_entropy_nats':INITIAL_TWO_SIDE_ENTROPY,
        'mean_normalized_entropy':mean,'fully_public_positions':int(uncertainty['fully_public'].sum()),
        'type_certain_but_not_public_positions':int((uncertainty['type_certain']&~uncertainty['fully_public']).sum()),
        'mean_matched_candidate_weight_range':[float(weights.min()),float(weights.max())],
        'candidate_mean_weight':float(weights.mean()),
        'entropy_target_entropy_correlation':float(np.corrcoef(normalized,target_entropy)[0,1]),
        'entropy_relative_target_entropy_correlation':float(np.corrcoef(normalized,relative)[0,1]),
        'bins':[],'episodes':meta}
    for label,mask in [('type-certain',uncertainty['type_certain'])]+[
        (name,(normalized>lower)&(normalized<=upper+1e-12))
        for name,lower,upper in [('0-.25',0.,.25),('.25-.5',.25,.5),('.5-.75',.5,.75),('.75-1',.75,1.)]]:
        row={'bin':label,'positions':int(mask.sum())}
        if mask.any():row.update(mean_live_dark_count=float(uncertainty['live_dark_count'][mask].mean()),
            mean_normalized_entropy=float(normalized[mask].mean()),mean_legal_actions=float(legal[mask].mean()),
            mean_search_target_entropy=float(target_entropy[mask].mean()),mean_relative_search_target_entropy=float(relative[mask].mean()))
        report['bins'].append(row)
    np.savez_compressed(out/'diagnostic.npz',**uncertainty,search_target_entropy=target_entropy,
        legal_action_count=legal,relative_search_target_entropy=relative,candidate_mean_matched_weight=weights)
    report['arrays_sha256']=sha(out/'diagnostic.npz');verify();save_json(out/'report.json',report)


if __name__=='__main__':main()

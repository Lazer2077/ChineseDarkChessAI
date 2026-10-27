"""Frozen observer-only expert-replay retention; no fit or new test-strength claim."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research.neural_replay import sha,save_json
from research.neural_search import load_model
from research.policy_value import REC,encode_records
from train_policy_value import load_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    path=Path(args.plan);plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Retention plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Retention input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    raw,actions,meta=load_data([ROOT/plan['dataset']]);records=raw.reshape(-1,112).view(REC).reshape(-1)
    static=np.tanh(records['score'].astype(np.float64)/860)
    terminal=records['result'].astype(np.float64);mixed=.7*static+.3*terminal
    full=((raw[:,:90]&15)!=8).all(1)
    report={'schema':1,'status':'running','scope':plan['scope'],'plan_sha256':digest,
            'dataset':meta,'fully_public_positions':int(full.sum()),'models':[]}
    reference=None;save_json(out/'report.json',report)
    for model in plan['models']:
        net,best=load_model(ROOT/model['directory'],'cpu');logits=[];values=[]
        with torch.inference_mode():
            for start in range(0,len(raw),plan['batch']):
                p,v=net(encode_records(torch.from_numpy(raw[start:start+plan['batch']])))
                logits.append(F.log_softmax(p.float(),dim=1).cpu().numpy());values.append(v.float().cpu().numpy())
        logs=np.concatenate(logits);value=np.concatenate(values).astype(np.float64)
        if reference is None:reference=logs.copy()
        ce=-logs[np.arange(len(raw)),actions].astype(np.float64)
        kl=(np.exp(reference.astype(np.float64))*(reference.astype(np.float64)-logs)).sum(1)
        if kl.min() < -1e-5:raise ValueError('Invalid policy KL')
        row={'label':model['label'],'model_sha256':best['npz_sha256'],'strata':{}}
        for label,mask in [('all',np.ones(len(raw),dtype=bool)),('partial',~full),('public',full)]:
            ids=np.flatnonzero(mask);subset=logs[mask]
            row['strata'][label]={'positions':int(mask.sum()),'teacher_action_ce':float(ce[mask].mean()),
                'teacher_action_top1':float((subset.argmax(1)==actions[mask]).mean()),
                'teacher_action_top5':float((np.argpartition(subset,-5,axis=1)[:,-5:]==actions[mask,None]).any(1).mean()),
                'warm_to_model_policy_kl_nats':float(kl[mask].mean()),
                'mixed_value_mse':float(((value[mask]-mixed[mask])**2).mean()),
                'terminal_value_mse':float(((value[mask]-terminal[mask])**2).mean()),
                'teacher_static_value_mse':float(((value[mask]-static[mask])**2).mean())}
        report['models'].append(row);verify();save_json(out/'report.json',report);print(json.dumps(row),flush=True)
        del net,logs,logits,values
    verify();report['status']='completed';save_json(out/'report.json',report)


if __name__=='__main__':main()

"""Four fixed weight/BN-buffer combinations on existing selection trajectories."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import torch.nn.functional as F
from research.policy_value import encode_records,export_npz,CONTRACT
from research.neural_search import load_model
from research.neural_replay import sha,save_json
from research.normalization_probe import combine
from train_neural_selfplay import load_split


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite a diagnostic')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Probe plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen input changed: '+name)
    verify();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    training=json.loads((ROOT/plan['training_plan']).read_text())
    generation=json.loads((ROOT/training['generation_plan']).read_text());folder=ROOT/generation['out']
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Require completed trajectories')
    raw,pi,z,episodes=load_split(folder,training['selection_validation_games'],manifest)
    data=torch.from_numpy(raw);target_pi=torch.from_numpy(pi);target_z=torch.from_numpy(z)
    nets={};metas={}
    for name,directory in plan['models'].items():nets[name],metas[name]=load_model(ROOT/directory,'cpu')
    out.mkdir(parents=True)
    report={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,
        'scope':'Inference-only weight/BN-buffer factorial on existing checkpoint-selection trajectories. No optimizer update or fresh strength test. RL buffers were estimated jointly with changing weights, so transplantation is not fixed-weight BN-only retraining.',
        'positions':len(raw),'episodes':episodes,'results':[]}
    save_json(out/'report.json',report);reference=None
    try:
        for arm in plan['arms']:
            net,keys=combine(nets[arm['weights']],nets[arm['statistics']]);arm_out=out/arm['id'];arm_out.mkdir()
            export_npz(net,arm_out/'best.npz')
            m=metas[arm['weights']]
            save_json(arm_out/'best.json',{'input_contract':CONTRACT,'channels':m['channels'],'blocks':m['blocks'],
                'phase_experts':m['phase_experts'],'npz_sha256':sha(arm_out/'best.npz'),
                'construction':'Inference-only BN-buffer intervention,no optimizer updates or checkpoint selection',
                'weights_from':plan['models'][arm['weights']],'running_buffers_from':plan['models'][arm['statistics']]})
            started=time.perf_counter();policies=[];values=[]
            with torch.inference_mode():
                for start in range(0,len(data),128):
                    p,v=net(encode_records(data[start:start+128]));policies.append(F.log_softmax(p.float(),1));values.append(v.float())
            logp=torch.cat(policies);value=torch.cat(values)
            if reference is None:reference=(logp.clone(),value.clone())
            ce=-(target_pi*logp).sum(1);mse=(target_z-value)**2
            kl=(reference[0].exp()*(reference[0]-logp)).sum(1)
            row={'id':arm['id'],'weights':arm['weights'],'statistics':arm['statistics'],
                'search_policy_ce':float(ce.mean()),'terminal_value_mse':float(mse.mean()),
                'warm_policy_kl_to_arm':float(kl.mean()),
                'value_rmse_difference_from_warm':float(((reference[1]-value)**2).mean().sqrt()),
                'running_buffers':len(keys),'all_state_sources_verified':True,
                'npz_sha256':sha(arm_out/'best.npz'),'seconds':time.perf_counter()-started}
            if arm['id']=='trained-control':
                expected=metas['trained']['validation']
                if abs(row['search_policy_ce']-expected['search_policy_ce'])>1e-4 or abs(row['terminal_value_mse']-expected['terminal_value_mse'])>1e-4:
                    raise ValueError('CPU control does not reproduce archived trained validation')
            report['results'].append(row);save_json(out/'report.json',report);print(json.dumps(row),flush=True)
        verify();report['status']='completed';save_json(out/'report.json',report)
    except BaseException as error:
        report.update(status='failed',error=str(error));save_json(out/'report.json',report);raise


if __name__=='__main__':main()

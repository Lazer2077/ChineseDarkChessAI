"""One fixed search-policy/terminal-value update, with full episode replay audits.

This is a development RL iteration, not a model promotion or strength claim.
"""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from research.policy_value import CONTRACT,encode_records,MIRROR_ACTION,export_npz
from research.neural_search import load_model
from research.neural_replay import load_episode,save_json,sha

def load_split(folder,indices,manifest):
    records=[];policies=[];values=[];metadata=[]
    entries={row['index']:row for row in manifest['episodes']}
    for index in indices:
        data,meta=load_episode(folder/f'game-{index:06d}.npz')
        expected=dict(entries[index]);expected.pop('index')
        if meta!=expected:raise ValueError('Episode does not match completed generation manifest')
        records.append(data['records']);policies.append(data['visit_counts'].astype(np.float32)/data['root_visits'][:,None])
        values.append(data['values']);metadata.append({'index':index,'seed':meta['seed'],'positions':meta['positions'],'sha256':meta['npz_sha256']})
    return np.concatenate(records),np.concatenate(policies),np.concatenate(values),metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Use fresh training output')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Training plan changed')
        for file,expected in plan['sha256'].items():
            if sha(ROOT/file)!=expected:raise ValueError('Frozen RL input changed: '+file)
    verify();generation=json.loads((ROOT/plan['generation_plan']).read_text());folder=ROOT/generation['out']
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['completed']!=generation['games'] or manifest['plan_sha256']!=sha(ROOT/plan['generation_plan']):raise ValueError('Require all fixed self-play episodes completed')
    train_ids=plan['training_games'];val_ids=plan['selection_validation_games']
    if len(set(train_ids+val_ids))!=len(train_ids)+len(val_ids) or set(train_ids+val_ids)!=set(range(generation['games'])):
        raise ValueError('Duplicate, missing or overlapping game split')
    raw,pi,z,train_meta=load_split(folder,train_ids,manifest)
    vraw,vpi,vz,val_meta=load_split(folder,val_ids,manifest)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(plan['seed'])
    dev=torch.device(plan['device']);net,initial=load_model(ROOT/plan['initial_model'],dev)
    if generation['model']!=plan['initial_model']:raise ValueError('This iteration must update its generating actor')
    optimizer=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001)
    data=torch.from_numpy(raw).to(dev);target_pi=torch.from_numpy(pi).to(dev);target_z=torch.from_numpy(z).to(dev)
    validation=torch.from_numpy(vraw).to(dev);val_pi=torch.from_numpy(vpi).to(dev);val_z=torch.from_numpy(vz).to(dev)
    rng=torch.Generator(device=dev).manual_seed(plan['seed']);mirror_actions=torch.as_tensor(MIRROR_ACTION,device=dev)
    batch=plan['batch'];out.mkdir(parents=True)
    state={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'input_contract':CONTRACT,
        'initial_network_sha256':initial['npz_sha256'],'generation_plan_sha256':sha(ROOT/plan['generation_plan']),
        'training':train_meta,'selection_validation':val_meta,'epochs_completed':0,'updates':0,
        'scope':'Search-guided neural self-play RL update using actual root visits and terminal values. Tiny development pilot; no teacher-strength gain, equilibrium guarantee or model promotion.'}
    save_json(out/'manifest.json',state)
    def evaluate():
        net.eval();totals=np.zeros(4)
        with torch.inference_mode():
            for start in range(0,len(validation),batch):
                logits,value=net(encode_records(validation[start:start+batch]))
                ce=-(val_pi[start:start+batch]*F.log_softmax(logits.float(),dim=1)).sum(1)
                mse=(value.float()-val_z[start:start+batch])**2
                full=(validation[start:start+batch,:90].bitwise_and(15)==8).sum(1)==0
                totals+=np.array([float(ce.sum()),float(mse.sum()),len(ce),float(mse[full].sum())])
        return {'loss':float((totals[0]+totals[1])/totals[2]),'search_policy_ce':float(totals[0]/totals[2]),'terminal_value_mse':float(totals[1]/totals[2])}
    try:
        initial_metrics=evaluate();save_json(out/'initial-validation.json',initial_metrics)
        log=[];best=float('inf');started=time.monotonic()
        for epoch in range(plan['epochs']):
            net.train();order=torch.randperm(len(data),device=dev,generator=rng);sums=np.zeros(3)
            for start in range(0,len(data),batch):
                ids=order[start:start+batch];x=encode_records(data[ids]);p=target_pi[ids];v=target_z[ids]
                flip=torch.rand(len(ids),device=dev,generator=rng)<.5
                x=torch.where(flip[:,None,None,None],x.flip(-1),x)
                p=torch.where(flip[:,None],p[:,mirror_actions],p)
                logits,value=net(x);ce=-(p*F.log_softmax(logits.float(),dim=1)).sum(1).mean();mse=F.mse_loss(value.float(),v)
                loss=ce+mse
                if not torch.isfinite(loss):raise ValueError('Nonfinite RL loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite RL gradient')
                optimizer.step();state['updates']+=1;sums+=np.array([float(ce.detach())*len(ids),float(mse.detach())*len(ids),len(ids)])
            metrics=evaluate();state['epochs_completed']=epoch+1
            row={'epoch':epoch+1,'updates':state['updates'],'train_policy_ce':float(sums[0]/sums[2]),'train_terminal_mse':float(sums[1]/sums[2]),'validation':metrics,'seconds':time.monotonic()-started}
            log.append(row);save_json(out/'training-log.json',log);print(json.dumps(row),flush=True)
            if metrics['loss']<best:
                best=metrics['loss'];export_npz(net,out/'best.npz.tmp');(out/'best.npz.tmp').replace(out/'best.npz')
                save_json(out/'best.json',{'epoch':epoch+1,'validation':metrics,'npz_sha256':sha(out/'best.npz'),'input_contract':CONTRACT,
                    'phase_experts':initial['phase_experts'],'channels':initial['channels'],'blocks':initial['blocks']})
            save_json(out/'manifest.json',state)
        verify();state.update(status='completed',best_npz_sha256=sha(out/'best.npz'));save_json(out/'manifest.json',state)
    except BaseException as error:
        state.update(status='failed',error=str(error));save_json(out/'manifest.json',state);raise

if __name__=='__main__':main()

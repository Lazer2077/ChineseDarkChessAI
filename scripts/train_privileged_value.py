"""Paired supervised critic diagnostic; test games first loaded after selection."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from research.neural_replay import sha, save_json
from research.neural_search import load_model
from research.policy_value import export_npz, load_npz
from research.privileged_value import CONTRACT, PrivilegedValueNet, encode_value_inputs
from privileged_data_preflight import replay


def load_games(entries, ids):
    raw = []; truth = []; z = []; groups = []; meta = []
    for index in ids:
        row = entries[index]; path = ROOT/row['file']
        if sha(path) != row['npz_sha256']: raise ValueError('Corpus array changed')
        with np.load(path, allow_pickle=False) as a: data = {k: a[k] for k in a.files}
        if replay(data, row['seed'], row['outcome'], row['reason']) != row['audit']:
            raise ValueError('Corpus independent replay differs')
        raw.append(data['records']); truth.append(data['dark_truth']); z.append(data['values'])
        groups.extend([index]*len(data['records']))
        meta.append({'index': index, 'seed': row['seed'], 'sha256': row['npz_sha256'],
                     'positions': len(data['records'])})
    return np.concatenate(raw), np.concatenate(truth), np.concatenate(z), np.asarray(groups), meta


def prediction(net, raw, truth, privileged, device, batch):
    net.eval(); pieces = []
    with torch.inference_mode():
        for start in range(0,len(raw),batch):
            x = encode_value_inputs(torch.from_numpy(raw[start:start+batch]).to(device),
                torch.from_numpy(truth[start:start+batch]).to(device), privileged)
            pieces.append(net(x).float().cpu().numpy())
    return np.concatenate(pieces)


def metrics(pred, target, groups):
    errors = (pred.astype(np.float64)-target)**2
    per_game = [float(errors[groups==i].mean()) for i in np.unique(groups)]
    return {'position_weighted_mse': float(errors.mean()), 'game_weighted_mse': float(np.mean(per_game)),
            'game_mse': per_game}


def summarize_test(predictions, z, groups, truth, seed, bootstrap):
    errors = {k:(p.astype(np.float64)-z)**2 for k,p in predictions.items()}
    ids = np.unique(groups)
    gains = np.asarray([float((errors['symmetric'][groups==i]-errors['privileged'][groups==i]).mean()) for i in ids])
    rng = np.random.default_rng(seed)
    means = gains[rng.integers(len(ids),size=(bootstrap,len(ids)))].mean(1)
    lo,hi = np.quantile(means,[.025,.975])
    report = {'models': {k:metrics(p,z,groups) for k,p in predictions.items()},
        'game_weighted_mse_reduction_symmetric_minus_privileged': float(gains.mean()),
        'paired_game_bootstrap_95': [float(lo),float(hi)],
        'privilege_predictive_benefit_supported': bool(lo>0 and len(ids)>=64),
        'minimum_test_games_for_positive_diagnostic': 64, 'bootstrap_replicates': bootstrap,
        'phases': [], 'calibration': {}}
    dark = (truth>0).sum(1)
    for name,lower,upper in [('public',0,0),('dark1-6',1,6),('dark7-15',7,15),('dark16-30',16,30)]:
        mask = (dark>=lower)&(dark<=upper)
        row = {'phase':name,'positions':int(mask.sum()),'games':len(np.unique(groups[mask]))}
        if mask.any():
            row['metrics'] = {k:metrics(p[mask],z[mask],groups[mask]) for k,p in predictions.items()}
        report['phases'].append(row)
    for name,p in predictions.items():
        rows=[]
        for j in range(10):
            lower,upper = -1+j*.2,-1+(j+1)*.2
            mask = (p>=lower)&((p<=upper) if j==9 else (p<upper))
            if mask.any():rows.append({'lower':lower,'upper':upper,'positions':int(mask.sum()),
                'mean_prediction':float(p[mask].mean()),'mean_terminal_return':float(z[mask].mean())})
        report['calibration'][name]=rows
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',required=True); args=parser.parse_args()
    path=Path(args.plan); plan=json.loads(path.read_text()); digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Value plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen value input changed: '+name)
    verify(); out=ROOT/plan['out']; out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    corpus_plan=json.loads((ROOT/plan['corpus_plan']).read_text())
    manifest=json.loads((ROOT/corpus_plan['out']/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['plan_sha256']!=sha(ROOT/plan['corpus_plan']):
        raise ValueError('Require complete frozen corpus')
    entries={r['index']:r for r in manifest['episodes']}; splits=corpus_plan['splits']
    all_ids=sum((splits[k] for k in ('training','selection','test')),[])
    if len(set(all_ids))!=len(all_ids) or set(all_ids)!=set(entries):raise ValueError('Invalid whole-game split')
    raw,truth,z,_,train_meta=load_games(entries,splits['training'])
    vr,vt,vz,vg,selection_meta=load_games(entries,splits['selection'])
    dev=torch.device(plan['device']); warm,initial=load_model(ROOT/plan['initial_model'],dev)
    state={'schema':1,'status':'running','pid':os.getpid(),'contract':CONTRACT,'plan_sha256':digest,
           'corpus_plan_sha256':manifest['plan_sha256'],'initial_model_sha256':initial['npz_sha256'],
           'training':train_meta,'selection':selection_meta,'profiles':{},'scope':plan['scope']}
    save_json(out/'report.json',state); tensors=[torch.from_numpy(a).to(dev) for a in (raw,truth,z)]
    data,labels,returns=tensors
    for label,privileged in [('symmetric',False),('privileged',True)]:
        verify(); torch.manual_seed(plan['seed']);net=PrivilegedValueNet(warm).to(dev)
        folder=out/label;folder.mkdir()
        optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=.0001)
        generator=torch.Generator(device=dev).manual_seed(plan['seed'])
        best=float('inf');log=[];updates=0;started=time.monotonic()
        for epoch in range(plan['epochs']):
            net.train();order=torch.randperm(len(raw),device=dev,generator=generator);sums=np.zeros(2)
            for start in range(0,len(raw),plan['batch']):
                ids=order[start:start+plan['batch']]
                x=encode_value_inputs(data[ids],labels[ids],privileged)
                flip=torch.rand(len(ids),device=dev,generator=generator)<.5
                x=torch.where(flip[:,None,None,None],x.flip(-1),x)
                pred=net(x);loss=F.mse_loss(pred.float(),returns[ids])
                if not torch.isfinite(loss):raise ValueError('Nonfinite conditional-value loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite conditional-value gradient')
                optimizer.step();updates+=1;sums += [float(loss.detach())*len(ids),len(ids)]
            val=metrics(prediction(net,vr,vt,privileged,dev,plan['batch']),vz,vg)
            row={'epoch':epoch+1,'updates':updates,'train_mse':float(sums[0]/sums[1]),
                 'selection':val,'seconds':time.monotonic()-started};log.append(row)
            if val['game_weighted_mse']<best:
                best=val['game_weighted_mse'];export_npz(net,folder/'best.npz')
                save_json(folder/'best.json',{'contract':CONTRACT,'privileged':privileged,'epoch':epoch+1,
                    'selection':val,'npz_sha256':sha(folder/'best.npz')})
            verify();save_json(folder/'training-log.json',log);print(json.dumps({'profile':label,**row}),flush=True)
        state['profiles'][label]={'selected':json.loads((folder/'best.json').read_text()),'updates':updates,
            'parameters':sum(p.numel() for p in net.parameters()),'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad)}
        save_json(out/'report.json',state);del net,optimizer
    # Both checkpoints are selected before touching final test arrays.
    tr,tt,tz,tg,test_meta=load_games(entries,splits['test']);predictions={}
    for label,privileged in [('symmetric',False),('privileged',True)]:
        net=PrivilegedValueNet(warm).to(dev);folder=out/label
        if sha(folder/'best.npz')!=state['profiles'][label]['selected']['npz_sha256']:raise ValueError('Selected critic changed')
        load_npz(net,folder/'best.npz');predictions[label]=prediction(net,tr,tt,privileged,dev,plan['batch']);del net
    np.savez_compressed(out/'test-predictions.npz',**predictions,returns=tz,groups=tg,dark_counts=(tt>0).sum(1))
    state['test']=dict(summarize_test(predictions,tz,tg,tt,plan['bootstrap_seed'],plan['bootstrap']),games=test_meta)
    state['test_predictions_sha256']=sha(out/'test-predictions.npz');verify();state['status']='completed'
    save_json(out/'report.json',state)


if __name__=='__main__':main()

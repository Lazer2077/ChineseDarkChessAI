"""Supervised policy/value pretraining from certified observer-only expert replay.

This is search-action imitation, not a claim of completed self-play RL or a Nash
solution. Every arm shares data, initialization, augmentation and optimization.
"""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from research.policy_value import REC,AZ,CONTRACT,PhasePolicyValueNet,encode_records,action_indices,MIRROR_ACTION,load_npz,export_npz
from build_temporal_targets import segment_games

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def load_data(paths):
    arrays=[];actions=[];metadata=[]
    for filename in paths:
        path=Path(filename);meta=json.loads(path.with_suffix('.complete.json').read_text())
        if path.suffix!='.bin' or path.stat().st_size%112 or sha(path)!=meta['sha256'] or meta['rejected_games']:
            raise ValueError('Require finalized valid full trajectories')
        rec=np.fromfile(path,dtype=REC);segment_games(rec,meta['recipe']['games'])
        if not np.isin(rec['result'],[-1,0,1]).all():raise ValueError('Invalid outcome')
        if ((rec['board']&15)>8).any() or ((rec['board']>>4)>1).any() or (rec['stm']>1).any():raise ValueError('Invalid board')
        if (rec['urn']<0).any() or (rec['urn']>np.array([2,2,2,2,2,5])).any():raise ValueError('Invalid inventory')
        action=action_indices(rec);moves=rec['move'].astype(np.int64);src=moves&127;dst=(moves>>7)&127;rows=np.arange(len(rec))
        mover=rec['board'][rows,src];victim=rec['board'][rows,dst]
        if (mover==0).any() or ((mover>>4)!=rec['stm']).any() or ((victim>0)&((victim>>4)==rec['stm'])).any():raise ValueError('Invalid recorded action ownership')
        arrays.append(rec.view(np.uint8).reshape(-1,112));actions.append(action)
        metadata.append({'sha256':meta['sha256'],'games':meta['recipe']['games'],'positions':len(rec)})
    if len({x['sha256'] for x in metadata})!=len(metadata):raise ValueError('Duplicate replay input')
    return np.concatenate(arrays),np.concatenate(actions),metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True);p.add_argument('--val',required=True)
    p.add_argument('--initial',required=True);p.add_argument('--out',required=True);p.add_argument('--experts',action='store_true')
    p.add_argument('--channels',type=int,default=64);p.add_argument('--blocks',type=int,default=15)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--plan',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise SystemExit('Refuse to overwrite a training run')
    plan=json.loads(Path(a.plan).read_text())
    plan_digest=sha(a.plan)
    if 'training' in plan:
        cfg=plan['training']
        expected={'channels':cfg['channels'],'blocks':cfg['residual_blocks'],'epochs':cfg['epochs'],'batch':cfg['batch'],'lr':cfg['lr'],'seed':cfg['seed']}
        if any(getattr(a,key)!=value for key,value in expected.items()) or a.data!=plan['data'] or a.val!=plan['selection_validation'] or a.initial!=plan['initial']:
            raise ValueError('CLI configuration differs from the fixed training plan')
        if a.experts not in [arm['phase_experts'] for arm in plan['arms']]:raise ValueError('Unplanned architecture arm')
    def verify():
        if sha(a.plan)!=plan_digest:raise RuntimeError('Experiment plan changed')
        for filename,digest in plan['sha256'].items():
            if sha(ROOT/filename)!=digest:raise RuntimeError('Frozen experiment input changed: '+filename)
    verify();out.mkdir(parents=True);torch.set_num_threads(2)
    torch.manual_seed(a.seed);dev=torch.device('cuda');torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    raw,labels,metadata=load_data(a.data);vraw,vlabels,vmetadata=load_data([a.val])
    if 'training_positions' in plan and (len(raw)!=plan['training_positions'] or sum(row['games'] for row in metadata)!=plan['unique_training_games']):
        raise ValueError('Loaded data volume differs from frozen recipe')
    if {x['sha256'] for x in metadata}&{x['sha256'] for x in vmetadata}:raise ValueError('Training/validation overlap')
    data=torch.from_numpy(raw).to(dev);target_action=torch.from_numpy(labels).to(dev)
    validation=torch.from_numpy(vraw).to(dev);val_action=torch.from_numpy(vlabels).to(dev)
    net=PhasePolicyValueNet(a.channels,a.blocks,a.experts).to(dev);load_npz(net,a.initial,common_initial=True)
    optimizer=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.0001)
    steps=len(data)//a.batch;total=steps*a.epochs;warmup=max(1,int(total*.05))
    if not steps:raise ValueError('Training batch exceeds replay')
    def schedule(step):
        if step<warmup:return (step+1)/warmup
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    generator=torch.Generator(device=dev).manual_seed(a.seed+17)
    mirror_actions=torch.as_tensor(MIRROR_ACTION,device=dev)
    manifest={'schema':1,'status':'running','pid':os.getpid(),'input_contract':CONTRACT,'phase_experts':a.experts,
      'parameters':sum(x.numel() for x in net.parameters()),'channels':a.channels,'blocks':a.blocks,
      'training':metadata,'selection_validation':vmetadata,'epochs':a.epochs,'batch':a.batch,'updates_planned':total,
      'initial_sha256':sha(a.initial),'plan_sha256':sha(a.plan),'precision':str(dtype),'seed':a.seed,
      'optimizer':'AdamW','learning_rate':a.lr,'weight_decay':.0001,'gradient_clip':1.,
      'scope':'Supervised warm start: one-hot teacher selected-action cross entropy plus mixed value MSE. No policy visits are invented, no current-model self-play improvement or strength gain claimed. Validation is checkpoint-selection data. Historical features are omitted consistently in replay and inference; search must retain rules/history.'}
    atomic_json(out/'manifest.json',manifest)
    def value_targets(batch):
        score=batch[:,103].int()+(batch[:,104].int()<<8)
        score=torch.where(score>=32768,score-65536,score).float()
        result=batch[:,105].to(torch.int8).float()
        return .7*torch.tanh(score/860)+.3*result
    def evaluate():
        net.eval();sums=np.zeros(5);full_sums=np.zeros(3)
        with torch.inference_mode():
            for start in range(0,len(validation),a.batch):
                b=validation[start:start+a.batch];y=val_action[start:start+a.batch];x=encode_records(b)
                with torch.autocast('cuda',dtype=dtype):policy,value=net(x)
                policy=policy.float();value=value.float();v=value_targets(b)
                ce=F.cross_entropy(policy,y,reduction='none');mse=(value-v)**2
                sums+=np.array([float(ce.sum()),float(mse.sum()),float((policy.argmax(1)==y).sum()),float((policy.topk(5,dim=1).indices==y[:,None]).any(1).sum()),len(b)])
                full=x[:,14:28].sum((1,2,3))==0
                full_sums+=np.array([float(ce[full].sum()),float(mse[full].sum()),int(full.sum())])
        net.train();n=sums[-1]
        return {'loss':float((sums[0]+sums[1])/n),'policy_ce':float(sums[0]/n),'value_mse':float(sums[1]/n),'teacher_action_top1':float(sums[2]/n),'teacher_action_top5':float(sums[3]/n),'full_public_positions':int(full_sums[2]),'full_public_policy_ce':float(full_sums[0]/max(1,full_sums[2])),'full_public_value_mse':float(full_sums[1]/max(1,full_sums[2]))}
    log=[];best=math.inf;start_time=time.monotonic();initial=evaluate()
    atomic_json(out/'initial-validation.json',initial);print(json.dumps({'initial':initial}),flush=True)
    for epoch in range(a.epochs):
        order=torch.randperm(len(data),device=dev,generator=generator);sum_ce=sum_value=0.
        for step in range(steps):
            ids=order[step*a.batch:(step+1)*a.batch];b=data[ids];y=target_action[ids]
            x=encode_records(b);mirror=torch.rand(len(b),device=dev,generator=generator)<.5
            x=torch.where(mirror[:,None,None,None],x.flip(-1),x);y=torch.where(mirror,mirror_actions[y],y)
            with torch.autocast('cuda',dtype=dtype):policy,value=net(x)
            ce=F.cross_entropy(policy.float(),y);value_loss=F.mse_loss(value.float(),value_targets(b));loss=ce+value_loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient; no skipped update is counted')
            scaler.step(optimizer);scaler.update();scheduler.step()
            sum_ce+=float(ce.detach());sum_value+=float(value_loss.detach())
        metrics=evaluate();row={'epoch':epoch+1,'updates':(epoch+1)*steps,'train_policy_ce':sum_ce/steps,'train_value_mse':sum_value/steps,'validation':metrics,'seconds':time.monotonic()-start_time}
        log.append(row);atomic_json(out/'training-log.json',log);print(json.dumps(row),flush=True)
        checkpoint={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'data_rng':generator.get_state(),'epoch':epoch+1,'plan_sha256':sha(a.plan)}
        temporary=out/'last.pt.tmp';torch.save(checkpoint,temporary);temporary.replace(out/'last.pt')
        if metrics['loss']<best:
            best=metrics['loss'];export_npz(net,out/'best.npz.tmp');(out/'best.npz.tmp').replace(out/'best.npz')
            atomic_json(out/'best.json',{'epoch':epoch+1,'validation':metrics,'npz_sha256':sha(out/'best.npz'),'input_contract':CONTRACT,'phase_experts':a.experts,'channels':a.channels,'blocks':a.blocks})
        manifest['epochs_completed']=epoch+1;atomic_json(out/'manifest.json',manifest)
    verify();manifest['status']='completed';manifest['best_npz_sha256']=sha(out/'best.npz');atomic_json(out/'manifest.json',manifest)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        if '--out' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--out')+1])/'manifest.json'
            if path.exists():
                state=json.loads(path.read_text());state.update(status='failed',error=str(error));atomic_json(path,state)
        raise

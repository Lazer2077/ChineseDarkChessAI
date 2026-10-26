"""Measure the partial-to-publicly-determined transition and dead-inventory sensitivity."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from observability import identity_entropy, INITIAL_ENTROPY, public_counts
from build_temporal_targets import segment_games

p=argparse.ArgumentParser();p.add_argument('--data',action='append',required=True,help='label:path:completed_games');p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi/engine/train/train.py'
spec=importlib.util.spec_from_file_location('observability_trainer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
if Path(a.out).exists():raise SystemExit('Use a new report')
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net=m.Net(state['ft.weight'].shape[1]-1,state['l1.weight'].shape[0],state['l2.weight'].shape[0]).to(dev);net.load_state_dict(state);net.eval();feat=m.Featurizer(dev)
def entropy(bags,dark):
 keys=np.concatenate([dark,bags.reshape(-1,12)],axis=1)
 unique,inverse=np.unique(keys,axis=0,return_inverse=True)
 vals=np.array([sum(identity_entropy(k[2+c*6:2+(c+1)*6],int(k[c])) for c in range(2)) for k in unique])
 return vals[inverse]
def describe(v):return {'mean':float(v.mean()),'p10_p50_p90':[float(x) for x in np.quantile(v,[.1,.5,.9])],'min':float(v.min()),'max':float(v.max())} if len(v) else None
report={'schema':1,'initial_joint_entropy_bits':INITIAL_ENTROPY,'checkpoint_sha256':sha(a.checkpoint),'trainer_sha256':sha(source),'scope':'Uniform without-replacement urn prior, not a strategy-conditioned posterior. Entropy refers to future-relevant on-board identities. Position descriptions on training/selection-validation data, not strength evidence.','datasets':{}}
for entry in a.data:
 label,filename,expected=entry.split(':');expected=int(expected);path=Path(filename)
 if path.suffix!='.bin' or path.stat().st_size%112:raise SystemExit('Require complete records')
 meta=json.loads(path.with_suffix('.complete.json').read_text())
 if meta['sha256']!=sha(path) or meta['recipe']['games']!=expected or meta['rejected_games']:raise SystemExit('Generation attestation mismatch')
 rec=np.fromfile(path,dtype=m.REC);starts,ends=segment_games(rec,expected);raw=rec.view(np.uint8).reshape(-1,112);keep=m.keep_mask(raw)
 typ=rec['board']&15;col=rec['board']>>4
 dark=np.stack([((typ==8)&(col==c)).sum(1) for c in range(2)],axis=1)
 actor=entropy(rec['urn'].astype(np.int64),dark)
 pub,known=public_counts(rec,starts,ends);public=np.full(len(rec),np.nan);public[known]=entropy(pub[known],dark[known])
 full=dark.sum(1)==0;residual=full&(rec['urn'].sum((1,2))>0)
 stages={}
 for name,mask in [('full_public',full),('public_entropy_zero_with_dark',known&(public<1e-10)&~full),('actor_entropy_zero_with_dark',(actor<1e-10)&~full),('private_certainty_only',known&(actor<1e-10)&(public>1e-10)),('low_public_0_8_bits',known&(public>1e-10)&(public<=8)),('medium_public_8_32_bits',known&(public>8)&(public<=32)),('high_public_over_32_bits',known&(public>32))]:
  stages[name]={'raw_positions':int(mask.sum()),'retained_positions':int((mask&keep).sum())}
 rows=raw[residual&keep].copy();sensitivity={'positions':len(rows)}
 if len(rows):
  altered=rows.copy();altered[:,91:103]=0
  def predict(data):
   outputs=[]
   with torch.no_grad():
    for start in range(0,len(data),4096):
     board,stm,urn,*_=m.fields(torch.from_numpy(data[start:start+4096]).to(dev));us,them=feat(board,stm,urn);outputs.append((net(us,them)*net.out_scale).cpu().numpy())
   return np.concatenate(outputs)
  difference=np.abs(predict(rows)-predict(altered))
  sensitivity.update(absolute_cp_difference=describe(difference),positions_over_10_cp=int((difference>10).sum()),interpretation='Only off-board inventory features change; board and public rule history are unchanged. Sensitivity is not evidence that canonicalization improves playing strength.')
 # Compare each observer with their next decision; do not compare alternating private views.
 rises=transitions=0
 for start,end in zip(starts,ends):
  if end-start>2:
   rises+=int((actor[start+2:end]>actor[start:end-2]+1e-9).sum());transitions+=end-start-2
 report['datasets'][label]={'sha256':meta['sha256'],'games':expected,'raw_positions':len(rec),'retained_positions':int(keep.sum()),'public_prior_unavailable_positions':int((~known).sum()),'actor_entropy_bits':describe(actor),'public_entropy_bits':describe(public[known]),'stages':stages,'fully_revealed_with_residual_dead_inventory_raw':int(residual.sum()),'same_observer_entropy_increases':rises,'same_observer_transitions':int(transitions),'dead_inventory_value_sensitivity':sensitivity}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

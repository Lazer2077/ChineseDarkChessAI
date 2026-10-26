"""Paired game-level validation diagnostics for raw and full-public feature models.

This uses checkpoint-selection data: intervals describe correlated game samples,
not an independent generalization test or a playing-strength comparison.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import torch
from build_temporal_targets import segment_games

p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
p.add_argument('--games',type=int,required=True)
p.add_argument('--model',action='append',required=True,help='label:raw|canonical:checkpoint')
p.add_argument('--out',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
source=root/'external/devinokeefe-jieqi/engine/train/train.py'
spec=importlib.util.spec_from_file_location('phase_trainer',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
data=Path(a.data);out=Path(a.out)
if out.exists():raise SystemExit('Refuse to overwrite evidence')
meta=json.loads(data.with_suffix('.complete.json').read_text())
if data.suffix!='.bin' or data.stat().st_size%112 or meta['sha256']!=sha(data) or meta['recipe']['games']!=a.games or meta['rejected_games']:
 raise SystemExit('Require finalized complete valid trajectories')
rec=np.fromfile(data,dtype=m.REC)
starts,ends=segment_games(rec,a.games)
raw=rec.view(np.uint8).reshape(-1,112)
keep=m.keep_mask(raw)
game_ids=np.repeat(np.arange(a.games),ends-starts)[keep]
records=raw[keep].copy()
dark=((records[:,:90]&15)==8).sum(1)
stages={'all':np.ones(len(records),dtype=bool),'full_public':dark==0,
        'dark_1_4':(dark>=1)&(dark<=4),'dark_5_14':(dark>=5)&(dark<=14),'dark_15_30':dark>=15}
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_,_,_,score,result,_=m.fields(torch.from_numpy(records))
search=torch.sigmoid(score.float()/430).numpy()
outcome=((result.float()+1)/2).numpy()
target=.7*search+.3*outcome
report={'schema':1,'data_sha256':meta['sha256'],'games':a.games,'retained_positions':len(records),
 'trainer_sha256':sha(source),'bootstrap_seed':20261026,'bootstrap_samples':10000,
 'scope':'Checkpoint-selection validation data only; descriptive diagnostics, not an independent test or strength result. Intervals resample whole games and compare equal-weight per-game conditional phase errors. First listed model is the reference. No use of hidden truth identities.', 'models':{}}
reference=None
for item in a.model:
 label,policy,filename=item.split(':',2)
 if label in report['models'] or policy not in ('raw','canonical'):raise SystemExit('Invalid model specification')
 state=torch.load(filename,map_location='cpu',weights_only=True)
 features,ft=state['ft.weight'].shape
 if features!=1471:raise SystemExit('This diagnostic requires the one-bucket backbone')
 net=m.Net(ft-1,state['l1.weight'].shape[0],state['l2.weight'].shape[0]).to(dev)
 net.load_state_dict(state);net.eval();feat=m.Featurizer(dev)
 inputs=records.copy()
 if policy=='canonical':inputs[dark==0,91:103]=0
 predictions=[]
 with torch.no_grad():
  for start in range(0,len(inputs),4096):
   board,stm,urn,*_=m.fields(torch.from_numpy(inputs[start:start+4096]).to(dev))
   us,them=feat(board,stm,urn)
   predictions.append(torch.sigmoid(net(us,them)*net.out_scale/430).cpu().numpy())
 pred=np.concatenate(predictions)
 if not np.isfinite(pred).all():raise SystemExit('Nonfinite prediction')
 errors={'mixed_target_mse':(pred-target)**2,'outcome_brier':(pred-outcome)**2,'search_target_mse':(pred-search)**2}
 if reference is None:reference=errors
 groups={}
 for name,mask in stages.items():
  groups[name]={'positions':int(mask.sum())}
  counts=np.bincount(game_ids[mask],minlength=a.games);present=counts>0
  groups[name]['games_with_phase']=int(present.sum())
  if not present.any():continue
  for metric,err in errors.items():
   delta=err-reference[metric]
   per_game=np.bincount(game_ids[mask],weights=delta[mask],minlength=a.games)[present]/counts[present]
   # Reset seed across models/metrics so comparisons use the same game samples.
   rng=np.random.default_rng(20261026)
   boot=np.mean(per_game[rng.integers(len(per_game),size=(10000,len(per_game)))],axis=1)
   groups[name][metric]={'position_mean':float(err[mask].mean()),'game_mean_delta_vs_reference':float(per_game.mean()),'paired_game_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
 report['models'][label]={'checkpoint_sha256':sha(filename),'input_policy':policy,'strata':groups}
 del net
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)

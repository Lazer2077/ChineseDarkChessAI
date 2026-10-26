"""Construct game/observable-phase weights on certified, identically filtered replay."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from build_temporal_targets import segment_games

def replay_weights(game_ids, phase_ids, mode):
    game_ids=np.asarray(game_ids);phase_ids=np.asarray(phase_ids)
    if game_ids.ndim!=1 or phase_ids.shape!=game_ids.shape or not len(game_ids):
        raise ValueError('Require nonempty aligned game and phase indices')
    if mode=='position':return np.ones(len(game_ids),dtype=np.float32)
    _,games=np.unique(game_ids,return_inverse=True)
    if mode=='phase':
        _,inverse,counts=np.unique(phase_ids,return_inverse=True,return_counts=True)
        weights=1/counts[inverse]
    elif mode=='game':
        counts=np.bincount(games)
        weights=1/counts[games]
    elif mode=='game-phase':
        pairs,inverse,counts=np.unique(np.stack([games,phase_ids],axis=1),axis=0,return_inverse=True,return_counts=True)
        phases_per_game=np.bincount(pairs[:,0])
        weights=1/(counts[inverse]*phases_per_game[games])
    else:raise ValueError('Unknown weighting objective')
    return (weights/weights.mean()).astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--games',type=int,required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi/engine/train/train.py'
    spec=importlib.util.spec_from_file_location('weight_trainer',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    data=Path(a.data);out=Path(a.out)
    if out.exists():raise SystemExit('Use fresh output')
    meta=json.loads(data.with_suffix('.complete.json').read_text())
    if data.suffix!='.bin' or data.stat().st_size%112 or sha(data)!=meta['sha256'] or meta['recipe']['games']!=a.games or meta['rejected_games']:
        raise SystemExit('Require finalized complete valid games')
    rec=np.fromfile(data,dtype=m.REC);starts,ends=segment_games(rec,a.games)
    raw=rec.view(np.uint8).reshape(-1,112);keep=m.keep_mask(raw)
    games=np.repeat(np.arange(a.games),ends-starts)[keep]
    dark=((rec['board'][keep]&15)==8).sum(1)
    phases=np.digitize(dark,[1,5,15])
    out.mkdir(parents=True)
    report={'schema':1,'source_sha256':meta['sha256'],'trainer_sha256':sha(source),
      'completed_games':a.games,'games_with_retained_positions':len(np.unique(games)),
      'retained_positions':len(games),'phase_dark_counts':['0','1-4','5-14','15-30'],
      'scope':'Observable dark-count phase weighting; no hidden truth identity input, relabeling, duplication or phase boundary reset. A game without retained positions cannot contribute. No strength claim.', 'modes':{}}
    for mode in ['position','game','phase','game-phase']:
        w=replay_weights(games,phases,mode)
        file=out/(mode+'.f32');w.tofile(file)
        report['modes'][mode]={'sha256':sha(file),'mean':float(w.mean()),'min':float(w.min()),'max':float(w.max()),
          'weight_effective_sample_size':float(w.astype(np.float64).sum()**2/(w.astype(np.float64)**2).sum()),
          'mass_by_phase':[float(w[phases==i].sum()/w.sum()) for i in range(4)]}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

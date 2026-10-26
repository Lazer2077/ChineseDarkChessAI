"""Fixed-corpus graph-sharing diagnostics; never interpret timings as wins."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from research.policy_value import REC
from research.neural_search import load_model
from research.neural_replay import fresh_game,load_episode,save_json,sha
from research.public_continuation import PublicContinuationPUCT,SEARCH_SHA256

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Use fresh diagnostic output')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Diagnostic plan changed')
        for file,expected in plan['sha256'].items():
            if sha(ROOT/file)!=expected:raise ValueError('Frozen profile input changed: '+file)
    verify();out.mkdir(parents=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    net,_=load_model(ROOT/plan['model'],plan['device']);cases=[];control_counts={}
    for game in plan['games']:
        data,meta=load_episode(ROOT/plan['episodes']/f'game-{game:06d}.npz')
        state,belief,rng=fresh_game(meta['seed']);history=[]
        for ply,row in enumerate(data['records']):
            rec=row.reshape(1,112).view(REC)[0];dark=(rec['board']&15)==8;n=int(dark.sum());kind=None
            if n==1 and int(rec['board'][dark][0]>>4)==int(rec['stm']) and np.count_nonzero(rec['urn'][rec['stm']])>=2:
                kind='uncertain-last-own-dark'
            elif n==0 and control_counts.get('full',0)<2:kind='full'
            elif ply==0 and control_counts.get('opening',0)<2:kind='opening'
            if kind is not None:
                control_counts[kind]=control_counts.get(kind,0)+1
                cases.append(({'game':game,'ply':ply,'kind':kind,'dark_count':n},state.clone(),belief.clone(),history.copy()))
            move=tuple(map(int,data['moves'][ply]))
            from research.policy_value import AZ
            AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng);history.append(move)
    save_json(out/'selected-cases.json',[c[0] for c in cases])
    if not cases:raise ValueError('No diagnostic states')
    # Warm CUDA/model loading separately. Timing samples still remain exploratory.
    warm=PublicContinuationPUCT(net,plan['device'],seed=plan['seed'],leaf_batch=plan['leaf_batch'])
    warm.search(cases[0][1],cases[0][2],cases[0][3],temperature=1e-6,simulations=8)
    if plan['device']=='cuda':torch.cuda.synchronize()
    report={'schema':1,'status':'running','plan_sha256':digest,'timed_search_sha256':SEARCH_SHA256,
        'cases':[c[0] for c in cases],'samples':[],
        'scope':'Correlated replay-state graph diagnostics, not games, strength evidence or independent timing confidence. Original key control reproduces the pinned search; public sharing changes decision abstraction only after all on-board dark pieces disappear.'}
    save_json(out/'report.json',report)
    for index,(case,state,belief,history) in enumerate(cases):
        for repeat in range(plan['repeats']):
            for share in ([False,True] if (index+repeat)%2==0 else [True,False]):
                engine=PublicContinuationPUCT(net,plan['device'],seed=plan['seed']+index,leaf_batch=plan['leaf_batch'],share_public=share)
                started=time.perf_counter();policy,move,root=engine.search(state,belief,history,temperature=1e-6,simulations=plan['simulations'],return_root=True)
                elapsed=time.perf_counter()-started
                if root.visits!=plan['simulations'] or abs(sum(policy)-1)>1e-6 or move not in state.legal_moves():raise ValueError('Incomplete or invalid diagnostic search')
                report['samples'].append({'case':index,'repeat':repeat,'share_public':share,'elapsed_seconds':elapsed,
                    'root_visits':root.visits,'leaf_evaluations':engine.leaf_evals_total,'unique_leaf_evaluations':engine.leaf_evals_unique,
                    'action':list(move),'root_value':root.value_sum/root.visits,'graph':engine.graph_statistics()})
                save_json(out/'report.json',report)
        print(json.dumps({'case':index,**case,'completed_samples':len(report['samples'])}),flush=True)
    verify();report['status']='completed';save_json(out/'report.json',report)

if __name__=='__main__':main()

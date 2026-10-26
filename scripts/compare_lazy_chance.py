"""Matched fixed-simulation check of the inherited lazy chance-allocation option."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research.neural_replay import fresh_game
from research.neural_search import ObservationPUCT,load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Refuse to overwrite diagnostics')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,_=load_model(a.model,'cuda');rows=[];pairs=[]
    for repeat in range(3):
        fingerprints={};elapsed={}
        for lazy in ([False,True] if repeat%2==0 else [True,False]):
            state,belief,_=fresh_game(20261031)
            engine=ObservationPUCT(model,'cuda',seed=20261031,leaf_batch=32)
            engine.config.lazy_chance_allocation=lazy
            engine.search(state,belief,[],temperature=0,simulations=32,return_root=True)
            engine.rng.seed(20261031);torch.cuda.synchronize();start=time.perf_counter()
            policy,move,root=engine.search(state,belief,[],temperature=0,simulations=1024,return_root=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            fingerprints[lazy]=(move,root.visits,root.value_sum,tuple(policy),
                tuple((m,c.visits,c.value_sum,c.prior) for m,c in root.children.items()))
            elapsed[lazy]=seconds
            seen=set();pending=[root]
            while pending:
                node=pending.pop()
                if id(node) in seen:continue
                seen.add(id(node))
                for attr in ['children','public_outcomes','decision_nodes']:
                    pending.extend(getattr(node,attr).values())
            rows.append({'repeat':repeat,'lazy':lazy,'wall_seconds':seconds,
                'reachable_nodes':len(seen),'root_visits':root.visits,'root_value_sum':root.value_sum,
                'move':list(move)})
        identical=fingerprints[False]==fingerprints[True]
        pairs.append({'repeat':repeat,'exact_root_action_policy_and_child_statistics_match':identical,
            'control_over_lazy_wall_ratio':elapsed[False]/elapsed[True]})
        if not identical:raise ValueError('Lazy allocation changed fixed-simulation root statistics')
    report={'schema':1,'status':'completed','scope':'Inherited upstream lazy_chance_allocation configuration; no new algorithm. One initial fully hidden root, three alternating-order repeats, 1024 completed simulations per variant; no playing-strength or broad phase claim.',
        'seed':20261031,'leaf_batch':32,'simulations_per_search':1024,'total_measured_simulations':6144,
        'precision':'FP32 CUDA','device':'cuda','cpu_threads':1,'rows':rows,'pairs':pairs,
        'median_control_over_lazy_wall_ratio':statistics.median(p['control_over_lazy_wall_ratio'] for p in pairs),
        'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [
            'scripts/compare_lazy_chance.py','research/neural_search.py','research/neural_replay.py',
            'external/yuan-bai-jieqi-lab/azero.py',str(Path(a.model)/'best.npz')]}}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

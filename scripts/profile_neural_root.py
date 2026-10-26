"""Profile one declared initial root; timing evidence, never strength evidence."""
import argparse,cProfile,hashlib,json,pstats,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research.neural_replay import fresh_game
from research.neural_search import ObservationPUCT,load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Refuse to overwrite profile')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,_=load_model(a.model,'cuda');state,belief,_=fresh_game(20261031)
    engine=ObservationPUCT(model,'cuda',seed=20261031,leaf_batch=32)
    engine.search(state,belief,[],temperature=0,add_dirichlet=False,simulations=32,return_root=True)
    engine.rng.seed(20261031)
    profiler=cProfile.Profile();torch.cuda.synchronize();started=time.perf_counter()
    profiler.enable();policy,move,root=engine.search(state,belief,[],temperature=0,add_dirichlet=False,simulations=1024,return_root=True)
    profiler.disable();torch.cuda.synchronize();elapsed=time.perf_counter()-started
    if root.visits!=1024 or sum(ch.visits for ch in root.children.values())!=1024:
        raise ValueError('Incomplete root backups')
    stats=pstats.Stats(profiler);rows=[]
    for (file,line,name),(primitive,calls,total,cumulative,_) in stats.stats.items():
        path=Path(file)
        if path.is_absolute():
            try:file=str(path.resolve().relative_to(ROOT))
            except ValueError:file='runtime/'+path.name
        rows.append({'file':file,'line':line,'name':name,'primitive_calls':primitive,
            'calls':calls,'self_seconds':total,'cumulative_seconds':cumulative})
    report={'schema':1,'status':'completed','scope':'One fresh initial fully hidden root, no Dirichlet, warm-start shared model. cProfile adds overhead; cumulative times overlap and must not be summed. No match, general throughput or strength claim.',
        'seed':20261031,'simulations':1024,'leaf_batch':32,'warmup_simulations':32,
        'wall_seconds':elapsed,'simulations_per_second_with_profiler':1024/elapsed,
        'device':'cuda','cpu_threads':1,'move':list(move),'visit_mass':float(sum(policy)),
        'top_cumulative':sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:40],
        'top_self':sorted(rows,key=lambda r:r['self_seconds'],reverse=True)[:40],
        'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [
            'scripts/profile_neural_root.py','research/neural_search.py','research/neural_replay.py',
            'research/policy_value.py',str(Path(a.model)/'best.npz')]}}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'seconds':elapsed,'top_cumulative':report['top_cumulative'][:12]}),flush=True)


if __name__=='__main__':main()

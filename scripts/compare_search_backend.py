"""Isolated-process, fixed-root check of original Python/Cython expand methods."""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(plan,backend,repeat,out):
    import numpy as np
    import torch
    from research.policy_value import AZ
    from research.neural_replay import fresh_game,load_episode
    from research.neural_search import ObservationPUCT,load_model
    if AZ.SEARCH_BACKEND!=backend:raise ValueError('Requested backend not active')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    net,_=load_model(plan['model'],'cuda');cases=[]
    for case in plan['cases']:
        data,meta=load_episode(ROOT/plan['episodes']/f"game-{case['game']:06d}.npz")
        state,belief,rng=fresh_game(meta['seed']);history=[]
        for move in data['moves'][:case['ply']]:
            move=tuple(map(int,move));AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng);history.append(move)
        cases.append((state,belief,history))
    warm=ObservationPUCT(net,'cuda',seed=plan['seed'],leaf_batch=32)
    if not warm.config.lazy_chance_allocation:raise ValueError('Expected actual lazy default')
    warm.search(*cases[0],temperature=0,simulations=32)
    rows=[]
    for index,(state,belief,history) in enumerate(cases):
        engine=ObservationPUCT(net,'cuda',seed=plan['seed']+index,leaf_batch=32)
        torch.cuda.synchronize();start=time.perf_counter()
        policy,move,root=engine.search(state,belief,history,temperature=0,
            simulations=plan['simulations'],return_root=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        if root.visits!=plan['simulations'] or sum(c.visits for c in root.children.values())!=root.visits:
            raise ValueError('Incomplete backups')
        fingerprint={'move':list(move),'policy':policy,'root_visits':root.visits,'root_value_sum':root.value_sum,
            'children':[(list(m),c.visits,c.value_sum,c.prior) for m,c in root.children.items()]}
        rows.append({'case':index,'backend':backend,'repeat':repeat,'seconds':seconds,
            'fingerprint':fingerprint})
    out.write_text(json.dumps({'backend':backend,'rows':rows},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True)
    p.add_argument('--worker',choices=['python','cython']);p.add_argument('--repeat',type=int)
    p.add_argument('--output');a=p.parse_args();plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan)
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    verify()
    if a.worker:
        worker(plan,a.worker,a.repeat,Path(a.output));verify();return
    out=ROOT/plan['out']
    if out.exists():raise ValueError('Use a fresh diagnostic directory')
    out.mkdir(parents=True)
    report={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'samples':[],
        'scope':'Original optional Cython expand versus actual Python expand default, both lazy. Three correlated replay roots; no independent strength evidence. Alternating backend order in separate processes; CUDA load/32sim warmup excluded.'}
    def save():
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'report.json')
    save()
    try:
        for repeat in range(plan['repeats']):
            for backend in (['python','cython'] if repeat%2==0 else ['cython','python']):
                verify();target=out/f'{repeat}-{backend}.json'
                env=dict(os.environ,JIEQI_SEARCH_BACKEND=backend,
                    PYTHONPATH=str(ROOT/'build/yuan-search/source'),
                    YUAN_NATIVE_DIR=str(ROOT/'build/yuan-native/source'),
                    JIEQI_RULES_BACKEND='cython',JIEQI_LEGAL_BACKEND='cython')
                with (out/f'{repeat}-{backend}.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,__file__,'--plan',a.plan,'--worker',backend,
                        '--repeat',str(repeat),'--output',str(target)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    report['child_pid']=child.pid;save()
                    if child.wait():raise RuntimeError('Backend child failed; see '+log.name)
                verify();report['samples'].extend(json.loads(target.read_text())['rows']);save()
        pairs=[]
        for repeat in range(plan['repeats']):
            for case in range(len(plan['cases'])):
                rows={r['backend']:r for r in report['samples'] if r['repeat']==repeat and r['case']==case}
                same=rows['python']['fingerprint']==rows['cython']['fingerprint']
                pairs.append({'repeat':repeat,'case':case,'exact_root_statistics_match':same,
                    'python_over_cython_seconds':rows['python']['seconds']/rows['cython']['seconds']})
        report['pairs']=pairs
        report['all_fixed_simulation_roots_match']=all(p['exact_root_statistics_match'] for p in pairs)
        report['median_ratios_by_case']={str(i):statistics.median(p['python_over_cython_seconds'] for p in pairs if p['case']==i) for i in range(len(plan['cases']))}
        verify();report['status']='completed';report['child_pid']=None;save()
    except BaseException as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()

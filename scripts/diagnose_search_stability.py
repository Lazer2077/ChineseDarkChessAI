"""Repeated-seed search stability, separating hidden and fully public roots."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from research.policy_value import AZ
from research.neural_replay import fresh_game,load_episode,sha,save_json
from research.neural_search import ObservationPUCT,load_model


def summarize(samples):
    policies=np.asarray([r['policy'] for r in samples]);values=[r['root_value'] for r in samples]
    actions=[tuple(r['move']) for r in samples]
    distribution={str(m):actions.count(m) for m in sorted(set(actions))}
    distances=[float(np.abs(policies[i]-policies[j]).sum()/2)
        for i in range(len(samples)) for j in range(i)]
    children={tuple(c['move']) for r in samples for c in r['children']};rows=[]
    for move in sorted(children):
        found=[next(c for c in r['children'] if tuple(c['move'])==move) for r in samples]
        observed=[c['root_q'] for c in found if c['visits']]
        rows.append({'move':list(move),'minimum_visits':min(c['visits'] for c in found),
            'mean_visits':statistics.mean(c['visits'] for c in found),
            'q_observed_seeds':len(observed),
            'mean_root_q':statistics.mean(observed) if observed else None,
            'root_q_std':statistics.stdev(observed) if len(observed)>1 else None})
    return {'seeds':len(samples),'selected_action_counts':distribution,
        'modal_action_fraction':max(distribution.values())/len(samples),
        'mean_root_value':statistics.mean(values),'root_value_std':statistics.stdev(values),
        'mean_pairwise_policy_total_variation':statistics.mean(distances),
        'all_root_statistics_identical':all((r['policy'],r['children'],r['root_value'])==
            (samples[0]['policy'],samples[0]['children'],samples[0]['root_value']) for r in samples),
        'children':sorted(rows,key=lambda r:r['mean_visits'],reverse=True)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    digest=sha(a.plan);plan=json.loads(Path(a.plan).read_text());out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite a stability diagnostic')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    verify();out.mkdir(parents=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    net,_=load_model(plan['model'],'cuda');cases=[]
    for case in plan['cases']:
        data,meta=load_episode(ROOT/plan['episodes']/f"game-{case['game']:06d}.npz")
        state,belief,rng=fresh_game(meta['seed']);history=[]
        for move in data['moves'][:case['ply']]:
            move=tuple(map(int,move));AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng);history.append(move)
        cases.append((state,belief,history))
    report={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'samples':[],
        'scope':'Repeated search RNG seeds on three correlated TRAINING replay roots, no Dirichlet, temperature0. Adaptive-search stability, not unbiased value estimates, confidence intervals, independent games or strength evidence.'}
    save_json(out/'report.json',report)
    try:
        warm=ObservationPUCT(net,'cuda',seed=0,leaf_batch=32)
        warm.search(*cases[0],temperature=0,simulations=32)
        for case_index,(state,belief,history) in enumerate(cases):
            for simulations in plan['simulation_budgets']:
                for seed in plan['search_seeds']:
                    engine=ObservationPUCT(net,'cuda',seed=seed,leaf_batch=32)
                    torch.cuda.synchronize();started=time.perf_counter()
                    policy,move,root=engine.search(state,belief,history,temperature=0,
                        simulations=simulations,add_dirichlet=False,return_root=True)
                    torch.cuda.synchronize()
                    if root.visits!=simulations or sum(c.visits for c in root.children.values())!=simulations:
                        raise ValueError('Incomplete root backups')
                    report['samples'].append({'case':case_index,'simulations':simulations,'seed':seed,
                        'seconds':time.perf_counter()-started,'move':list(move),'policy':policy,
                        'root_value':root.value_sum/root.visits,
                        'children':[{'move':list(m),'visits':c.visits,
                            'root_q':-c.value_sum/c.visits if c.visits else None}
                            for m,c in root.children.items()]})
                    save_json(out/'report.json',report)
            print(json.dumps({'completed_case':case_index,'samples':len(report['samples'])}),flush=True)
        report['summaries']=[{'case':i,'phase':case['phase'],'simulations':budget,
            **summarize([r for r in report['samples'] if r['case']==i and r['simulations']==budget])}
            for i,case in enumerate(plan['cases']) for budget in plan['simulation_budgets']]
        verify();report['status']='completed';save_json(out/'report.json',report)
    except BaseException as error:
        report.update(status='failed',error=str(error));save_json(out/'report.json',report);raise


if __name__=='__main__':main()

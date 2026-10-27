"""Three reused TRAIN-root diagnostic: same search, different policy targets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import load_episode,fresh_game,sha,save_json
from research.neural_search import ObservationPUCT,load_model
from research.policy_value import AZ
from research.native_opponent_policy import FixedDepthOpponent
from research.teacher_leaf import observer_fen
from research.completed_q_target import completed_q_target


def entropy(p):
    positive=p>0
    return float(-(p[positive]*np.log(p[positive])).sum())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    path=Path(args.plan);plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Q-target plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen diagnostic input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    net,meta=load_model(ROOT/plan['model'],'cpu')
    report={'schema':1,'status':'running','scope':plan['scope'],'plan_sha256':digest,
            'model_sha256':meta['npz_sha256'],'cases':[]}
    save_json(out/'report.json',report)
    with FixedDepthOpponent(ROOT/plan['teacher_binary'],ROOT/plan['teacher_net'],depth=plan['teacher_depth']) as teacher:
        for case in plan['cases']:
            data,metadata=load_episode(ROOT/plan['episodes']/f"game-{case['game']:06d}.npz")
            state,belief,rng=fresh_game(metadata['seed']);history=[]
            for raw_move in data['moves'][:case['ply']]:
                move=tuple(map(int,raw_move));AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng);history.append(move)
            engine=ObservationPUCT(net,seed=plan['world_seed'],leaf_batch=plan['leaf_batch'])
            info=belief.info_state(state,state.side);_,value=engine._evaluate(info,history)
            pi,move,root=engine.search(state,belief,history,simulations=plan['simulations'],temperature=0.,return_root=True)
            moves=list(root.children);children=list(root.children.values())
            visits=np.asarray([c.visits for c in children],dtype=np.int64)
            if root.visits!=plan['simulations'] or visits.sum()!=root.visits:raise ValueError('Incomplete root backups')
            prior=np.asarray([c.prior for c in children],dtype=np.float64);prior/=prior.sum()
            q=np.asarray([-c.q_value() if c.visits else np.nan for c in children])
            scale=plan['q_scale']*(plan['c_visit']+int(visits.max()))
            target=completed_q_target(prior,visits,q,value,scale);recommend=target['policy']
            actions=[AZ.jq.format_move_uci(m) for m in moves]
            scores=teacher.scores(observer_fen(info),actions);cp=np.asarray([scores[a] for a in actions],dtype=np.float64)
            count_policy=visits/visits.sum()
            if not np.allclose([pi[AZ.move_to_index(m)] for m in moves],count_policy,atol=1e-7):
                raise ValueError('Count policy differs from actual completed visits')
            row={**case,'root_network_value':value,'completed_simulations':root.visits,'visited_actions':int((visits>0).sum()),
                'legal_actions':len(moves),'scale':scale,'mixed_completion_value':target['mixed_value'],
                'own_completed_q_objective_gain':target['estimated_completed_q_gain'],
                'target_to_prior_kl':target['target_to_prior_kl'],
                'policy_entropy':{k:entropy(p) for k,p in [('prior',prior),('visits',count_policy),('completed_q',recommend)]},
                'expected_teacher_history_reset_depth_proxy_cp':{k:float(np.dot(p,cp)) for k,p in [('prior',prior),('visits',count_policy),('completed_q',recommend)]},
                'selected_by_original_search':AZ.jq.format_move_uci(move),
                'actions':[{'move':a,'prior':float(prior[i]),'visits':int(visits[i]),
                    'root_q':float(q[i]) if visits[i] else None,'completed_q':float(target['completed_q'][i]),
                    'target':float(recommend[i]),'teacher_proxy_cp':scores[a]} for i,a in enumerate(actions)]}
            report['cases'].append(row);verify();save_json(out/'report.json',report)
            print(json.dumps({k:v for k,v in row.items() if k!='actions'}),flush=True)
    verify();report['status']='completed';save_json(out/'report.json',report)


if __name__=='__main__':main()

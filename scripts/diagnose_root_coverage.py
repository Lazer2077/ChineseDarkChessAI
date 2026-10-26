"""Measure legal-action coverage of real completed self-play visit targets."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.policy_value import AZ
from research.neural_replay import fresh_game,load_episode,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-plan',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise ValueError('Refuse to overwrite target diagnostics')
    digest=sha(a.training_plan);plan=json.loads(Path(a.training_plan).read_text())
    generation=json.loads((ROOT/plan['generation_plan']).read_text());folder=ROOT/generation['out']
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['plan_sha256']!=sha(ROOT/plan['generation_plan']):
        raise ValueError('Require complete frozen generation')
    entries={r['index']:r for r in manifest['episodes']};groups={};game_rows=[]
    def verify():
        if sha(a.training_plan)!=digest:raise ValueError('Training plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    verify()
    for index in plan['training_games']:
        data,meta=load_episode(folder/f'game-{index:06d}.npz')
        expected=dict(entries[index]);expected.pop('index')
        if meta!=expected:raise ValueError('Episode manifest mismatch')
        state,belief,rng=fresh_game(meta['seed']);local=[]
        for i,counts in enumerate(data['visit_counts']):
            legal=state.legal_moves();ids=[AZ.move_to_index(m) for m in legal]
            visits=counts[ids].astype(np.float64);total=int(visits.sum())
            if total!=meta['simulations']:raise ValueError('Incomplete or illegal visit support')
            pi=visits/total;nonzero=pi>0;visited=int(nonzero.sum())
            entropy=float(-(pi[nonzero]*np.log(pi[nonzero])).sum())
            dark=int(((data['records'][i,:90]&15)==8).sum())
            phase='fully-public' if dark==0 else ('dark-1-to-5' if dark<=5 else 'dark-6-plus')
            row=[len(legal),visited,visited/len(legal),int(visited<len(legal)),
                entropy,float(np.exp(entropy)),float(pi.max()),int((visits==1).sum())]
            groups.setdefault(phase,[]).append(row);local.append(row)
            move=tuple(map(int,data['moves'][i]));AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
        game_rows.append(np.mean(local,axis=0))
    names=['legal_actions','visited_actions','fraction_legal_visited','any_unvisited_legal_action',
        'visit_entropy_nats','effective_visit_actions','maximum_visit_mass','actions_visited_once']
    def summary(rows):
        arr=np.asarray(rows,dtype=float)
        return {'count':len(rows),'means':dict(zip(names,arr.mean(0).tolist()))}
    verify()
    report={'schema':1,'status':'completed','scope':'Real first-iteration TRAINING episodes only, no selection-validation or strength-test data. Position means and equal-game means; correlated samples, not strength evidence. Targets include exploration noise, so coverage alone cannot establish policy improvement or its absence.',
        'training_plan':a.training_plan,'training_plan_sha256':digest,
        'generation_manifest_sha256':sha(folder/'manifest.json'),'games':len(game_rows),
        'by_phase':{key:summary(rows) for key,rows in groups.items()},
        'all_positions':summary([r for rows in groups.values() for r in rows]),
        'equal_game':summary(game_rows),'script_sha256':sha(__file__),
        'interpretation_reference':'https://openreview.net/pdf?id=bERaNdoegnO',
        'theory_limit':'Gumbel policy-improvement results assume correct action-value evaluation; no direct guarantee for our approximate private-information search.'}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

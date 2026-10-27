"""Describe expert/visit agreement without mistaking sampled moves for argmax."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import load_episode,sha,save_json
from research.policy_value import AZ


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Expert label diagnostic plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen label diagnostic input changed: '+name)
    verify();output=ROOT/plan['out']
    if output.exists():raise ValueError('Refuse to overwrite diagnostic')
    source=json.loads((ROOT/plan['annotation_report']).read_text())
    if source['status']!='completed':raise ValueError('Require completed annotation')
    rows=[];quarantined=[]
    for entry in source['episodes']:
        game=entry['game']
        if entry['status']=='quarantined':quarantined.append(game);continue
        if entry['status']!='compatible-roots':raise ValueError('Unknown annotation status')
        f=ROOT/plan['labels']/f'game-{game:06d}-labels.json'
        if sha(f)!=entry['labels_sha256']:raise ValueError('Expert labels changed')
        labels=json.loads(f.read_text())
        raw=ROOT/plan['episodes']/f'game-{game:06d}.npz';data,metadata=load_episode(raw)
        if sha(raw)!=entry['source_npz_sha256'] or labels['source_npz_sha256']!=sha(raw):
            raise ValueError('Expert labels refer to another episode')
        if len(labels['labels'])!=metadata['positions']:raise ValueError('Partial labels')
        for ply,label in enumerate(labels['labels']):
            if label['ply']!=ply or label['actor']!=int(data['sides'][ply]):raise ValueError('Wrong label root')
            action=label['expert_move'];move=(AZ.jq.name_to_square(action[:2]),AZ.jq.name_to_square(action[2:]))
            counts=data['visit_counts'][ply];n=int(data['root_visits'][ply]);count=int(counts[AZ.move_to_index(move)])
            sampled=AZ.jq.format_move_uci(tuple(map(int,data['moves'][ply])))
            if sampled!=label['learner_move']:raise ValueError('Wrong sampled learner action')
            rows.append({'game':game,'ply':ply,'expert_visit_mass':count/n,'unvisited':count==0,
                'tied_visit_max':count==int(counts.max()),'matches_sampled':action==sampled,
                'public':not ((data['records'][ply,:90]&15)==8).any()})
    def summarize(items):
        if not items:return {'positions':0}
        return {'positions':len(items),'games':len({r['game'] for r in items}),
                'mean_expert_visit_mass':float(np.mean([r['expert_visit_mass'] for r in items])),
                **{key:int(sum(r[key] for r in items)) for key in ['unvisited','tied_visit_max','matches_sampled']}}
    report={'schema':1,'status':'completed','scope':plan['scope'],'plan_sha256':digest,
        'quarantined_games_excluded':quarantined,'strata':{
        'all':summarize(rows),'first30_temperature1':summarize([r for r in rows if r['ply']<30]),
        'later_temperature_point1':summarize([r for r in rows if r['ply']>=30]),
        'fully_public':summarize([r for r in rows if r['public']])}}
    verify();save_json(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

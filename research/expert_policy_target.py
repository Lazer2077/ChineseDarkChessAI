"""Replace only policy targets with bound observer-only expert actions."""
import json
from pathlib import Path
import numpy as np
from research.neural_replay import sha
from research.policy_value import AZ
from scripts.train_neural_selfplay import load_split as original_load_split

ROOT=Path(__file__).resolve().parents[1]


def annotation_report(plan):
    file=ROOT/plan['annotation_report']
    if plan['sha256'].get(plan['annotation_report'])!=sha(file):raise ValueError('Annotation report not pinned')
    report=json.loads(file.read_text())
    if report['status']!='completed':raise ValueError('Annotation not completed')
    return report


def validate_split(plan,generation,train_ids,val_ids):
    report=annotation_report(plan)
    accepted={r['game'] for r in report['episodes'] if r['status']=='compatible-roots'}
    ids=train_ids+val_ids
    if len(set(ids))!=len(ids) or not set(ids)<=accepted:raise ValueError('Overlapping or quarantined annotation split')
    if not set(ids)<=set(range(generation['games'])):raise ValueError('Game outside generating corpus')
    source=ROOT/plan['annotation_plan']
    if plan['sha256'].get(plan['annotation_plan'])!=sha(source):raise ValueError('Annotation recipe not pinned')
    original=json.loads(source.read_text())
    if report['plan_sha256']!=sha(source):raise ValueError('Wrong annotation recipe')
    if plan['research_stage']=='integration':
        if plan['device']!='cpu' or plan['epochs']!=1:raise ValueError('Integration must be a one-epoch CPU smoke')
    elif plan['research_stage']=='development':
        if set(train_ids)!=accepted.intersection(original['training_games']) or set(val_ids)!=accepted.intersection(original['selection_validation_games']):
            raise ValueError('Development split differs from frozen annotation roles')
    else:raise ValueError('Unknown expert-target research stage')


def load_split(plan,folder,indices,manifest):
    raw,pi,z,metadata=original_load_split(folder,indices,manifest)
    report=annotation_report(plan);entries={r['game']:r for r in report['episodes']}
    expert_ids=[]
    for meta in metadata:
        row=entries[meta['index']]
        if row['status']!='compatible-roots' or row['source_npz_sha256']!=meta['sha256'] or row['source_positions']!=meta['positions']:
            raise ValueError('Expert episode differs from learner replay')
        file=ROOT/plan['expert_labels']/f"game-{meta['index']:06d}-labels.json"
        if sha(file)!=row['labels_sha256']:raise ValueError('Expert label digest differs')
        data=json.loads(file.read_text())
        if data['source_npz_sha256']!=meta['sha256'] or len(data['labels'])!=meta['positions']:raise ValueError('Incomplete/foreign labels')
        # These are already replay-certified source arrays, not CP/mate targets.
        from research.neural_replay import load_episode
        episode,_=load_episode(folder/f"game-{meta['index']:06d}.npz")
        for ply,label in enumerate(data['labels']):
            if label['ply']!=ply or label['actor']!=int(episode['sides'][ply]):raise ValueError('Expert label root shifted')
            played=AZ.jq.format_move_uci(tuple(map(int,episode['moves'][ply])))
            if label['learner_move']!=played:raise ValueError('Expert label attached to another history')
            action=label['expert_move']
            if len(action)!=4:raise ValueError('Require a coordinate expert action')
            move=(AZ.jq.name_to_square(action[:2]),AZ.jq.name_to_square(action[2:]))
            expert_ids.append(AZ.move_to_index(move))
    if len(expert_ids)!=len(raw):raise ValueError('Wrong number of expert policy targets')
    if plan['policy_target']=='observer-expert-onehot':
        target=np.zeros_like(pi);target[np.arange(len(raw)),expert_ids]=1
        pi=target
    elif plan['policy_target']!='original-search-visits':raise ValueError('Unknown policy target')
    return raw,pi,z,metadata

"""Factorial policy/value supervision; score labels are declared proxies."""
import json
from pathlib import Path
import numpy as np
from research import expert_policy_target as original
from research.expert_value_diagnostic import search_score_proxy
from research.neural_replay import sha

ROOT=Path(__file__).resolve().parents[1]


def load_split(plan,folder,indices,manifest):
    raw,pi,z,metadata=original.load_split(plan,folder,indices,manifest)
    mode=plan['value_target']
    if mode=='original-learner-terminal':return raw,pi,z,metadata
    if mode!='observer-expert-search-proxy':raise ValueError('Unknown value-target mode')
    if plan['value_score_scale']!=860.:raise ValueError('This experiment fixes the score scale at860')
    report=original.annotation_report(plan);entries={r['game']:r for r in report['episodes']};values=[]
    for meta in metadata:
        row=entries[meta['index']];file=ROOT/plan['expert_labels']/f"game-{meta['index']:06d}-labels.json"
        if sha(file)!=row['labels_sha256']:raise ValueError('Expert value-label digest differs')
        labels=json.loads(file.read_text())
        if labels['source_npz_sha256']!=meta['sha256'] or len(labels['labels'])!=meta['positions']:
            raise ValueError('Wrong value-label episode')
        for ply,label in enumerate(labels['labels']):
            if label['ply']!=ply:raise ValueError('Value labels reordered')
            values.append(search_score_proxy(label['score'],860.))
    target=np.asarray(values,dtype=z.dtype)
    if target.shape!=z.shape or not np.isfinite(target).all():raise ValueError('Invalid typed value proxies')
    return raw,pi,target,metadata

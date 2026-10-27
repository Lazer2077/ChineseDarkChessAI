import copy
import json
from pathlib import Path
import numpy as np
import pytest
from research.neural_replay import sha
import research.expert_policy_target as target

ROOT=Path(__file__).resolve().parents[1]


def recipe():
    report='results/20261027-observer-expert-relabel-corpus.json'
    plan='results/observer-expert-relabel-corpus-plan.json'
    if not (ROOT/report).exists():pytest.skip('Bound local research corpus unavailable')
    return {'annotation_report':report,'annotation_plan':plan,
        'expert_labels':'runs/observer-expert-relabel-corpus-01','research_stage':'integration',
        'device':'cpu','epochs':1,'policy_target':'original-search-visits',
        'sha256':{report:sha(ROOT/report),plan:sha(ROOT/plan)}}


def test_exact_role_and_whole_episode_exclusions():
    plan=recipe();source=json.loads((ROOT/plan['annotation_plan']).read_text())
    report=json.loads((ROOT/plan['annotation_report']).read_text())
    accepted={r['game'] for r in report['episodes'] if r['status']=='compatible-roots'}
    train=[i for i in source['training_games'] if i in accepted]
    val=[i for i in source['selection_validation_games'] if i in accepted]
    plan['research_stage']='development';target.validate_split(plan,{'games':288},train,val)
    with pytest.raises(ValueError,match='quarantined'):target.validate_split(plan,{'games':288},train+[31],val)
    with pytest.raises(ValueError,match='roles'):target.validate_split(plan,{'games':288},train[1:],val+[train[0]])


def test_policy_substitution_preserves_observation_and_terminal_targets():
    plan=recipe();folder=ROOT/'runs/legal-anchor-integration-01/episodes'
    manifest=json.loads((folder/'manifest.json').read_text())
    original=target.load_split(plan,folder,[0],manifest)
    plan['policy_target']='observer-expert-onehot';expert=target.load_split(plan,folder,[0],manifest)
    assert np.array_equal(original[0],expert[0]) and np.array_equal(original[2],expert[2])
    assert original[3]==expert[3] and original[1].dtype==expert[1].dtype
    assert np.all(expert[1].sum(1)==1) and np.all(np.count_nonzero(expert[1],axis=1)==1)
    assert not np.array_equal(original[1],expert[1])


def test_cp_mate_fields_are_unused_and_label_corruption_is_rejected(tmp_path,monkeypatch):
    plan=recipe();plan['policy_target']='observer-expert-onehot'
    folder=ROOT/'runs/legal-anchor-integration-01/episodes';manifest=json.loads((folder/'manifest.json').read_text())
    before=target.load_split(plan,folder,[0],manifest)
    report=json.loads((ROOT/plan['annotation_report']).read_text())
    labels=json.loads((ROOT/plan['expert_labels']/'game-000000-labels.json').read_text())
    for row in labels['labels']:row['score']={'future_truth_poison':'must not become a value target'}
    (tmp_path/'labels').mkdir();file=tmp_path/'labels/game-000000-labels.json';file.write_text(json.dumps(labels))
    report['episodes'][0]['labels_sha256']=sha(file)
    r=tmp_path/'report.json';r.write_text(json.dumps(report))
    plan=copy.deepcopy(plan);plan.update(annotation_report='report.json',expert_labels='labels')
    plan['sha256']['report.json']=sha(r);monkeypatch.setattr(target,'ROOT',tmp_path)
    after=target.load_split(plan,folder,[0],manifest)
    assert all(np.array_equal(a,b) for a,b in zip(before[:3],after[:3]))
    labels['labels'][0]['expert_move']='a0a1';file.write_text(json.dumps(labels))
    with pytest.raises(ValueError,match='digest'):target.load_split(plan,folder,[0],manifest)

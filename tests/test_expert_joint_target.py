import json
from pathlib import Path
import numpy as np
import pytest
from research import expert_policy_target as original
from research import expert_joint_target as joint
from research.neural_replay import sha

ROOT=Path(__file__).resolve().parents[1]


def recipe(policy='original-search-visits'):
    p=ROOT/'results/expert-policy-integration-control-plan.json'
    if not p.exists():pytest.skip('Local bound research corpus unavailable')
    plan=json.loads(p.read_text());plan.update(policy_target=policy,value_target='original-learner-terminal',value_score_scale=860.)
    folder=ROOT/'runs/legal-anchor-integration-01/episodes'
    return plan,folder,json.loads((folder/'manifest.json').read_text())


@pytest.mark.parametrize('policy',['original-search-visits','observer-expert-onehot'])
def test_terminal_control_is_exact_old_loader(policy):
    plan,folder,manifest=recipe(policy)
    before=original.load_split(plan,folder,[0],manifest);after=joint.load_split(plan,folder,[0],manifest)
    assert all(np.array_equal(a,b) for a,b in zip(before[:3],after[:3])) and before[3]==after[3]


def test_only_values_change_and_scores_keep_actor_sign():
    plan,folder,manifest=recipe('observer-expert-onehot');terminal=joint.load_split(plan,folder,[0],manifest)
    plan['value_target']='observer-expert-search-proxy';proxy=joint.load_split(plan,folder,[0],manifest)
    assert np.array_equal(terminal[0],proxy[0]) and np.array_equal(terminal[1],proxy[1]) and terminal[3]==proxy[3]
    assert terminal[2].dtype==proxy[2].dtype and not np.array_equal(terminal[2],proxy[2])
    labels=json.loads((ROOT/plan['expert_labels']/'game-000000-labels.json').read_text())['labels']
    expected=[np.tanh(row['score']['value']/860.) if row['score']['kind']=='cp' else np.sign(row['score']['value']) for row in labels]
    assert np.allclose(proxy[2],expected,rtol=0,atol=3e-8)
    plan['value_score_scale']=600.
    with pytest.raises(ValueError,match='fixes'):joint.load_split(plan,folder,[0],manifest)

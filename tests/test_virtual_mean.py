import random
import pytest
import torch
from research.policy_value import AZ,PhasePolicyValueNet
from research.neural_replay import fresh_game,episode,audit_episode
from research.neural_search import ObservationPUCT
from research.virtual_mean import CreditPath,VirtualMeanPUCT


def test_pending_mean_preserves_q_and_replaces_same_player_chance_credits():
    root=AZ.PUCTNode(1.,0);chance=AZ.PUCTNode(1.,1);outcome=AZ.PUCTNode(1.,1)
    chance.visits=2;chance.value_sum=1.
    first=CreditPath(root);first.reserve(chance);first.append(chance)
    first.reserve(outcome);first.extend([outcome])
    assert chance.q_value()==pytest.approx(.5)
    second=CreditPath(root);second.reserve(chance);second.append(chance)
    second.reserve(outcome);second.append(outcome)
    # Back up a later path first: credits belong to paths, not a FIFO on nodes.
    VirtualMeanPUCT._backup(second,.25,3.)
    VirtualMeanPUCT._backup(first,-.75,3.)
    assert root.visits==2 and root.value_sum==pytest.approx(.5)
    assert chance.visits==4 and chance.value_sum==pytest.approx(.5)
    assert outcome.visits==2 and outcome.value_sum==pytest.approx(-.5)
    with pytest.raises(ValueError):VirtualMeanPUCT._backup(first,0.,3.)


def test_single_leaf_batch_matches_sequential_original_with_rounding_tolerance():
    torch.set_num_threads(1);torch.manual_seed(19)
    net=PhasePolicyValueNet(4,1).eval();outputs=[]
    for kind in [ObservationPUCT,VirtualMeanPUCT]:
        state,belief,_=fresh_game(13)
        engine=kind(net,'cpu',seed=71,leaf_batch=1)
        pi,move,root=engine.search(state,belief,[],temperature=0,simulations=32,return_root=True)
        outputs.append((pi,move,root))
    assert outputs[0][:2]==outputs[1][:2]
    a,b=outputs[0][2],outputs[1][2]
    assert a.visits==b.visits==32
    assert a.value_sum==pytest.approx(b.value_sum,abs=1e-10)
    for move in a.children:
        assert a.children[move].visits==b.children[move].visits
        assert a.children[move].value_sum==pytest.approx(b.children[move].value_sum,abs=1e-10)


def test_batched_real_game_keeps_completed_visits_and_auditable_private_observations():
    torch.set_num_threads(1);torch.manual_seed(7)
    net=PhasePolicyValueNet(4,1).eval()
    engine=VirtualMeanPUCT(net,'cpu',seed=11,leaf_batch=4)
    data,meta=episode(engine,seed=8260001,simulations=4)
    audit=audit_episode(data,meta)
    assert audit['observer_records_match'] and audit['root_visit_counts_match']
    assert audit['terminal_values_match'] and audit['positions_replayed']>0

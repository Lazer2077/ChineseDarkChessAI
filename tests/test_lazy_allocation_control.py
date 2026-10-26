"""The faster lazy mode is already enabled; this is not a new optimization."""
import torch
from research.policy_value import PhasePolicyValueNet
from research.neural_replay import fresh_game
from research.neural_search import ObservationPUCT


def test_actual_default_and_explicit_eager_counterfactual_agree():
    torch.set_num_threads(1);torch.manual_seed(42)
    net=PhasePolicyValueNet(4,1).eval();fingerprints=[]
    assert ObservationPUCT(net).config.lazy_chance_allocation is True
    for lazy in [False,True]:
        state,belief,_=fresh_game(73)
        engine=ObservationPUCT(net,'cpu',seed=89,leaf_batch=8)
        engine.config.lazy_chance_allocation=lazy
        pi,move,root=engine.search(state,belief,[],temperature=0,simulations=64,return_root=True)
        fingerprints.append((move,pi,root.visits,root.value_sum,
            [(m,c.visits,c.value_sum) for m,c in root.children.items()]))
    assert fingerprints[0]==fingerprints[1]
    assert ObservationPUCT(net).config.lazy_chance_allocation is True

from types import SimpleNamespace
import torch
import pytest
from research.anytime_halving import AnytimeHalvingPUCT, SurvivorHalving
from research.sequential_halving import HalvingPUCT
from research.policy_value import PhasePolicyValueNet
from adapters.yuan_observer import reconstruct


def summary(result):
    policy, move, root = result
    return policy, move, root.visits, root.value_sum, [(m,c.visits,c.value_sum) for m,c in root.children.items()]


def model():
    torch.set_num_threads(1); torch.manual_seed(7)
    return PhasePolicyValueNet(4,1).eval()


def test_exact_prefix_matches_completed_fixed_work_variant():
    net = model(); state, belief, _ = reconstruct(['b2b9R'])
    fixed = HalvingPUCT(net, seed=13, leaf_batch=8)
    timed = AnytimeHalvingPUCT(net, seed=13, leaf_batch=8)
    a = fixed.search(state,belief,[],simulations=256,gumbel_seed=0,return_root=True)
    b = timed.search(state,belief,[],simulations=256,return_root=True)
    assert summary(a) == summary(b)
    assert timed.halving.tail_started == 256 and len(timed.halving.active) == 2


def test_tail_reuses_tree_and_stays_inside_two_survivors():
    net = model(); state,belief,_ = reconstruct([])
    engine = AnytimeHalvingPUCT(net,seed=19,leaf_batch=8,prefix_simulations=32)
    policy,move,root = engine.search(state,belief,[],simulations=96,return_root=True)
    assert root.visits == engine.halving.assigned == 96 and sum(policy) == pytest.approx(1.)
    assert engine.halving.tail_started == 32 and move in state.legal_moves()
    assert all(c.visits <= 8 for m,c in root.children.items() if m not in engine.halving.active)


def test_expired_deadline_finishes_one_batch_with_no_pending_virtual_values():
    state,belief,_ = reconstruct([]); engine = AnytimeHalvingPUCT(model(),leaf_batch=8)
    engine.deadline = 0.
    policy,move,root = engine.search(state,belief,[],simulations=1000000,return_root=True)
    assert root.visits == 8 and engine.halving.assigned == 8
    assert move in state.legal_moves() and sum(policy) == pytest.approx(1.)
    assert all(abs(c.value_sum) <= c.visits+1e-6 for c in root.children.values())


def test_actor_private_isolation_with_tail_search():
    net = model(); outputs=[]
    for poison in (False,True):
        state,belief,_ = reconstruct(['b2b9R'])
        if poison:
            belief.pool_counts[1-state.side] = [[0]*8,[0]*8]
            belief.pool_total[1-state.side] = [0,0]; belief.unknown_removals[1-state.side] = [99,99]
        engine = AnytimeHalvingPUCT(net,seed=17,leaf_batch=8,prefix_simulations=32)
        outputs.append(summary(engine.search(state,belief,[],simulations=96,return_root=True)))
    assert outputs[0] == outputs[1]


def test_tail_uses_conditioned_survivor_priors_and_refuses_pending_boundary():
    class Edge:
        def __init__(self,p,n,q): self.prior=p; self.visits=n; self.value_sum=n*q
        def q_value(self): return self.value_sum/self.visits
    root = SimpleNamespace(visits=32,children={0:Edge(.001,16,0.),1:Edge(.009,16,.1)})
    allocator = SurvivorHalving(32,candidates=2); allocator.active=[0,1]; allocator.tail_started=32; allocator.assigned=32
    move,_ = allocator.choose(root)
    assert move == 1  # conditional0.9prior wins; unnormalized0.009would lose.
    root = SimpleNamespace(visits=0,children={0:Edge(.5,0,0.),1:Edge(.5,0,0.)})
    allocator = SurvivorHalving(32,candidates=2); allocator.capacity(root)
    for _ in range(32): allocator.choose(root)
    with pytest.raises(ValueError,match='virtual credits'): allocator.capacity(root)

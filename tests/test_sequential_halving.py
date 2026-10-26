from types import SimpleNamespace
import math
import pytest
import torch
from research.sequential_halving import RootHalving, HalvingPUCT
from research.policy_value import PhasePolicyValueNet
from adapters.yuan_observer import reconstruct


class Edge:
    def __init__(self, prior): self.prior = prior; self.visits = 0; self.value_sum = 0.
    def q_value(self): return self.value_sum/self.visits if self.visits else 0.


@pytest.mark.parametrize('budget,actions', [(256, 16), (33, 7), (1, 30), (8, 30), (37, 1)])
def test_exact_budget_round_balance_feedback_and_correct_q_perspective(budget, actions):
    root = SimpleNamespace(children={m: Edge(1/actions) for m in range(actions)}, visits=0)
    allocator = RootHalving(budget, candidates=16, seed=7)
    while root.visits < budget:
        capacity = allocator.capacity(root)
        assigned = []
        for _ in range(min(8, capacity)):
            move, edge = allocator.choose(root); assigned.append((move, edge)); edge.visits += 1
        for move, edge in assigned:
            edge.value_sum -= move/max(actions, 1); root.visits += 1
    recommendation = allocator.recommend(root)
    assert sum(c.visits for c in root.children.values()) == budget
    assert root.children[recommendation].visits > 0
    if (budget, actions) == (256, 16):
        assert [len(t['active']) for t in allocator.trace] == [16, 8, 4, 2]
        assert [t['cost'] for t in allocator.trace] == [64]*4
        assert recommendation == 15


def test_refuse_halving_with_outstanding_virtual_credits():
    root = SimpleNamespace(children={m: Edge(.5) for m in range(2)}, visits=0)
    allocator = RootHalving(2); assert allocator.capacity(root) == 2
    allocator.choose(root); allocator.choose(root)
    with pytest.raises(ValueError, match='virtual credits'): allocator.capacity(root)
    with pytest.raises(ValueError): allocator.recommend(root)


def test_real_particle_search_legal_visits_complete_rounds_and_private_isolation():
    torch.set_num_threads(1); torch.manual_seed(7)
    net = PhasePolicyValueNet(4, 1).eval(); outputs = []
    for poison in (False, True):
        state, belief, _ = reconstruct(['b2b9R'])
        if poison:
            belief.pool_counts[1-state.side] = [[0]*8, [0]*8]
            belief.pool_total[1-state.side] = [0, 0]
            belief.unknown_removals[1-state.side] = [99, 99]
        engine = HalvingPUCT(net, seed=13, leaf_batch=8)
        policy, move, root = engine.search(state, belief, [], simulations=64, gumbel_seed=19,
                                         noise_scale=1., candidates=8, return_root=True)
        assert move in state.legal_moves() and root.visits == 64 and sum(policy) == pytest.approx(1.)
        outputs.append((policy, move, root.value_sum, [(m, c.visits, c.value_sum) for m, c in root.children.items()]))
    assert outputs[0] == outputs[1]


def test_noise_has_an_explicit_independent_random_stream():
    root = SimpleNamespace(children={m: Edge(1/8) for m in range(8)}, visits=0)
    a = RootHalving(32, candidates=4, seed=31, noise_scale=1.)
    b = RootHalving(32, candidates=4, seed=31, noise_scale=1.)
    a.capacity(root); b.capacity(root)
    assert a.active == b.active and a.log_prior_noise == b.log_prior_noise
    assert all(math.isfinite(s) for s in a.log_prior_noise.values())

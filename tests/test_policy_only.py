import torch
import pytest
from research.policy_value import AZ, PhasePolicyValueNet
from research.policy_only import GreedyPolicy
from adapters.yuan_observer import reconstruct


def test_policy_only_reuses_original_prior_mask_and_has_zero_actual_visits():
    torch.set_num_threads(1); torch.manual_seed(7)
    net = PhasePolicyValueNet(4, 1).eval(); state, belief, info = reconstruct(['b2b9R'])
    agent = GreedyPolicy(net, seed=13, leaf_batch=8)
    priors, _ = agent._evaluate(info, []); expected = AZ.PUCTNode(1., state.side)
    expected.expand(state.legal_moves(), priors)
    policy, move, root = agent.search(state, belief, [], simulations=1000000, return_root=True)
    assert root.visits == 0 and sum(c.visits for c in root.children.values()) == 0
    assert agent.leaf_evals_total == 0
    assert move == max(expected.children, key=lambda m: expected.children[m].prior)
    assert sum(policy) == pytest.approx(1.)
    for m, child in expected.children.items(): assert policy[AZ.move_to_index(m)] == child.prior


def test_actual_opponent_private_slots_do_not_change_greedy_policy():
    torch.set_num_threads(1); torch.manual_seed(11)
    net = PhasePolicyValueNet(4, 1).eval(); outputs = []
    for poison in (False, True):
        state, belief, _ = reconstruct(['b2b9R'])
        if poison:
            belief.pool_counts[1-state.side] = [[0]*8, [0]*8]
            belief.pool_total[1-state.side] = [0, 0]
            belief.unknown_removals[1-state.side] = [99, 99]
        outputs.append(GreedyPolicy(net).search(state, belief, []))
    assert outputs[0] == outputs[1]

import copy
import numpy as np
import torch
import pytest
from research.private_inventory import PrivateInventory
from research.private_inventory_policy import counterfactual_records, action_likelihoods
from research.policy_value import AZ, PhasePolicyValueNet
from adapters.yuan_observer import reconstruct


def test_counterfactual_input_ignores_root_private_enemy_inventory():
    # BLACK previously lost an own dark piece; RED now acts. Construct the
    # observer view with the same public facts but deliberately different enemy
    # private ledgers. Neither is an input to the opponent model.
    state, belief, _ = reconstruct(['b2b9R', 'a6a5p'], viewer=AZ.jq.BLACK)
    info = belief.info_state(state, AZ.jq.BLACK)
    b = PrivateInventory(tuple(info.public_pool_known[AZ.jq.BLACK][t] for t in AZ.jq.POOL_TYPES), 1)
    first = counterfactual_records(info, AZ.jq.BLACK, b)
    changed = copy.deepcopy(info)
    changed.pool_known[AZ.jq.RED] = [99]*8
    changed.unknown_removals = [999, 999]
    second = counterfactual_records(changed, AZ.jq.BLACK, b)
    assert first[0] == second[0] and np.array_equal(first[1], second[1])
    torch.set_num_threads(1); torch.manual_seed(5)
    net = PhasePolicyValueNet(4, 1).eval()
    legal = state.legal_moves()
    all_likelihoods = [action_likelihoods(net, info, AZ.jq.BLACK, b, legal, m) for m in legal]
    for h in b.probabilities:
        assert sum(p[h] for p in all_likelihoods) == pytest.approx(1.)
        assert min(p[h] for p in all_likelihoods) >= .1/len(legal)

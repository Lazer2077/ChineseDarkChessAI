import copy
import random
import torch
import pytest
from research.private_inventory import PrivateInventory
from research.private_inventory_search import InventoryRootBelief, jq, im
from research.neural_search import ObservationPUCT
from research.policy_value import PhasePolicyValueNet
from adapters.yuan_observer import reconstruct


def inventory(info, side):
    return PrivateInventory(tuple(info.pool_known[side][t] for t in jq.POOL_TYPES), info.unknown_removals[side])


def summary(result):
    policy, move, root = result
    return policy, move, root.visits, root.value_sum, [(m, c.visits, c.value_sum) for m, c in root.children.items()]


def test_uniform_prior_preserves_original_rng_world_and_fixed_simulation_search():
    torch.set_num_threads(1); torch.manual_seed(7)
    net = PhasePolicyValueNet(4, 1).eval()
    outputs = []
    for bridge in (False, True):
        state, belief, info = reconstruct(['b2b9R'])
        wrapped = InventoryRootBelief(belief, state, state.side, inventory(info, state.side))
        r1, r2 = random.Random(123), random.Random(123)
        a, b = info.determinize(r1), wrapped.info.determinize(r2)
        assert a.pool == b.pool and r1.getstate() == r2.getstate()
        engine = ObservationPUCT(net, seed=53, leaf_batch=2)
        outputs.append(summary(engine.search(state, wrapped if bridge else belief, [], simulations=8, temperature=1e-6, return_root=True)))
    assert outputs[0] == outputs[1]


def test_informative_sampling_preserves_root_view_all_history_and_private_isolation():
    state, belief, info = reconstruct(['b2b9R'])
    b = inventory(info, state.side)
    preferred = next(iter(b.probabilities))
    b.condition_action({h: float(h == preferred) for h in b.probabilities})
    before = copy.deepcopy(info)
    other = 1-state.side
    belief.pool_counts[other] = [[0]*8, [0]*8]
    belief.pool_total[other] = [0, 0]
    belief.unknown_removals[other] = [99, 99]
    wrapped = InventoryRootBelief(belief, state, state.side, b)
    assert im.information_set_key(info, (), ()) == im.information_set_key(wrapped.info, (), ())
    assert info == before
    for seed in range(20):
        world = wrapped.info.determinize(random.Random(seed))
        assert tuple(world.pool[state.side][t] for t in jq.POOL_TYPES) == tuple(k-n for k, n in zip(b.public, preferred))
        for field in ('board', 'no_capture_plies', 'consecutive_checks', 'check_anchor', 'chase_streak', 'chase_target', 'chase_target_code', 'chase_anchor', 'position_counts'):
            assert getattr(world, field) == getattr(info, field)
        assert world.rules == info.rules
        sim = im.BeliefModel.for_simulation(wrapped.info, state.side, world)
        their = sim.info_state(world, other)
        assert their.pool_known[state.side] == world.pool[state.side]
    assert info == before
    with pytest.raises(ValueError): wrapped.info_state(state, other)


def test_informative_search_never_uses_poisoned_opponent_ledger():
    torch.set_num_threads(1); torch.manual_seed(11)
    net = PhasePolicyValueNet(4, 1).eval(); outputs = []
    for poison in (False, True):
        state, belief, info = reconstruct(['b2b9R'])
        b = inventory(info, state.side)
        b.condition_action({h: .1+.1*h[0] for h in b.probabilities})
        if poison:
            belief.pool_counts[1-state.side] = [[0]*8, [0]*8]
            belief.pool_total[1-state.side] = [0, 0]
            belief.unknown_removals[1-state.side] = [99, 99]
        wrapped = InventoryRootBelief(belief, state, state.side, b)
        engine = ObservationPUCT(net, seed=53, leaf_batch=2)
        outputs.append(summary(engine.search(state, wrapped, [], simulations=8, temperature=1e-6, return_root=True)))
    assert outputs[0] == outputs[1]

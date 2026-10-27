import copy
import torch
import pytest
from adapters.yuan_observer import reconstruct
from research.policy_value import PhasePolicyValueNet
from research.neural_search import ObservationPUCT
from research.factorial_network import FactorialNetwork


def test_complete_function_outputs_and_no_parameter_or_bn_transplant():
    torch.set_num_threads(1); torch.manual_seed(114)
    policy = PhasePolicyValueNet(4, 1).eval()
    value = PhasePolicyValueNet(4, 1).eval()
    before = [copy.deepcopy(n.state_dict()) for n in (policy, value)]
    calls = [0, 0]
    def count(i):
        def hook(*args): calls[i] += 1
        return hook
    policy.register_forward_hook(count(0)); value.register_forward_hook(count(1))
    x = torch.randn(3, 238, 10, 9)
    with torch.inference_mode():
        expected_p, _ = policy(x); _, expected_v = value(x)
        p, v = FactorialNetwork(policy, value).eval()(x)
    assert torch.equal(p, expected_p) and torch.equal(v, expected_v)
    assert calls == [2, 2]
    for net, saved in zip((policy, value), before):
        assert all(torch.equal(net.state_dict()[k], val) for k, val in saved.items())


def test_identical_checkpoints_preserve_search_and_private_isolation():
    torch.set_num_threads(1); torch.manual_seed(115)
    net = PhasePolicyValueNet(4, 1).eval(); outputs = []
    for dual, poison in ((False, False), (True, False), (True, True)):
        state, belief, _ = reconstruct(['b2b9R'])
        if poison:
            belief.pool_counts[1-state.side] = [[0]*8, [0]*8]
            belief.pool_total[1-state.side] = [0, 0]
            belief.unknown_removals[1-state.side] = [99, 99]
        model = FactorialNetwork(copy.deepcopy(net), copy.deepcopy(net)) if dual else net
        engine = ObservationPUCT(model, seed=117, leaf_batch=2)
        pi, move, root = engine.search(state, belief, [], simulations=16,
                                      temperature=1e-6, return_root=True)
        assert move in state.legal_moves() and root.visits == 16
        assert sum(pi) == pytest.approx(1.)
        outputs.append((pi, move, root.value_sum,
                        [(m, c.visits, c.value_sum) for m, c in root.children.items()]))
    assert outputs[0] == outputs[1] == outputs[2]

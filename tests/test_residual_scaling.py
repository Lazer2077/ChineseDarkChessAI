import numpy as np
import pytest
import torch
from research.policy_value import PhasePolicyValueNet, export_npz, load_npz
from research.residual_scaling import double_width


def test_initial_function_and_batchnorm_state_are_preserved(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(73)
    small = PhasePolicyValueNet(4, 2)
    with torch.no_grad():
        for module in small.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.normal_(0, .2)
                module.running_var.uniform_(.5, 1.5)
                module.num_batches_tracked.fill_(17)
    small.eval()
    before = {k: v.clone() for k, v in small.state_dict().items()}
    rng = torch.get_rng_state().clone()
    wide = double_width(small)
    assert torch.equal(rng, torch.get_rng_state())
    x = torch.randn(5, 238, 10, 9)
    for train in (False, True):
        small.train(train); wide.train(train)
        for a, b in zip(small(x), wide(x)):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    # Source weights are not touched by expansion (BN changes above are forward updates).
    for k, v in before.items():
        if not any(t in k for t in ('running_', 'num_batches_tracked')):
            assert torch.equal(v, small.state_dict()[k])
    export_npz(wide, tmp_path/'wide.npz')
    loaded = PhasePolicyValueNet(8, 2)
    load_npz(loaded, tmp_path/'wide.npz')
    for k, v in wide.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])


def test_gradient_symmetry_is_broken_and_seed_is_repeatable():
    torch.manual_seed(2)
    small = PhasePolicyValueNet(4, 1).eval()
    wide = double_width(small, seed=3).eval()
    other = double_width(small, seed=3)
    assert all(torch.equal(v, other.state_dict()[k]) for k,v in wide.state_dict().items())
    x = torch.randn(3, 238, 10, 9)
    policy, value = wide(x)
    (policy[:, 1].sum() + value.sum()).backward()
    grad = wide.conv.weight.grad
    assert (grad[:4] - grad[4:]).abs().max() > 1e-7


def test_rejects_unplanned_architecture_or_noise():
    with pytest.raises(ValueError):
        double_width(PhasePolicyValueNet(4, 1, True))
    with pytest.raises(ValueError):
        double_width(PhasePolicyValueNet(4, 1), noise_fraction=float('nan'))

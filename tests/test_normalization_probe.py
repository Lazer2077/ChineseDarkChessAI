import pytest
import torch
from research.policy_value import PhasePolicyValueNet
from research.normalization_probe import combine


def test_only_running_buffers_move_affine_parameters_stay_with_weights():
    torch.set_num_threads(1);torch.manual_seed(11)
    warm=PhasePolicyValueNet(4,1).eval();adapted=PhasePolicyValueNet(4,1).eval()
    with torch.no_grad():
        for mod in adapted.modules():
            if isinstance(mod,torch.nn.BatchNorm2d):
                mod.running_mean.fill_(.3);mod.running_var.fill_(2.);mod.num_batches_tracked.fill_(99)
                mod.weight.fill_(1.7);mod.bias.fill_(.4)
    rng=torch.get_rng_state().clone();mixed,keys=combine(warm,adapted)
    assert torch.equal(rng,torch.get_rng_state()) and len(keys)>0
    for name,value in mixed.state_dict().items():
        expected=adapted.state_dict()[name] if name in keys else warm.state_dict()[name]
        assert torch.equal(value,expected)
    for name,param in mixed.named_parameters():assert torch.equal(param,warm.state_dict()[name])
    x=torch.randn(2,238,10,9)
    identical,_=combine(warm,warm)
    assert all(torch.equal(a,b) for a,b in zip(warm(x),identical(x)))
    assert not torch.equal(warm(x)[0],mixed(x)[0])


def test_cross_architecture_buffers_are_rejected():
    with pytest.raises(ValueError):combine(PhasePolicyValueNet(4,1),PhasePolicyValueNet(8,1))

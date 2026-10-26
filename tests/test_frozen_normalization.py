import torch
from research.policy_value import PhasePolicyValueNet
from research.frozen_normalization import freeze_statistics,running_buffers


def test_statistics_stay_fixed_but_affine_and_other_weights_can_learn():
    torch.set_num_threads(1);torch.manual_seed(21)
    net=PhasePolicyValueNet(4,1).eval();x=torch.randn(3,238,10,9)
    before=net(x);buffers=running_buffers(net)
    net=freeze_statistics(net);net.train()
    assert net.training and all(not m.training for m in net.modules() if isinstance(m,torch.nn.BatchNorm2d))
    assert all(torch.equal(a,b) for a,b in zip(before,net(x)))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    saved=net.policy_fc.weight.detach().clone();logits,value=net(x)
    loss=logits[:,0].sum()+value.sum();loss.backward()
    assert net.bn.weight.grad is not None and net.value_bn.bias.grad is not None
    optimizer.step();net.eval();net.check_statistics()
    assert not torch.equal(saved,net.policy_fc.weight)
    assert all(torch.equal(v,running_buffers(net)[k]) for k,v in buffers.items())
    other=PhasePolicyValueNet(4,1).train()
    assert other.bn.training  # No global class/training-mode mutation.

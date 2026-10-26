"""Inference-only factorial intervention on learned weights and BN buffers."""
import torch
from research.policy_value import PhasePolicyValueNet


def combine(weights,statistics):
    shape=lambda n:(n.conv.out_channels,len(n.residual),n.phase_experts)
    if shape(weights)!=shape(statistics):raise ValueError('Require identical architectures')
    keys=set()
    for name,module in weights.named_modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            keys.update(name+'.'+suffix for suffix in ['running_mean','running_var','num_batches_tracked'])
    left=weights.state_dict();right=statistics.state_dict()
    if left.keys()!=right.keys() or any(left[k].shape!=right[k].shape for k in left):
        raise ValueError('State structures differ')
    # Keep affine gamma/beta with the learned weights, never with the buffers.
    state={k:(right[k] if k in keys else left[k]).detach().cpu().clone() for k in left}
    with torch.random.fork_rng(devices=[]):
        net=PhasePolicyValueNet(*shape(weights))
    net.load_state_dict(state);net.eval()
    for key,value in net.state_dict().items():
        expected=right[key] if key in keys else left[key]
        if not torch.equal(value,expected.detach().cpu()):raise ValueError('Unexpected state change')
    return net,sorted(keys)

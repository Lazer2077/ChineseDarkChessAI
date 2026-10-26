"""Train weights/BN affine parameters while retaining warm-start running buffers."""
import torch
from research.policy_value import PhasePolicyValueNet


def running_buffers(net):
    state=net.state_dict();keys=[]
    for name,module in net.named_modules():
        if isinstance(module,torch.nn.BatchNorm2d):
            keys.extend(name+'.'+suffix for suffix in ['running_mean','running_var','num_batches_tracked'])
    return {k:state[k].detach().cpu().clone() for k in keys}


class FrozenStatisticsNet(PhasePolicyValueNet):
    def check_statistics(self):
        current=running_buffers(self)
        if current.keys()!=self._statistics_reference.keys() or any(
            not torch.equal(value,self._statistics_reference[key]) for key,value in current.items()):
            raise ValueError('Frozen normalization buffers changed')

    def train(self,mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,torch.nn.BatchNorm2d):module.eval()
        self.check_statistics()
        return self


def freeze_statistics(net):
    if type(net) is not PhasePolicyValueNet:raise ValueError('Require the known shared/phase architecture')
    reference=running_buffers(net)
    # Only this model instance changes type; no original class or module is patched.
    net.__class__=FrozenStatisticsNet
    net._statistics_reference=reference
    net.train(net.training)
    return net

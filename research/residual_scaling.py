"""Function-preserving width doubling for the pinned residual policy/value net.

Net2Wider idea: Chen, Goodfellow & Shlens, ICLR 2016,
https://arxiv.org/abs/1511.05641 . Residual channels use a common duplication map
so skip connections are preserved. This is an inherited initialization method,
not a new Jieqi algorithm or evidence that scaling improves strength.
"""
import math
import torch
from research.policy_value import PhasePolicyValueNet


def double_width(source, seed=20261030, noise_fraction=0.001):
    """Copy BN state and split each downstream input weight across two copies.

    Seeded opposite perturbations on the two input copies sum to zero. They
    preserve the initial function (up to FP rounding) while allowing replicated
    channels to acquire different gradients. Does not modify the source model.
    """
    if not math.isfinite(noise_fraction) or noise_fraction < 0:
        raise ValueError('Require finite nonnegative perturbation')
    if source.phase_experts:
        raise ValueError('This controlled experiment uses the shared value head')
    channels = source.conv.out_channels
    # Constructor initialization is overwritten, and must not perturb caller RNG.
    with torch.random.fork_rng(devices=[]):
        target = PhasePolicyValueNet(channels * 2, len(source.residual), False)
    generator = torch.Generator().manual_seed(seed)
    original = {k: v.detach().cpu() for k, v in source.state_dict().items()}
    copied = {}
    for name, expected in target.state_dict().items():
        weight = original[name]
        if name == 'conv.weight':
            value = weight.repeat(2, 1, 1, 1)
        elif name.startswith('bn.') or (name.startswith('residual.') and '.bn' in name):
            value = weight.repeat(2) if weight.ndim else weight.clone()
        elif name.startswith('residual.') and '.conv' in name:
            value = weight.repeat(2, 2, 1, 1) / 2
        elif name in ('policy_conv.weight', 'value_conv.weight'):
            value = weight.repeat(1, 2, 1, 1) / 2
        else:
            value = weight.clone()
        if weight.ndim == 4 and name != 'conv.weight' and value.shape[1] == 2 * weight.shape[1]:
            # Independent output-channel perturbations break duplicated gradients.
            delta = torch.randn(value[:, :channels].shape, generator=generator,
                                dtype=value.dtype) * (weight.square().mean().sqrt() * noise_fraction)
            value[:, :channels] += delta
            value[:, channels:] -= delta
        if value.shape != expected.shape:
            raise ValueError('Unhandled shape transformation: ' + name)
        copied[name] = value
    target.load_state_dict(copied)
    target.train(source.training)
    return target

"""Training-only conditional value model with twelve extra dark-identity planes."""
import copy
import torch
import torch.nn.functional as F
from research.policy_value import AZ, encode_records

CONTRACT = 'cdca-training-only-observer-plus-live-dark-value-v1'


def encode_value_inputs(raw, truth, privileged):
    if truth.dtype != torch.uint8 or truth.shape != (len(raw), 90):
        raise ValueError('Require uint8 live-dark labels in Yuan square order')
    x = encode_records(raw)
    board = raw[:, :90].long().reshape(-1, 10, 9).flip(1).reshape(-1, 90)
    dark = (board & 15) == 8
    valid = torch.zeros_like(dark)
    for t in AZ.jq.POOL_TYPES: valid |= truth == t
    if not torch.equal(truth > 0, dark) or not bool((valid | ~dark).all()):
        raise ValueError('Identity label/public dark geometry mismatch')
    extra = torch.zeros((len(raw), 12, 90), device=raw.device)
    if privileged:
        colors = (board >> 4) ^ 1
        for color in (0, 1):
            for slot, t in enumerate(AZ.jq.POOL_TYPES):
                extra[:, color*6+slot] = ((truth == t) & (colors == color)).float()
    return torch.cat((x, extra.reshape(-1, 12, 10, 9)), dim=1)


class PrivilegedValueNet(torch.nn.Module):
    def __init__(self, warm):
        super().__init__()
        if warm.phase_experts: raise ValueError('Require the frozen shared warm architecture')
        self.network = copy.deepcopy(warm)
        old = self.network.conv
        conv = torch.nn.Conv2d(old.in_channels+12, old.out_channels,
                               kernel_size=3, padding=1, bias=False).to(old.weight.device)
        with torch.no_grad():
            conv.weight.zero_(); conv.weight[:, :old.in_channels].copy_(old.weight)
        self.network.conv = conv
        for name, parameter in self.network.named_parameters():
            if name.startswith('policy_'): parameter.requires_grad_(False)

    def forward(self, x):
        n = self.network
        trunk = n.residual(F.relu(n.bn(n.conv(x))))
        features = F.relu(n.value_bn(n.value_conv(trunk))).flatten(1)
        return torch.tanh(n.value_fc2(F.relu(n.value_fc1(features)))).squeeze(-1)

"""Relative-action convolutional policy decoder; original observation/value net.

The decoder follows the established convolution-and-gather design used in
Lc0, adapted independently to the pinned2550Jieqi move geometry. It does not
assert translation invariance of palace/river/dark-home rules or expose types.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from research.policy_value import AZ,CONTRACT,PhasePolicyValueNet,load_npz

DISPLACEMENTS=tuple(sorted({(t//9-f//9,t%9-f%9) for f,t in AZ.LEGAL_MOVE_TABLE}))
DISPLACEMENT_INDEX={delta:i for i,delta in enumerate(DISPLACEMENTS)}
POLICY_GATHER=np.asarray([DISPLACEMENT_INDEX[(t//9-f//9,t%9-f%9)]*90+f for f,t in AZ.LEGAL_MOVE_TABLE],dtype=np.int64)
if len(DISPLACEMENTS)!=50 or len(set(POLICY_GATHER))!=AZ.MOVE_SIZE:raise ValueError('Pinned relative action geometry changed')
POLICY_MAPPING_SHA256=hashlib.sha256(POLICY_GATHER.astype('<i8').tobytes()).hexdigest()


def gather_policy(grid):
    if grid.ndim!=4 or tuple(grid.shape[1:])!=(50,10,9):raise ValueError('Require50relative planes on the10x9board')
    return grid.flatten(1).index_select(1,torch.as_tensor(POLICY_GATHER,device=grid.device))


class RelativePolicyValueNet(PhasePolicyValueNet):
    def __init__(self,channels=64,blocks=15,phase_experts=False):
        # Construct the old head first so a shared seed initializes ALL common
        # trunk/value/policy-conv parameters identically to the dense control.
        super().__init__(channels,blocks,phase_experts)
        del self.policy_fc
        self.policy_spatial=torch.nn.Conv2d(32,50,kernel_size=1,bias=True)
        self.register_buffer('policy_gather',torch.from_numpy(POLICY_GATHER.copy()),persistent=False)

    def forward(self,x):
        trunk=self.residual(F.relu(self.bn(self.conv(x))))
        features=F.relu(self.policy_bn(self.policy_conv(trunk)))
        grid=self.policy_spatial(features)
        policy=grid.flatten(1).index_select(1,self.policy_gather)
        values=F.relu(self.value_bn(self.value_conv(trunk))).flatten(1)
        value=torch.tanh(self.value_fc2(F.relu(self.value_fc1(values)))).squeeze(-1)
        if self.phase_experts:
            public=torch.tanh(self.public_fc2(F.relu(self.public_fc1(values)))).squeeze(-1)
            value=torch.where(x[:,14:28].sum(dim=(1,2,3))==0,public,value)
        return policy,value


def make_model(policy_head,channels=64,blocks=15,phase_experts=False):
    if policy_head=='dense2550':return PhasePolicyValueNet(channels,blocks,phase_experts)
    if policy_head=='relative50':return RelativePolicyValueNet(channels,blocks,phase_experts)
    raise ValueError('Unknown controlled policy decoder')


def load_model(directory,device='cpu'):
    directory=Path(directory);metadata=json.loads((directory/'best.json').read_text());file=directory/'best.npz'
    if metadata['input_contract']!=CONTRACT or hashlib.sha256(file.read_bytes()).hexdigest()!=metadata['npz_sha256']:
        raise ValueError('Policy decoder model contract or digest differs')
    head=metadata.get('policy_head','dense2550')
    if head=='relative50' and metadata.get('policy_mapping_sha256')!=POLICY_MAPPING_SHA256:
        raise ValueError('Relative policy geometry digest differs')
    net=make_model(head,metadata['channels'],metadata['blocks'],metadata['phase_experts'])
    load_npz(net,file);net.to(device);net.eval()
    return net,metadata

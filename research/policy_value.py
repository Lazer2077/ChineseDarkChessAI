"""Policy/value warm start with a conservative, publicly observable phase gate.

Residual architecture and 2550-action geometry are attributed to pinned MIT
Yuan-Bai/jieqi-lab. Our input adapter deliberately omits unavailable history
features from both replay and inference; search must still retain rule history.
"""
import copy,hashlib,importlib.util,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external/yuan-bai-jieqi-lab'
REC=np.dtype([('board','u1',(90,)),('stm','u1'),('urn','i1',(2,6)),('score','<i2'),('result','i1'),('flags','u1'),('ply','<u2'),('move','<u2'),('nocap','u1')])
assert REC.itemsize==112
CONTRACT='cdca-board-private-inventory-nocap-canonical-fullpublic-v1'

@lru_cache(None)
def upstream():
    sys.path.insert(0,str(ROOT/'adapters'))
    from yuan_observer import load_sources
    load_sources()
    if hashlib.sha256((SOURCE/'azero.py').read_bytes()).hexdigest()!='4a184dd7ea1d93a8fe2328ace3439ddbe3f78c200cd8dd50ccd610558f1223f2':
        raise ValueError('Pinned neural architecture changed')
    sys.path.insert(0,str(SOURCE))
    spec=importlib.util.spec_from_file_location('cdca_yuan_azero',SOURCE/'azero.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

AZ=upstream()
ACTION_NATIVE=np.array([((9-f//9)*9+f%9)|(((9-t//9)*9+t%9)<<7) for f,t in AZ.LEGAL_MOVE_TABLE],dtype=np.int64)
ACTION_LOOKUP=np.full(128*128,-1,dtype=np.int64);ACTION_LOOKUP[ACTION_NATIVE]=np.arange(AZ.MOVE_SIZE)
MIRROR_ACTION=np.array([AZ.move_to_index(AZ.mirror_move(move)) for move in AZ.LEGAL_MOVE_TABLE],dtype=np.int64)
HOME_KIND=np.zeros(90,dtype=np.int64)
for index in range(90):
    try:HOME_KIND[index]=AZ.jq.fake_type_for_square(index)
    except ValueError:pass

def action_indices(records):
    moves=records['move'].astype(np.int64)
    ids=ACTION_LOOKUP[moves & 16383]
    if (ids<0).any():raise ValueError('A teacher action is outside the public move geometry')
    return ids

def encode_records(raw,canonical=True):
    """GPU-capable encoding; raw is (B,112) uint8, with no true hidden types."""
    if raw.ndim!=2 or raw.shape[1]!=112 or raw.dtype!=torch.uint8:raise ValueError('Require byte records')
    n=len(raw);dev=raw.device
    board=raw[:,:90].long().reshape(n,10,9).flip(1).reshape(n,90)
    types=board&15;colors=(board>>4)^1;dark=types==8
    homes=torch.as_tensor(HOME_KIND,device=dev)
    kinds=torch.where(dark,homes.unsqueeze(0),types)
    if ((dark&(kinds==0))|((types>8))).any():raise ValueError('Invalid dark home or piece code')
    x=torch.zeros((n,AZ.STATE_CHANNELS,90),device=dev)
    batch,square=torch.nonzero(types>0,as_tuple=True)
    planes=dark[batch,square].long()*14+colors[batch,square]*7+kinds[batch,square]-1
    x[batch,planes,square]=1
    counts=raw[:,91:103].to(torch.int8).float().reshape(n,2,6).flip(1)
    live=torch.stack([(dark&(colors==c)).sum(1) for c in [0,1]],dim=1).float()
    unknown=counts.sum(2)-live
    if (unknown<0).any():raise ValueError('Inventory cannot cover on-board dark pieces')
    if canonical:
        full=live.sum(1)==0
        counts=counts.masked_fill(full[:,None,None],0)
        unknown=unknown.masked_fill(full[:,None],0)
    constants=torch.cat([(raw[:,90]==0).float().unsqueeze(1),
        (counts/torch.tensor([2,2,2,2,2,5],device=dev)).reshape(n,12),
        live/15,unknown/15,raw[:,111:112].float()/80,
        torch.zeros((n,2),device=dev)],dim=1)
    x[:,28:48]=constants.unsqueeze(-1)
    # No recorded repetition/check/chase/recent-move history is fabricated.
    # Their planes and the upstream complete-history indicator remain zero.
    return x.reshape(n,AZ.STATE_CHANNELS,10,9)

def records_from_info(info):
    """Identical input contract for live search; no hidden identities are read."""
    rec=np.zeros(1,dtype=REC)
    for square,piece in enumerate(info.board):
        if not piece:continue
        native=(9-square//9)*9+square%9
        color=0 if AZ.jq.piece_color(piece)==AZ.jq.RED else 1
        kind=8 if AZ.jq.is_dark(piece) else AZ.jq.piece_type(piece)
        rec['board'][0,native]=kind|(color<<4)
    rec['stm'][0]=0 if info.side==AZ.jq.RED else 1
    for c,jq_c in enumerate([AZ.jq.RED,AZ.jq.BLACK]):
        rec['urn'][0,c]=[info.pool_known[jq_c][k] for k in AZ.jq.POOL_TYPES]
    rec['nocap'][0]=min(255,info.no_capture_plies)
    return rec.view(np.uint8).reshape(1,112)

class PhasePolicyValueNet(AZ.PolicyValueNet):
    def __init__(self,channels=64,blocks=15,phase_experts=False):
        super().__init__(channels,blocks);self.phase_experts=phase_experts
        if phase_experts:
            self.public_fc1=copy.deepcopy(self.value_fc1)
            self.public_fc2=copy.deepcopy(self.value_fc2)

    def forward(self,x):
        trunk=self.residual(F.relu(self.bn(self.conv(x))))
        policy=F.relu(self.policy_bn(self.policy_conv(trunk))).flatten(1)
        policy=self.policy_fc(policy)
        features=F.relu(self.value_bn(self.value_conv(trunk))).flatten(1)
        value=torch.tanh(self.value_fc2(F.relu(self.value_fc1(features)))).squeeze(-1)
        if self.phase_experts:
            public=torch.tanh(self.public_fc2(F.relu(self.public_fc1(features)))).squeeze(-1)
            full=x[:,14:28].sum(dim=(1,2,3))==0
            value=torch.where(full,public,value)
        return policy,value

def export_npz(net,path):
    """Array-only interchange; never load Python objects from inference artifacts."""
    with Path(path).open('wb') as handle:
        np.savez_compressed(handle,**{name:t.detach().cpu().numpy() for name,t in net.state_dict().items()})

def load_npz(net,path,common_initial=False):
    expected=net.state_dict();state={}
    with np.load(path,allow_pickle=False) as arrays:
        if not set(arrays.files)<=set(expected):raise ValueError('Unexpected parameter name')
        for name in arrays.files:
            arr=arrays[name]
            if arr.shape!=tuple(expected[name].shape) or arr.dtype!=expected[name].cpu().numpy().dtype or not np.isfinite(arr).all():
                raise ValueError('Invalid parameter shape, dtype or finite values')
            state[name]=torch.from_numpy(arr.copy())
    missing=set(expected)-set(state)
    public={name for name in expected if name.startswith(('public_fc1.','public_fc2.'))}
    if missing and (not common_initial or missing!=public):raise ValueError('Missing parameters')
    net.load_state_dict(state,strict=not missing)
    if missing:
        net.public_fc1.load_state_dict(net.value_fc1.state_dict())
        net.public_fc2.load_state_dict(net.value_fc2.state_dict())

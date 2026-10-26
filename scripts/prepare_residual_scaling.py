"""Prepare and numerically certify a warm-start width control, before training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.policy_value import REC, PhasePolicyValueNet, load_npz, export_npz, encode_records
from research.residual_scaling import double_width


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial',required=True)
    p.add_argument('--records',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists(): raise ValueError('Refuse to overwrite preparation')
    torch.set_num_threads(1)
    small=PhasePolicyValueNet(64,15).eval();load_npz(small,a.initial)
    wide=double_width(small,seed=20261030,noise_fraction=.001).eval()
    records=np.fromfile(a.records,dtype=REC)
    full=~((records['board'] & 15)==8).any(1)
    ids=np.concatenate([np.flatnonzero(~full)[:128],np.flatnonzero(full)[:128]])
    if len(ids)!=256: raise ValueError('Need 128 partial and 128 full-public inputs')
    raw=records[ids].view(np.uint8).reshape(-1,112)
    maximum=np.zeros(2)
    with torch.inference_mode():
        for start in range(0,len(raw),16):
            x=encode_records(torch.from_numpy(raw[start:start+16]))
            original=small(x);expanded=wide(x)
            for j,(u,v) in enumerate(zip(original,expanded)):
                maximum[j]=max(maximum[j],float((u-v).abs().max()))
                torch.testing.assert_close(u,v,rtol=2e-4,atol=2e-4)
    out.mkdir(parents=True)
    export_npz(small,out/'small-initial.npz');export_npz(wide,out/'wide-initial.npz')
    report={'schema':1,'status':'completed','scope':'Initialization equivalence only; no training or strength result',
        'method':'Net2Wider residual channel duplication, split downstream weights, antisymmetric seeded perturbation',
        'reference':'https://arxiv.org/abs/1511.05641','seed':20261030,'noise_fraction':.001,
        'precision':'CPU FP32','positions':256,'partial_positions':128,'full_public_positions':128,
        'max_policy_logit_difference':float(maximum[0]),'max_value_difference':float(maximum[1]),
        'assert_close':{'rtol':.0002,'atol':.0002},'optimizer_state':'fresh for both arms; no optimizer transfer',
        'small_parameters':sum(p.numel() for p in small.parameters()),
        'wide_parameters':sum(p.numel() for p in wide.parameters()),
        'sha256':{a.initial:sha(a.initial),a.records:sha(a.records),
            'research/residual_scaling.py':sha(ROOT/'research/residual_scaling.py'),
            'scripts/prepare_residual_scaling.py':sha(__file__),
            str(out/'small-initial.npz'):sha(out/'small-initial.npz'),
            str(out/'wide-initial.npz'):sha(out/'wide-initial.npz')}}
    (out/'initialization-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

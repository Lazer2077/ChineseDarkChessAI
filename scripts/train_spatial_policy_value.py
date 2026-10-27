"""Extend the pinned full-replay trainer with a controlled policy decoder."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_policy_value as original
from research.neural_replay import sha
from research.spatial_policy_value import make_model,POLICY_MAPPING_SHA256


def install():
    if sha(ROOT/'scripts/train_policy_value.py')!='34f8bec1d2d252fe7d243e250cdf832c1b04adecb2b69ece040794c2cffdb21f':
        raise ValueError('Pinned full-replay trainer changed')
    source=textwrap.dedent(inspect.getsource(original.main))
    def replace(needle,value,count=1):
        nonlocal source
        if source.count(needle)!=count:raise ValueError('Policy decoder trainer patch site changed')
        source=source.replace(needle,value)
    replace("    p.add_argument('--plan',required=True);a=p.parse_args()",
        "    p.add_argument('--policy-head',choices=['dense2550','relative50'],required=True)\n    p.add_argument('--plan',required=True);a=p.parse_args()")
    replace("    plan_digest=sha(a.plan)","    plan_digest=sha(a.plan)\n    if a.policy_head!=plan['policy_head']:raise ValueError('Unplanned policy decoder')")
    replace('    net=PhasePolicyValueNet(a.channels,a.blocks,a.experts).to(dev);load_npz(net,a.initial,common_initial=True)',
            '    net=_make_model(a.policy_head,a.channels,a.blocks,a.experts).to(dev);load_npz(net,a.initial,common_initial=True)')
    replace("'phase_experts':a.experts", "'phase_experts':a.experts,'policy_head':a.policy_head,'policy_mapping_sha256':_mapping_sha",2)
    namespace=dict(original.__dict__,_make_model=make_model,_mapping_sha=POLICY_MAPPING_SHA256)
    exec(compile(source,'<cdca-spatial-policy-full-replay-training>','exec'),namespace)
    return namespace['main'],hashlib.sha256(source.encode()).hexdigest()


if __name__=='__main__':
    main,_=install();main()

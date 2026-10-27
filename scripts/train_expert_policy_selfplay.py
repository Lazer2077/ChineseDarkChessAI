"""Controlled policy-target substitution; original return targets are retained."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_neural_selfplay as original
from research.neural_replay import sha
from research.expert_policy_target import load_split,validate_split

ORIGINAL_SHA='0d8f6fdf9c5536ed8bf996a79a53e8d38b4224905a6f9a716b92fd1c61613e57'


def install():
    if sha(ROOT/'scripts/train_neural_selfplay.py')!=ORIGINAL_SHA:raise ValueError('Pinned original trainer changed')
    source=textwrap.dedent(inspect.getsource(original.main))
    def replace(needle,value):
        nonlocal source
        if source.count(needle)!=1:raise ValueError('Original policy-target patch site changed')
        source=source.replace(needle,value)
    replace("    if len(set(train_ids+val_ids))!=len(train_ids)+len(val_ids) or set(train_ids+val_ids)!=set(range(generation['games'])):\n        raise ValueError('Duplicate, missing or overlapping game split')",
            '    _validate_split(plan,generation,train_ids,val_ids)')
    replace('    raw,pi,z,train_meta=load_split(folder,train_ids,manifest)',
            '    raw,pi,z,train_meta=_expert_load_split(plan,folder,train_ids,manifest)')
    replace('    vraw,vpi,vz,val_meta=load_split(folder,val_ids,manifest)',
            '    vraw,vpi,vz,val_meta=_expert_load_split(plan,folder,val_ids,manifest)')
    replace("'scope':'Search-guided neural self-play RL update using actual root visits and terminal values. Tiny development pilot; no teacher-strength gain, equilibrium guarantee or model promotion.'",
            "'scope':'Controlled observer-only policy-target substitution on identical accepted learner games; unchanged learner terminal-return targets. No policy-gradient, DAgger theorem, native terminal-value or strength guarantee.', 'policy_target':plan['policy_target'], 'checkpoint_selection':'fixed-final-epoch', 'annotation_report_sha256':sha(ROOT/plan['annotation_report'])")
    # Both arms export the last prespecified epoch. Profile-specific selection
    # losses are logged but cannot select different training durations.
    replace("            if metrics['loss']<best:","            if epoch==plan['epochs']-1:")
    namespace=dict(original.__dict__,_validate_split=validate_split,_expert_load_split=load_split)
    exec(compile(source,'<cdca-expert-policy-target-training>','exec'),namespace)
    return namespace['main'],hashlib.sha256(source.encode()).hexdigest()


if __name__=='__main__':
    main,_=install();main()

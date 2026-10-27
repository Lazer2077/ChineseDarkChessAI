"""Fixed policy/value target factorial with unchanged network and optimizer."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_neural_selfplay as original
from research.neural_replay import sha
from research.expert_policy_target import validate_split
from research.expert_joint_target import load_split

ORIGINAL_SHA='0d8f6fdf9c5536ed8bf996a79a53e8d38b4224905a6f9a716b92fd1c61613e57'


def install():
    if sha(ROOT/'scripts/train_neural_selfplay.py')!=ORIGINAL_SHA:raise ValueError('Pinned original trainer changed')
    source=textwrap.dedent(inspect.getsource(original.main))
    def replace(needle,value):
        nonlocal source
        if source.count(needle)!=1:raise ValueError('Joint-target patch site changed')
        source=source.replace(needle,value)
    replace("    if len(set(train_ids+val_ids))!=len(train_ids)+len(val_ids) or set(train_ids+val_ids)!=set(range(generation['games'])):\n        raise ValueError('Duplicate, missing or overlapping game split')",
            '    _validate_split(plan,generation,train_ids,val_ids)')
    replace('    raw,pi,z,train_meta=load_split(folder,train_ids,manifest)',
            '    raw,pi,z,train_meta=_joint_load_split(plan,folder,train_ids,manifest)')
    replace('    vraw,vpi,vz,val_meta=load_split(folder,val_ids,manifest)',
            '    vraw,vpi,vz,val_meta=_joint_load_split(plan,folder,val_ids,manifest)')
    replace("'scope':'Search-guided neural self-play RL update using actual root visits and terminal values. Tiny development pilot; no teacher-strength gain, equilibrium guarantee or model promotion.'",
            "'scope':'Controlled policy/value supervision factorial on identical accepted learner games. Observer-only expert score targets are uncalibrated search proxies, not true returns. No policy-gradient, asymmetric-critic, DAgger convergence, native terminal-value or strength guarantee.', 'policy_target':plan['policy_target'], 'value_target':plan['value_target'], 'value_score_scale':plan['value_score_scale'], 'checkpoint_selection':'fixed-final-epoch', 'annotation_report_sha256':sha(ROOT/plan['annotation_report'])")
    replace("            if metrics['loss']<best:","            if epoch==plan['epochs']-1:")
    replace("'terminal_value_mse':float(totals[1]/totals[2])","'value_target_mse':float(totals[1]/totals[2])")
    replace("'search_policy_ce':float(totals[0]/totals[2])","'policy_target_ce':float(totals[0]/totals[2])")
    replace("'train_terminal_mse':float(sums[1]/sums[2])","'train_value_target_mse':float(sums[1]/sums[2])")
    namespace=dict(original.__dict__,_validate_split=validate_split,_joint_load_split=load_split)
    exec(compile(source,'<cdca-joint-expert-target-training>','exec'),namespace)
    return namespace['main'],hashlib.sha256(source.encode()).hexdigest()


if __name__=='__main__':
    main,_=install();main()

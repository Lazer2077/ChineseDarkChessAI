"""Guarded extension of the existing anchor trainer; original files unchanged."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_anchored_selfplay as anchored
from research.entropy_policy_anchor import load_entropy_weights,weighted_legal_teacher_kl
from research.neural_replay import sha

ANCHOR_SHA='b72ccc656f4c357fcca32976daec5355f29bf0d894f64cac3e8a65338fc55cd9'


def install():
    if sha(ROOT/'scripts/train_anchored_selfplay.py')!=ANCHOR_SHA:
        raise ValueError('Pinned constant-anchor wrapper changed')
    # Retrieve its explicit generated source by changing only the installer return.
    # No bytecode, global function interception or on-disk source mutation occurs.
    factory=textwrap.dedent(inspect.getsource(anchored.install))
    needle="    return namespace['main'],hashlib.sha256(source.encode()).hexdigest()"
    if factory.count(needle)!=1:raise ValueError('Anchor installer return changed')
    factory=factory.replace(needle,'    return source,namespace')
    namespace=dict(anchored.__dict__)
    exec(compile(factory,'<cdca-anchor-source-factory>','exec'),namespace)
    source,environment=namespace['install']()
    def replace(needle,value):
        nonlocal source
        if source.count(needle)!=1:raise ValueError('Entropy patch site changed: '+needle)
        source=source.replace(needle,value)
    replace('    masks=_load_masks(plan,train_meta,raw,pi,dev)',
            '    masks=_load_masks(plan,train_meta,raw,pi,dev)\n    entropy_weights=_load_entropy_weights(plan,train_meta,raw,dev)')
    replace('anchor=_legal_teacher_kl(logits,teacher_logits,legal)',
            'anchor=_weighted_legal_teacher_kl(logits,teacher_logits,legal,entropy_weights[ids])')
    replace("'policy_anchor_lambda':anchor_lambda,",
            "'policy_anchor_lambda':anchor_lambda, 'entropy_weight_mode':plan['entropy_weight_mode'], 'entropy_weight_mean':float(entropy_weights.mean()), 'entropy_weight_min':float(entropy_weights.min()), 'entropy_weight_max':float(entropy_weights.max()),")
    replace("'train_legal_teacher_kl':anchor_sum/sums[2]",
            "'train_weighted_legal_teacher_kl':anchor_sum/sums[2]")
    environment.update(_load_entropy_weights=load_entropy_weights,_weighted_legal_teacher_kl=weighted_legal_teacher_kl)
    exec(compile(source,'<cdca-observable-entropy-anchor-training>','exec'),environment)
    return environment['main'],hashlib.sha256(source.encode()).hexdigest()


if __name__=='__main__':
    main,_=install();main()

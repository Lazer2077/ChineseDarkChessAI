"""Matched prefix/label controls with the original optimizer and fixed final epoch."""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import train_neural_selfplay as original
from research.neural_replay import sha
from research.native_terminal_targets import load_split, validate_split
from research.spatial_policy_value import load_model, POLICY_MAPPING_SHA256


def install():
    if sha(ROOT / 'scripts/train_neural_selfplay.py') != '0d8f6fdf9c5536ed8bf996a79a53e8d38b4224905a6f9a716b92fd1c61613e57':
        raise ValueError('Pinned original self-play trainer changed')
    source = textwrap.dedent(inspect.getsource(original.main))

    def replace(needle, value):
        nonlocal source
        if source.count(needle) != 1:
            raise ValueError('Native terminal target patch site changed')
        source = source.replace(needle, value)

    replace("    if len(set(train_ids+val_ids))!=len(train_ids)+len(val_ids) or set(train_ids+val_ids)!=set(range(generation['games'])):\n        raise ValueError('Duplicate, missing or overlapping game split')",
            '    _validate_split(plan,generation,train_ids,val_ids)')
    replace('    raw,pi,z,train_meta=load_split(folder,train_ids,manifest)',
            '    raw,pi,z,train_meta=_native_load_split(plan,folder,train_ids,manifest)')
    replace('    vraw,vpi,vz,val_meta=load_split(folder,val_ids,manifest)',
            '    vraw,vpi,vz,val_meta=_native_load_split(plan,folder,val_ids,manifest)')
    replace("'initial_network_sha256':initial['npz_sha256']",
            "'target_profile':plan['target_profile'],'policy_head':initial.get('policy_head','dense2550'),'policy_mapping_sha256':_mapping_sha,'initial_network_sha256':initial['npz_sha256']")
    replace("'scope':'Search-guided neural self-play RL update using actual root visits and terminal values. Tiny development pilot; no teacher-strength gain, equilibrium guarantee or model promotion.'",
            "'scope':plan['scope'],'checkpoint_selection':'fixed-final-epoch','native_terminal_annotations_sha256':plan['native_terminal_annotations_sha256']")
    replace("            if metrics['loss']<best:", "            if epoch==plan['epochs']-1:")
    replace("'phase_experts':initial['phase_experts'],'channels':initial['channels'],'blocks':initial['blocks']})",
            "'phase_experts':initial['phase_experts'],'channels':initial['channels'],'blocks':initial['blocks'],\n                    'policy_head':initial.get('policy_head','dense2550'),'policy_mapping_sha256':_mapping_sha})")
    namespace = dict(original.__dict__, load_model=load_model, _validate_split=validate_split,
                     _native_load_split=load_split, _mapping_sha=POLICY_MAPPING_SHA256)
    exec(compile(source, '<cdca-native-terminal-target-training>', 'exec'), namespace)
    return namespace['main'], hashlib.sha256(source.encode()).hexdigest()


if __name__ == '__main__':
    main, _ = install()
    main()

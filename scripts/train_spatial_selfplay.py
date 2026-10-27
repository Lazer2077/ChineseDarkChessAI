"""Keep the original visit/terminal update while preserving decoder metadata."""
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
from research.spatial_policy_value import load_model, POLICY_MAPPING_SHA256


def validate_model(plan, initial):
    if (plan['policy_head'] not in ['dense2550', 'relative50'] or
            initial.get('policy_head', 'dense2550') != plan['policy_head'] or
            initial.get('policy_mapping_sha256', POLICY_MAPPING_SHA256) != POLICY_MAPPING_SHA256):
        raise ValueError('Training policy decoder differs from frozen plan')


def install():
    if sha(ROOT / 'scripts/train_neural_selfplay.py') != '0d8f6fdf9c5536ed8bf996a79a53e8d38b4224905a6f9a716b92fd1c61613e57':
        raise ValueError('Pinned original self-play trainer changed')
    source = textwrap.dedent(inspect.getsource(original.main))

    def replace(needle, value):
        nonlocal source
        if source.count(needle) != 1:
            raise ValueError('Self-play decoder metadata patch site changed')
        source = source.replace(needle, value)

    replace("    dev=torch.device(plan['device']);net,initial=load_model(ROOT/plan['initial_model'],dev)",
            "    dev=torch.device(plan['device']);net,initial=load_model(ROOT/plan['initial_model'],dev)\n    _validate_model(plan,initial)")
    replace("'initial_network_sha256':initial['npz_sha256']",
            "'policy_head':plan['policy_head'],'policy_mapping_sha256':_mapping_sha,'initial_network_sha256':initial['npz_sha256']")
    replace("'phase_experts':initial['phase_experts'],'channels':initial['channels'],'blocks':initial['blocks']})",
            "'phase_experts':initial['phase_experts'],'channels':initial['channels'],'blocks':initial['blocks'],\n                    'policy_head':plan['policy_head'],'policy_mapping_sha256':_mapping_sha})")
    namespace = dict(original.__dict__, load_model=load_model,
                     _validate_model=validate_model, _mapping_sha=POLICY_MAPPING_SHA256)
    exec(compile(source, '<cdca-spatial-selfplay-training>', 'exec'), namespace)
    return namespace['main'], hashlib.sha256(source.encode()).hexdigest()


if __name__ == '__main__':
    main, _ = install()
    main()

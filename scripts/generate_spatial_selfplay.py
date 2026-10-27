"""Original complete-episode self-play with a safe dense/relative model loader."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_neural_selfplay as original
from research.neural_replay import sha
from research.spatial_policy_value import POLICY_MAPPING_SHA256


def validate_model(plan):
    metadata = json.loads((ROOT / plan['model'] / 'best.json').read_text())
    if (plan['policy_head'] not in ['dense2550', 'relative50'] or
            metadata.get('policy_head', 'dense2550') != plan['policy_head'] or
            metadata.get('policy_mapping_sha256', POLICY_MAPPING_SHA256) != POLICY_MAPPING_SHA256):
        raise ValueError('Generating policy decoder differs from frozen plan')


def initialize(model, device, leaf_batch):
    # Top-level initializer stays picklable by the original spawn workers.
    import torch
    from research.neural_search import ObservationPUCT
    from research.spatial_policy_value import load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    net, _ = load_model(ROOT / model, device)
    original.ENGINE = ObservationPUCT(net, device, leaf_batch=leaf_batch)


def install():
    if sha(ROOT / 'scripts/generate_neural_selfplay.py') != 'e47125ae9a6110ca4f97483c0efe3130f7cea41a1680d7d65709fb2b44b076a7':
        raise ValueError('Pinned original self-play generator changed')
    source = textwrap.dedent(inspect.getsource(original.main))
    needle = "    verify();out.mkdir(parents=True)"
    if source.count(needle) != 1:
        raise ValueError('Generator validation patch site changed')
    source = source.replace(needle, "    verify();_validate_model(plan);out.mkdir(parents=True)")
    namespace = dict(original.__dict__, initialize=initialize, _validate_model=validate_model)
    exec(compile(source, '<cdca-spatial-selfplay-generation>', 'exec'), namespace)
    return namespace['main'], hashlib.sha256(source.encode()).hexdigest()


if __name__ == '__main__':
    main, _ = install()
    main()

"""Whole-split equality checks before the matched raw/pruned learner fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.pruned_policy_targets import load_split
from research.native_terminal_targets import validate_split
from research.neural_replay import sha, save_json


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-plan', required=True)
    p.add_argument('--pruned-plan', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise ValueError('Refuse to overwrite pruned-data preflight')
    raw, pruned = [json.loads((ROOT / path).read_text()) for path in [args.raw_plan, args.pruned_plan]]
    generation = json.loads((ROOT / raw['generation_plan']).read_text())
    folder = ROOT / generation['out']
    manifest = json.loads((folder / 'manifest.json').read_text())
    for plan in [raw, pruned]:
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen paired target input changed: ' + name)
        validate_split(plan, generation, plan['training_games'], plan['selection_validation_games'])
    if raw['policy_target_scheme'] != 'raw-control' or pruned['policy_target_scheme'] != 'upstream-heuristic-pruned':
        raise ValueError('Wrong planned policy-target arms')
    rows = []
    for role in ['training_games', 'selection_validation_games']:
        if raw[role] != pruned[role]:
            raise ValueError('Policy-target arms change game membership')
        x, pi, z, meta = load_split(raw, folder, raw[role], manifest)
        px, ppi, pz, pmeta = load_split(pruned, folder, pruned[role], manifest)
        if not np.array_equal(x, px) or not np.array_equal(z, pz) or meta != pmeta:
            raise ValueError('Policy-target intervention changed observations, native returns or source binding')
        rows.append({'role': role, 'games': len(meta), 'positions': len(x),
            'observations_sha256': digest(x), 'native_returns_sha256': digest(z),
            'raw_policy_sha256': digest(pi), 'pruned_policy_sha256': digest(ppi),
            'changed_policy_rows': int(np.count_nonzero(np.any(np.abs(pi-ppi) > 1e-7, axis=1))),
            'bitwise_matched_observations_and_returns': True,
            'whole_game_membership_and_source_binding_equal': True})
    save_json(out, {'schema': 1, 'status': 'completed', 'roles': rows,
        'raw_plan_sha256': sha(ROOT / args.raw_plan), 'pruned_plan_sha256': sha(ROOT / args.pruned_plan),
        'scope': 'Whole actual native-prefix TRAIN/selection split boundary checks. Only policy targets change; no fit, teacher label imitation, strength match or promotion.'})


if __name__ == '__main__':
    main()

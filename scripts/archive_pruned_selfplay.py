"""Validate exact-search-derived targets and optionally archive numeric arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import load_episode, sha, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--raw')
    args = p.parse_args()
    path, out = ROOT / args.plan, ROOT / args.out
    if out.exists() or (args.raw and (ROOT / args.raw).exists()):
        raise ValueError('Refuse to overwrite derived-target archives')
    plan = json.loads(path.read_text())
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen target-reanalysis input changed: ' + name)
    folder = ROOT / plan['out']
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['status'] != 'completed' or manifest['plan_sha256'] != sha(path):
        raise ValueError('Incomplete or changed target reanalysis')
    if sorted(r['index'] for r in manifest['episodes']) != sorted(plan['games']):
        raise ValueError('Missing or substituted source prefixes')
    annotations = {r['index']: r for r in json.loads((ROOT / plan['annotations']).read_text())['games']}
    rows, files = [], []
    for row in manifest['episodes']:
        index, n = row['index'], row['positions']
        source = ROOT / plan['episodes'] / f'game-{index:06d}.npz'
        original, metadata = load_episode(source)
        target = folder / source.name
        if (n != annotations[index]['native_terminal_ply'] or
                sha(source) != row['source_npz_sha256'] or
                sha(target) != row['target_npz_sha256'] or
                hashlib.sha256(original['visit_counts'][:n].tobytes()).hexdigest() != row['source_prefix_visits_sha256'] or
                not row['source_visits_reproduced_exactly'] or
                not row['actual_moves_and_reveals_reproduced_exactly']):
            raise ValueError('Source-prefix search reproduction boundary differs')
        if json.loads(target.with_suffix('.json').read_text()) != row:
            raise ValueError('Derived-target sidecar changed')
        with np.load(target, allow_pickle=False) as arrays:
            if arrays.files != ['policy_targets']:
                raise ValueError('Unexpected derived-target arrays')
            policy = arrays['policy_targets']
        if (policy.shape != (n, 2550) or policy.dtype != np.float32 or
                not np.isfinite(policy).all() or (policy < 0).any() or
                not np.allclose(policy.sum(1), 1, atol=2e-7) or
                np.any(policy[original['visit_counts'][:n] == 0] != 0) or
                hashlib.sha256(policy.tobytes()).hexdigest() != row['policy_targets_sha256']):
            raise ValueError('Derived target shape, numeric content or real visit support differs')
        rows.append(row)
        files.extend([target, target.with_suffix('.json')])
    result = {'schema': 1, 'status': 'completed', 'scope': plan['scope'],
        'plan_sha256': sha(path), 'manifest_sha256': sha(folder / 'manifest.json'),
        'generating_model_sha256': sha(ROOT / plan['model'] / 'best.npz'),
        'annotations_sha256': sha(ROOT / plan['annotations']),
        'games': len(rows), 'positions': sum(r['positions'] for r in rows),
        'source_root_counts_and_acting_reproduced_exactly': True,
        'changed_target_roots': sum(r['changed_target_roots'] for r in rows),
        'mean_raw_entropy': sum(r['positions'] * r['mean_raw_entropy'] for r in rows) / sum(r['positions'] for r in rows),
        'mean_pruned_entropy': sum(r['positions'] * r['mean_pruned_entropy'] for r in rows) / sum(r['positions'] for r in rows),
        'episodes': rows, 'limitations': plan['limitations']}
    if args.raw:
        archive = ROOT / args.raw
        with tarfile.open(archive, 'w:gz') as tar:
            for file in files:
                tar.add(file, arcname=file.name, recursive=False)
        result['raw_targets'] = {'path': args.raw, 'sha256': sha(archive), 'files': len(files)}
    save_json(out, result)


if __name__ == '__main__':
    main()

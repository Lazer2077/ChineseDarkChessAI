"""Separate audited pruning targets on the same native-prefix observations/returns."""
import hashlib
import json
import numpy as np
from research.neural_replay import load_episode, sha
from research.native_terminal_targets import load_split as native_load_split
from research.policy_value import ROOT


def load_split(plan, folder, indices, source_manifest):
    scheme = plan['policy_target_scheme']
    if scheme not in ('raw-control', 'upstream-heuristic-pruned') or plan['target_profile'] != 'native-prefix':
        raise ValueError('Unplanned policy target or value profile')
    directory = ROOT / plan['pruned_target_directory']
    manifest_file = directory / 'manifest.json'
    if sha(manifest_file) != plan['pruned_manifest_sha256']:
        raise ValueError('Derived-target manifest changed')
    derived = json.loads(manifest_file.read_text())
    if derived['status'] != 'completed' or derived['games'] != 288:
        raise ValueError('Require all288completed native-prefix derived targets')
    rows = {r['index']: r for r in derived['episodes']}
    if set(rows) != set(range(288)):
        raise ValueError('Missing or substituted derived-target games')
    raw, actual_pi, z, metadata = native_load_split(plan, folder, indices, source_manifest)
    targets = []
    for index, meta in zip(indices, metadata):
        row = rows[index]
        source, _ = load_episode(folder / f'game-{index:06d}.npz')
        n = meta['positions']
        file = directory / f'game-{index:06d}.npz'
        if (row['positions'] != n or row['source_npz_sha256'] != meta['sha256'] or
                sha(file) != row['target_npz_sha256'] or
                json.loads(file.with_suffix('.json').read_text()) != row or
                not row['source_visits_reproduced_exactly'] or
                not row['actual_moves_and_reveals_reproduced_exactly'] or
                hashlib.sha256(source['visit_counts'][:n].tobytes()).hexdigest() != row['source_prefix_visits_sha256']):
            raise ValueError('Derived target is not bound to the original exact-search prefix')
        with np.load(file, allow_pickle=False) as arrays:
            if arrays.files != ['policy_targets']:
                raise ValueError('Unexpected pruned target arrays')
            policy = arrays['policy_targets']
        if (policy.shape != (n, 2550) or policy.dtype != np.float32 or
                not np.isfinite(policy).all() or (policy < 0).any() or
                not np.allclose(policy.sum(1), 1, atol=2e-7) or
                np.any(policy[source['visit_counts'][:n] == 0] != 0) or
                hashlib.sha256(policy.tobytes()).hexdigest() != row['policy_targets_sha256']):
            raise ValueError('Pruned numeric values or actual visit support changed')
        targets.append(policy)
        meta['derived_targets_sha256'] = row['target_npz_sha256']
    return raw, actual_pi if scheme == 'raw-control' else np.concatenate(targets), z, metadata

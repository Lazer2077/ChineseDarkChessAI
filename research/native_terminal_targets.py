"""Native-terminal prefix targets; original source trajectories stay immutable."""
import json

import numpy as np
from research.neural_replay import load_episode, sha
from research.policy_value import AZ

PROFILES = ('source-full-control', 'source-prefix-control', 'native-prefix')


def annotation(plan, root):
    path = root / plan['native_terminal_annotations']
    if sha(path) != plan['native_terminal_annotations_sha256']:
        raise ValueError('Native terminal annotations changed')
    report = json.loads(path.read_text())
    if report['status'] != 'completed':
        raise ValueError('Incomplete native terminal reanalysis')
    return {row['index']: row for row in report['games']}


def validate_split(plan, generation, train, selection):
    from research.policy_value import ROOT
    if plan['target_profile'] not in PROFILES:
        raise ValueError('Unknown terminal-target profile')
    if len(set(train + selection)) != len(train) + len(selection):
        raise ValueError('Overlapping training/selection games')
    if plan['target_profile'] == 'source-full-control':
        expected = set(range(generation['games']))
    else:
        expected = {i for i, row in annotation(plan, ROOT).items() if row['accepted']}
    if set(train + selection) != expected or not train or not selection:
        raise ValueError('Missing or unsupported native-compatible whole-game split')
    if set(train) != expected.intersection(plan['original_training_games']):
        raise ValueError('Training membership changed from frozen source roles')
    if set(selection) != expected.intersection(plan['original_selection_games']):
        raise ValueError('Selection membership changed from frozen source roles')


def load_split(plan, folder, indices, manifest):
    from research.policy_value import ROOT
    from scripts.train_neural_selfplay import load_split as original_load_split
    if plan['target_profile'] == 'source-full-control':
        return original_load_split(folder, indices, manifest)
    annotations = annotation(plan, ROOT)
    entries = {row['index']: row for row in manifest['episodes']}
    records, policies, values, metadata = [], [], [], []
    for index in indices:
        file = folder / f'game-{index:06d}.npz'
        data, meta = load_episode(file)
        expected = dict(entries[index])
        expected.pop('index')
        if expected != meta:
            raise ValueError('Source episode differs from completed generation')
        row = annotations[index]
        if not row['accepted'] or row['source_npz_sha256'] != meta['npz_sha256']:
            raise ValueError('Unaccepted or unbound native prefix')
        n = row['native_terminal_ply']
        if not 0 < n <= meta['positions'] or row['native_red_return'] not in (-1, 0, 1):
            raise ValueError('Prefix has no genuine native terminal')
        records.append(data['records'][:n])
        policies.append(data['visit_counts'][:n].astype(np.float32) / data['root_visits'][:n, None])
        if plan['target_profile'] == 'native-prefix':
            red_value = row['native_red_return']
            z = np.where(data['sides'][:n] == AZ.jq.RED, red_value, -red_value).astype(np.float32)
        else:
            z = data['values'][:n]
        values.append(z)
        metadata.append({'index': index, 'seed': meta['seed'], 'positions': n,
                         'source_positions': meta['positions'], 'sha256': meta['npz_sha256'],
                         'native_terminal_ply': n, 'target_profile': plan['target_profile']})
    return np.concatenate(records), np.concatenate(policies), np.concatenate(values), metadata

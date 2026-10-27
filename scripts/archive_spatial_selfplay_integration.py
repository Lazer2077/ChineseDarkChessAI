"""Require unchanged dense behavior and complete relative self-play/update audits."""
import argparse
import gzip
import io
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.neural_replay import load_episode, save_json, sha
from research.policy_value import encode_records
from research.spatial_policy_value import load_model, POLICY_MAPPING_SHA256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    out = ROOT / plan['out']
    if out.exists():
        raise ValueError('Refuse to overwrite self-play integration report')
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen integration input changed: ' + name)
    torch.set_num_threads(1)
    generations = {}
    archive_files = []
    arrays = {}
    for label, spec in plan['generations'].items():
        gp = ROOT / spec['plan']
        generation = json.loads(gp.read_text())
        folder = ROOT / generation['out']
        manifest = json.loads((folder / 'manifest.json').read_text())
        if (manifest['status'] != 'completed' or manifest['completed'] != generation['games'] or
                manifest['plan_sha256'] != sha(gp)):
            raise ValueError('Incomplete generation integration')
        rows = []
        arrays[label] = []
        for index in range(generation['games']):
            file = folder / f'game-{index:06d}.npz'
            data, meta = load_episode(file)
            if (meta['seed'], meta['simulations']) != (generation['seed'] + index, generation['simulations']):
                raise ValueError('Wrong generated game identity/work')
            arrays[label].append((data, meta))
            rows.append({'index': index, 'seed': meta['seed'], 'npz_sha256': meta['npz_sha256'],
                         'audit': meta['audit']})
            for file in [file, file.with_suffix('.json')]:
                archive_files.append((label + '/' + file.name, file))
        generations[label] = {'plan_sha256': sha(gp), 'games': len(rows), 'episodes': rows}
    legacy = arrays['dense-legacy']
    dense = arrays['dense-spatial']
    if len(legacy) != len(dense):
        raise ValueError('Different dense integration game count')
    for (a, am), (b, bm) in zip(legacy, dense):
        if set(a) != set(b) or any(not np.array_equal(a[k], b[k]) for k in a):
            raise ValueError('New loader changed original dense self-play trajectories')
        if any(am[k] != bm[k] for k in ['seed', 'positions', 'outcome', 'reason']):
            raise ValueError('Dense game outcomes changed')
    trainings = {}
    for label, spec in plan['trainings'].items():
        tp = ROOT / spec['plan']
        training = json.loads(tp.read_text())
        folder = ROOT / training['out']
        manifest = json.loads((folder / 'manifest.json').read_text())
        best = json.loads((folder / 'best.json').read_text())
        curve = json.loads((folder / 'training-log.json').read_text())
        expected = training['epochs'] * math.ceil(sum(row['positions'] for row in manifest['training']) / training['batch'])
        if (manifest['status'] != 'completed' or manifest['plan_sha256'] != sha(tp) or
                manifest['epochs_completed'] != training['epochs'] or manifest['updates'] != expected or
                best['npz_sha256'] != sha(folder / 'best.npz')):
            raise ValueError('Incomplete optimizer budget or wrong selected artifact')
        model, metadata = load_model(folder, 'cpu')
        val = arrays[spec['generation']][training['selection_validation_games'][0]][0]['records']
        with torch.inference_mode():
            policy, value = model(encode_records(torch.from_numpy(val)))
        if not torch.isfinite(policy).all() or not torch.isfinite(value).all():
            raise ValueError('Nonfinite safely reloaded policy/value')
        trainings[label] = {'plan_sha256': sha(tp), 'model_sha256': sha(folder / 'best.npz'),
                            'updates': expected, 'training_curve': curve,
                            'safe_reload_policy_head': metadata.get('policy_head', 'dense2550')}
    if trainings['dense-legacy']['model_sha256'] != trainings['dense-spatial']['model_sha256']:
        raise ValueError('New trainer changed original dense update arrays')
    if trainings['relative-spatial']['safe_reload_policy_head'] != 'relative50':
        raise ValueError('Relative decoder metadata lost after training')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for name, file in archive_files:
            content = file.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(content))
    raw = ROOT / plan['raw_archive']
    if raw.exists():
        raise ValueError('Refuse to overwrite episode archive')
    with raw.open('xb') as handle:
        handle.write(gzip.compress(buffer.getvalue(), mtime=0))
    report = {'schema': 1, 'status': 'completed', 'plan_sha256': sha(path),
              'scope': plan['scope'], 'generations': generations, 'trainings': trainings,
              'dense_trajectories_exact': True, 'dense_training_npz_exact': True,
              'relative_safe_model_roundtrip': True, 'policy_mapping_sha256': POLICY_MAPPING_SHA256,
              'raw_episodes': {'path': plan['raw_archive'], 'sha256': sha(raw)},
              'limitations': plan['limitations']}
    save_json(out, report)
    print(json.dumps({'status': 'completed', 'dense_trajectories_exact': True,
                      'dense_training_npz_exact': True, 'relative_safe_roundtrip': True}), flush=True)


if __name__ == '__main__':
    main()

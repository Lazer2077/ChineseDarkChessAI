"""Bind completed derived arrays to prespecified paired fits, then execute once."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import sha, save_json


def write_new(path, data):
    path = ROOT / path
    if path.exists():
        raise ValueError('Refuse to overwrite a prepared training plan')
    save_json(path, data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True)
    args = p.parse_args()
    recipe_file = ROOT / args.recipe
    recipe = json.loads(recipe_file.read_text())
    for name, expected in recipe['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen paired-fit preparation input changed: ' + name)
    report = json.loads((ROOT / recipe['derived_report']).read_text())
    folder = ROOT / recipe['derived_directory']
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (report['status'] != 'completed' or report['games'] != 288 or report['positions'] != 15331 or
            not report['source_root_counts_and_acting_reproduced_exactly'] or
            report['manifest_sha256'] != sha(folder / 'manifest.json') or
            manifest['status'] != 'completed' or manifest['positions'] != 15331):
        raise ValueError('Require all exact-search-derived native-prefix targets before fitting')
    template = json.loads((ROOT / recipe['native_training_template']).read_text())
    if (template['seed'] != 20261031 or template['epochs'] != 4 or
            template['batch'] != 128 or template['lr'] != 1e-5 or template['target_profile'] != 'native-prefix'):
        raise ValueError('Original native-prefix control recipe changed')
    hashes = {**template['sha256'], **recipe['sha256'], args.recipe: sha(recipe_file)}
    hashes[recipe['derived_report']] = sha(ROOT / recipe['derived_report'])
    hashes[str((folder / 'manifest.json').relative_to(ROOT))] = sha(folder / 'manifest.json')
    for row in manifest['episodes']:
        file = folder / f"game-{row['index']:06d}.npz"
        if sha(file) != row['target_npz_sha256']:
            raise ValueError('Derived array changed before fit-plan preparation')
        hashes[str(file.relative_to(ROOT))] = sha(file)
        hashes[str(file.with_suffix('.json').relative_to(ROOT))] = sha(file.with_suffix('.json'))
    plans = {}
    for scheme in ['raw-control', 'upstream-heuristic-pruned']:
        config = {**template, 'out': recipe['training_root'] + '/' + scheme,
            'policy_target_scheme': scheme, 'pruned_target_directory': recipe['derived_directory'],
            'pruned_manifest_sha256': sha(folder / 'manifest.json'),
            'expected_raw_control_sha256': recipe['expected_raw_control_sha256'],
            'scope': recipe['scope'], 'sha256': dict(hashes)}
        path = recipe['training_plans'][scheme]
        write_new(path, config)
        plans[scheme] = path
    pipeline = {'schema': 1, 'out': recipe['training_controller'], 'scope': recipe['scope'],
        'sha256': {**hashes, **{path: sha(ROOT / path) for path in plans.values()}},
        'steps': [{'id': 'data-preflight', 'argv': ['scripts/preflight_pruned_policy_targets.py',
            '--raw-plan', plans['raw-control'], '--pruned-plan', plans['upstream-heuristic-pruned'],
            '--out', recipe['preflight_report']]}]}
    for scheme, path in plans.items():
        pipeline['steps'] += [
            {'id': 'fit-' + scheme, 'argv': ['scripts/train_pruned_native_selfplay.py', '--plan', path]},
            {'id': 'archive-' + scheme, 'argv': ['scripts/archive_pruned_policy_training.py',
                '--plan', path, '--out', recipe['training_reports'][scheme]]}]
    write_new(recipe['pipeline_plan'], pipeline)
    os.execv(sys.executable, [sys.executable, str(ROOT / 'scripts/run_frozen_experiment.py'),
                             '--plan', recipe['pipeline_plan']])


if __name__ == '__main__':
    main()

"""Freeze all prespecified decoder matches after both full fits finish."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def validate_training(report, protocol, head):
    if report['status'] != 'completed' or report['policy_head'] != head:
        raise ValueError('Wrong or incomplete training profile')
    if (report['channels'], report['blocks']) != (
            protocol['body']['channels'], protocol['body']['residual_blocks']):
        raise ValueError('Wrong backbone')
    epochs = protocol['training']['epochs']
    updates = protocol['training']['updates_each']
    curve = report['training_curve']
    if (report['epochs'], report['updates']) != (epochs, updates):
        raise ValueError('Incomplete prespecified training budget')
    if [r['epoch'] for r in curve] != list(range(1, epochs + 1)):
        raise ValueError('Incomplete epoch curve')
    if [r['updates'] for r in curve] != [updates // epochs * i for i in range(1, epochs + 1)]:
        raise ValueError('Wrong optimization schedule')
    if any(not math.isfinite(r['validation']['loss']) for r in curve):
        raise ValueError('Nonfinite selection loss')
    selected = min(curve, key=lambda r: r['validation']['loss'])['epoch']
    if selected != report['selected_epoch']:
        raise ValueError('Checkpoint not selected by frozen validation rule')


def prepare(recipe, recipe_path, root=ROOT):
    protocol = read(root / recipe['protocol'])
    frozen = dict(recipe['sha256'])
    frozen[recipe_path] = sha(root / recipe_path)
    artifacts = {}
    reports = {}
    for head, spec in recipe['arms'].items():
        state = read(root / spec['state'])
        if state['status'] != 'completed' or state['plan_sha256'] != spec['controller_plan_sha256']:
            raise ValueError('Training controller not completed with frozen plan')
        report = read(root / spec['report'])
        validate_training(report, protocol, head)
        if report['training_plan_sha256'] != sha(root / spec['training_plan']):
            raise ValueError('Wrong training report plan')
        folder = root / spec['model_dir']
        manifest, best = read(folder / 'manifest.json'), read(folder / 'best.json')
        model_path = str(Path(spec['model_dir']) / 'best.npz')
        model_sha = sha(root / model_path)
        if (manifest['status'] != 'completed' or
                manifest['plan_sha256'] != report['training_plan_sha256'] or
                manifest['best_npz_sha256'] != model_sha or
                best['npz_sha256'] != model_sha or report['model_sha256'] != model_sha or
                best['epoch'] != report['selected_epoch'] or
                best['policy_head'] != head or
                best['policy_mapping_sha256'] != report['policy_mapping_sha256']):
            raise ValueError('Selected training artifact differs')
        for path in [spec['report'], spec['training_plan'], spec['state'], model_path,
                     str(Path(spec['model_dir']) / 'manifest.json'),
                     str(Path(spec['model_dir']) / 'best.json'),
                     str(Path(spec['model_dir']) / 'training-log.json')]:
            frozen[path] = sha(root / path)
        artifacts[head] = {'path': model_path, 'sha256': model_sha}
        reports[head] = report
    a, b = reports['dense2550'], reports['relative50']
    if any(a[k] != b[k] for k in ['training', 'selection_validation', 'policy_mapping_sha256']):
        raise ValueError('Decoder profiles used different replay/selection/geometry')
    plans = {}
    for head, destination in recipe['evaluations'].items():
        steps = []
        for match in protocol['matches']:
            actor = 'dense2550' if match['candidate'] == 'dense128x15' else match['candidate']
            if actor != head:
                continue
            opponent = match['opponent']
            if opponent == 'matched-dense128x15':
                baseline = recipe['adapter']
                net = artifacts['dense2550']['path']
            elif opponent == 'old-warm64x15':
                baseline, net = recipe['adapter'], recipe['warm_model']
            elif opponent == 'unchanged-native-teacher':
                baseline, net = recipe['teacher_engine'], recipe['teacher_net']
            else:
                raise ValueError('Unknown prespecified opponent')
            tag = head + '-vs-' + opponent
            run = destination['out'] + '/' + tag
            archive = destination['report_prefix'] + '-' + tag
            argv = ['scripts/native_match.py', '--candidate=' + recipe['adapter'],
                    '--candidate-net=' + artifacts[head]['path'], '--candidate-options=LeafBatch=32',
                    '--baseline=' + baseline, '--net=' + net, '--tc=10+0.1',
                    '--games=' + str(match['games']), '--seed=' + str(match['seed']),
                    '--concurrency=2', '--stage=development', '--out=' + run]
            if baseline == recipe['adapter']:
                argv.append('--baseline-options=LeafBatch=32')
            steps.extend([
                {'id': tag, 'argv': argv},
                {'id': tag + '-audit', 'argv': ['scripts/audit_native_games.py', '--run', run,
                                               '--out', archive + '-audit.json']},
                {'id': tag + '-archive', 'argv': ['scripts/archive_native_results.py', '--out',
                                                 archive + '.json', '--raw-dir', 'results/raw', run]}])
        plans[head] = {'schema': 1, 'out': destination['out'],
                       'scope': protocol['scope'], 'resources': protocol['match_conditions'],
                       'outcome_dependence': protocol['outcome_dependence'],
                       'artifacts': artifacts, 'steps': steps, 'sha256': frozen}
    if sum(len(p['steps']) for p in plans.values()) != 3 * len(protocol['matches']):
        raise ValueError('Prespecified match omitted')
    return plans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True)
    p.add_argument('--relative-device')
    p.add_argument('--dense-device')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    recipe_path = Path(args.recipe).as_posix()
    recipe = read(ROOT / recipe_path)
    digest = sha(ROOT / recipe_path)
    out = ROOT / recipe['out']
    out.mkdir(parents=True, exist_ok=False)
    state = {'schema': 1, 'status': 'waiting', 'pid': os.getpid(), 'recipe_sha256': digest}

    def verify():
        if sha(ROOT / recipe_path) != digest:
            raise ValueError('Queue recipe changed')
        for name, expected in recipe['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen evaluation input changed: ' + name)

    def update(**values):
        state.update(values)
        save(out / 'state.json', state)

    try:
        if not args.prepare_only and (not args.relative_device or not args.dense_device or
                                     args.relative_device == args.dense_device):
            raise ValueError('Require two separate inference GPUs')
        while True:
            verify()
            waiting = []
            for spec in recipe['arms'].values():
                dependency = read(ROOT / spec['state'])
                if dependency['plan_sha256'] != spec['controller_plan_sha256']:
                    raise ValueError('Wrong training controller recipe')
                if dependency['status'] == 'completed':
                    continue
                if dependency['status'] != 'running':
                    raise ValueError('Training controller failed')
                command = Path('/proc') / str(dependency['pid']) / 'cmdline'
                if (not command.exists() or b'scripts/run_frozen_experiment.py' not in command.read_bytes()
                        or spec['controller_plan'].encode() not in command.read_bytes()):
                    raise ValueError('Training controller missing; never restart here')
                waiting.append(dependency['pid'])
            update(waiting_for=waiting)
            if not waiting:
                break
            time.sleep(15)
        plans = prepare(recipe, recipe_path)
        for head, plan in plans.items():
            path = ROOT / recipe['evaluations'][head]['plan']
            if path.exists() or (ROOT / plan['out']).exists():
                raise ValueError('Refuse to overwrite/restart evaluation')
        for head, plan in plans.items():
            save(ROOT / recipe['evaluations'][head]['plan'], plan)
        update(status='prepared', artifacts=plans['dense2550']['artifacts'])
        if args.prepare_only:
            update(status='completed_preparation_only')
            return
        # Training has released the devices. Refuse to overlap another CUDA job.
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                            '--format=csv,noheader,nounits'], text=True)
        uuids = dict(line.split(', ', 1) for line in inventory.strip().splitlines())
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                           '--format=csv,noheader,nounits'], text=True).splitlines()
        if any(uuids[d] in occupied for d in [args.relative_device, args.dense_device]):
            raise ValueError('An inference GPU remains occupied; no automatic restart')
        children = []
        for head, device in [('relative50', args.relative_device), ('dense2550', args.dense_device)]:
            verify()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (out / (head + '.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, 'scripts/run_frozen_experiment.py',
                                          '--plan', recipe['evaluations'][head]['plan']],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            update(status='running', child_pids=[c.pid for c in children])
        codes = [child.wait() for child in children]
        verify()
        if any(codes):
            raise RuntimeError('One or more fixed evaluation campaigns failed: ' + str(codes))
        update(status='completed', child_pids=[])
    except BaseException as error:
        update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()

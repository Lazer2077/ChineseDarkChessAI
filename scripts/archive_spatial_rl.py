"""Archive one fully completed spatial self-play iteration and audited matches."""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from research.neural_replay import load_episode, save_json, sha
from research.policy_value import encode_records
from research.spatial_policy_value import load_model, POLICY_MAPPING_SHA256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    plan_path = ROOT / args.plan
    plan = json.loads(plan_path.read_text())
    out = ROOT / plan['report']
    raw = ROOT / plan['raw_archive']
    if out.exists() or raw.exists():
        raise ValueError('Refuse to overwrite RL archive')
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen RL archive input changed: ' + name)
    recipe_path = ROOT / plan['series_plan']
    recipe = json.loads(recipe_path.read_text())
    folder = ROOT / recipe['out']
    state = json.loads((folder / 'state.json').read_text())
    if (recipe['iterations'] != 1 or state['status'] != 'completed' or state['completed'] != 1 or
            state['plan_sha256'] != sha(recipe_path)):
        raise ValueError('Require complete fixed one-iteration series')
    iteration = folder / 'iter-001'
    gp, tp = iteration / 'generation-plan.json', iteration / 'training-plan.json'
    generation, training = json.loads(gp.read_text()), json.loads(tp.read_text())
    episodes = ROOT / generation['out']
    gm = json.loads((episodes / 'manifest.json').read_text())
    trained = ROOT / training['out']
    tm = json.loads((trained / 'manifest.json').read_text())
    best = json.loads((trained / 'best.json').read_text())
    curve = json.loads((trained / 'training-log.json').read_text())
    if (gm['status'] != 'completed' or gm['completed'] != generation['games'] or
            gm['plan_sha256'] != sha(gp) or tm['status'] != 'completed' or tm['plan_sha256'] != sha(tp)):
        raise ValueError('Incomplete generation or training')
    if generation['model'] != recipe['initial_model'] or training['initial_model'] != generation['model']:
        raise ValueError('Update did not use its generating actor')
    if len(set(training['training_games'] + training['selection_validation_games'])) != generation['games']:
        raise ValueError('Wrong whole-game split')
    for name, expected in training['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Bound RL replay/training input changed: ' + name)
    rows, files = [], []
    selection_sample = None
    for index in range(generation['games']):
        file = episodes / f'game-{index:06d}.npz'
        data, meta = load_episode(file)
        if (meta['seed'], meta['simulations']) != (generation['seed'] + index, generation['simulations']):
            raise ValueError('Wrong source-game identity/search work')
        rows.append({'index': index, 'seed': meta['seed'], 'positions': meta['positions'],
                     'npz_sha256': meta['npz_sha256'], 'audit': meta['audit']})
        files.extend([file, file.with_suffix('.json')])
        if selection_sample is None and index in training['selection_validation_games']:
            selection_sample = data['records'][:8].copy()
    expected_updates = training['epochs'] * math.ceil(sum(r['positions'] for r in rows
        if r['index'] in training['training_games']) / training['batch'])
    if ([r['epoch'] for r in curve] != list(range(1, training['epochs'] + 1)) or
            curve[-1]['updates'] != expected_updates or
            any(not math.isfinite(r['validation']['loss']) for r in curve)):
        raise ValueError('Incomplete or nonfinite training curve')
    selected = min(curve, key=lambda r: r['validation']['loss'])['epoch']
    digest = sha(trained / 'best.npz')
    if (tm['epochs_completed'] != training['epochs'] or tm['updates'] != expected_updates or
            best['epoch'] != selected or digest != best['npz_sha256'] or
            digest != state['iterations'][0]['network_sha256'] or
            best['policy_head'] != recipe['policy_head'] or best['policy_mapping_sha256'] != POLICY_MAPPING_SHA256):
        raise ValueError('Wrong budget, selected epoch, decoder or played artifact')
    torch.set_num_threads(1)
    model, _ = load_model(trained, 'cpu')
    with torch.inference_mode():
        logits, value = model(encode_records(torch.from_numpy(selection_sample)))
    if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
        raise ValueError('Safe model reload is nonfinite')
    matches = []
    for name in ['parent', 'teacher']:
        report_path = ROOT / plan['matches'][name]['report']
        audit_path = ROOT / plan['matches'][name]['audit']
        report = json.loads(report_path.read_text())['reports'][0]
        audit = json.loads(audit_path.read_text())
        raw_hash = hashlib.sha256(gzip.decompress((ROOT / report['raw_games']['path']).read_bytes())).hexdigest()
        if (report['hashes']['candidate_net'] != digest or
                raw_hash != audit['raw_games_sha256'] or raw_hash != report['raw_games']['uncompressed_sha256'] or
                report['result']['games'] != recipe['match_games'] or audit['games'] != recipe['match_games']):
            raise ValueError('Played model/raw audit differs')
        matches.append({'opponent': name, 'report': plan['matches'][name]['report'],
                        'report_sha256': sha(report_path), 'audit': plan['matches'][name]['audit'],
                        'audit_sha256': sha(audit_path), 'result': report['result'],
                        'independent_audit': audit})
    with raw.open('xb') as handle, gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as tar:
            for file in files:
                content = file.read_bytes()
                info = tarfile.TarInfo(file.name)
                info.size, info.mtime = len(content), 0
                tar.addfile(info, io.BytesIO(content))
    if sha(trained / 'best.npz') != digest:
        raise ValueError('Selected model changed during archive')
    save_json(out, {'schema': 1, 'status': 'completed', 'scope': plan['scope'],
        'archive_plan_sha256': sha(plan_path), 'series_plan_sha256': sha(recipe_path),
        'generation_plan_sha256': sha(gp), 'training_plan_sha256': sha(tp),
        'initial_model_sha256': recipe['initial_model_sha256'], 'model_sha256': digest,
        'policy_head': best['policy_head'], 'selected_epoch': selected,
        'games': len(rows), 'positions': sum(r['positions'] for r in rows),
        'epochs': tm['epochs_completed'], 'updates': tm['updates'],
        'initial_validation': json.loads((trained / 'initial-validation.json').read_text()),
        'training_curve': curve, 'episodes': rows,
        'raw_episodes': {'path': plan['raw_archive'], 'sha256': sha(raw)},
        'matches': matches, 'limitations': plan['limitations']})
    print(json.dumps({'status': 'archived', 'model_sha256': digest, 'updates': expected_updates}), flush=True)


if __name__ == '__main__':
    main()

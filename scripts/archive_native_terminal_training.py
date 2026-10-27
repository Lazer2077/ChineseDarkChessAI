"""Archive fixed-final-epoch native terminal target controls, including reproduction."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from research.neural_replay import sha, save_json
from research.spatial_policy_value import load_model, POLICY_MAPPING_SHA256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--expected-control')
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    out = ROOT / args.out
    if out.exists():
        raise ValueError('Refuse to overwrite target fit report')
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen native-target input changed: ' + name)
    folder = ROOT / plan['out']
    state = json.loads((folder / 'manifest.json').read_text())
    best = json.loads((folder / 'best.json').read_text())
    curve = json.loads((folder / 'training-log.json').read_text())
    n = sum(r['positions'] for r in state['training'])
    nv = sum(r['positions'] for r in state['selection_validation'])
    updates = math.ceil(n / plan['batch']) * plan['epochs']
    digest = sha(folder / 'best.npz')
    if (state['status'] != 'completed' or state['plan_sha256'] != sha(path) or
            state['checkpoint_selection'] != 'fixed-final-epoch' or
            state['target_profile'] != plan['target_profile'] or
            state['epochs_completed'] != plan['epochs'] or state['updates'] != updates or
            [r['epoch'] for r in curve] != list(range(1, plan['epochs'] + 1)) or
            curve[-1]['updates'] != updates or best['epoch'] != plan['epochs'] or
            best['npz_sha256'] != digest or state['best_npz_sha256'] != digest or
            best['policy_mapping_sha256'] != POLICY_MAPPING_SHA256):
        raise ValueError('Wrong target, complete budget, final epoch or artifact')
    if args.expected_control and (plan['target_profile'] != 'source-full-control' or digest != args.expected_control):
        raise ValueError('Full source reproduction failed; stop prefix experiments')
    torch.set_num_threads(1)
    net, _ = load_model(folder, 'cpu')
    if best['policy_head'] != 'relative50' or sum(p.numel() for p in net.parameters()) != 5453555:
        raise ValueError('Wrong frozen relative architecture')
    save_json(out, {'schema': 1, 'status': 'completed', 'scope': plan['scope'],
        'training_plan_sha256': sha(path), 'target_profile': plan['target_profile'],
        'checkpoint_selection': state['checkpoint_selection'],
        'initial_model_sha256': state['initial_network_sha256'], 'model_sha256': digest,
        'native_terminal_annotations_sha256': plan['native_terminal_annotations_sha256'],
        'training_games': len(state['training']), 'training_positions': n,
        'selection_games': len(state['selection_validation']), 'selection_positions': nv,
        'epochs': state['epochs_completed'], 'updates': updates, 'selected_epoch': best['epoch'],
        'initial_validation': json.loads((folder / 'initial-validation.json').read_text()),
        'training_curve': curve, 'source_control_reproduced_bitwise': bool(args.expected_control),
        'limitations': 'Reused rollouts and one fixed training seed; different-label losses are not comparable strength metrics. Prefix controls share data/steps; full-source versus prefix also changes data and step count. No native match, policy-gradient theorem or promotion follows from fitting.'})
    print(json.dumps({'status': 'archived', 'profile': plan['target_profile'],
                      'model_sha256': digest, 'updates': updates}), flush=True)


if __name__ == '__main__':
    main()

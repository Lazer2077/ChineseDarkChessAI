"""Archive matched policy-target fits, enforcing the exact raw-control gate."""
import argparse
import json
import math
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import sha, save_json
from research.spatial_policy_value import load_model, POLICY_MAPPING_SHA256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    path, out = ROOT / args.plan, ROOT / args.out
    if out.exists():
        raise ValueError('Refuse to overwrite policy-target fit report')
    plan = json.loads(path.read_text())
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen pruning-fit input changed: ' + name)
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
            state['target_profile'] != 'native-prefix' or
            state['epochs_completed'] != plan['epochs'] or state['updates'] != updates or
            [r['epoch'] for r in curve] != list(range(1, plan['epochs'] + 1)) or
            curve[-1]['updates'] != updates or updates != 428 or
            best['epoch'] != 4 or best['npz_sha256'] != digest or state['best_npz_sha256'] != digest or
            best['policy_mapping_sha256'] != POLICY_MAPPING_SHA256 or n != 13646 or nv != 1685):
        raise ValueError('Wrong matched data, complete budget, final epoch or model')
    raw = plan['policy_target_scheme'] == 'raw-control'
    if raw and digest != plan['expected_raw_control_sha256']:
        raise ValueError('Exact raw native-prefix control reproduction failed; stop pruned fit')
    if plan['policy_target_scheme'] not in ('raw-control', 'upstream-heuristic-pruned'):
        raise ValueError('Unknown policy-target scheme')
    torch.set_num_threads(1)
    net, _ = load_model(folder, 'cpu')
    if best['policy_head'] != 'relative50' or sum(p.numel() for p in net.parameters()) != 5453555:
        raise ValueError('Wrong frozen relative architecture')
    save_json(out, {'schema': 1, 'status': 'completed', 'scope': plan['scope'],
        'training_plan_sha256': sha(path), 'policy_target_scheme': plan['policy_target_scheme'],
        'checkpoint_selection': state['checkpoint_selection'],
        'initial_model_sha256': state['initial_network_sha256'], 'model_sha256': digest,
        'raw_control_reproduced_bitwise': raw,
        'native_terminal_annotations_sha256': plan['native_terminal_annotations_sha256'],
        'derived_targets_manifest_sha256': plan['pruned_manifest_sha256'],
        'training_games': len(state['training']), 'training_positions': n,
        'selection_games': len(state['selection_validation']), 'selection_positions': nv,
        'epochs': state['epochs_completed'], 'updates': updates, 'selected_epoch': best['epoch'],
        'initial_validation': json.loads((folder / 'initial-validation.json').read_text()),
        'training_curve': curve,
        'limitations': 'One training seed, reused rollouts, approximate original search and upstream pruning heuristic. Only CE policy targets change against the matched raw native-prefix control. Different-target losses and fits alone do not establish playing strength.'})


if __name__ == '__main__':
    main()

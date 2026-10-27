"""Exactly reproduce recorded source-prefix searches; archive separate pruned targets."""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from research.neural_replay import fresh_game, load_episode, sha, save_json
from research.policy_value import AZ, records_from_info
from research.policy_target_pruning import audited_targets, entropy

ENGINE = None
PLAN = None
ANNOTATIONS = None
RULES = None


def initialize(plan):
    global ENGINE, PLAN, ANNOTATIONS, RULES
    import torch
    from research.spatial_policy_value import load_model
    from research.neural_search import ObservationPUCT
    from audit_spatial_source_rules import NativeRules
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PLAN = plan
    ANNOTATIONS = {r['index']: r for r in json.loads((ROOT / plan['annotations']).read_text())['games']}
    net, _ = load_model(ROOT / plan['model'], plan['device'])
    ENGINE = ObservationPUCT(net, plan['device'], leaf_batch=plan['leaf_batch'])
    RULES = NativeRules(ROOT / plan['library'])


def work(index):
    from audit_spatial_source_rules import public_start, Color, Kind, LETTERS
    started = time.monotonic()
    source = ROOT / PLAN['episodes'] / f'game-{index:06d}.npz'
    data, metadata = load_episode(source)
    annotation = ANNOTATIONS[index]
    if not annotation['accepted'] or annotation['source_npz_sha256'] != sha(source):
        raise ValueError('Unadmitted or changed original source game')
    n = annotation['native_terminal_ply']
    if not 0 < n <= metadata['positions']:
        raise ValueError('No positive genuine native terminal prefix')
    state, belief, environment_rng = fresh_game(metadata['seed'])
    history = []
    ENGINE.deadline = None
    ENGINE.rng.seed(metadata['seed'] ^ 0x5A17C9)
    targets, entropies, variations = [], [], []
    for ply in range(n):
        if state.game_outcome()[0] is not None:
            raise ValueError('Source already terminal before admitted prefix root')
        info = belief.info_state(state, state.side)
        if not np.array_equal(records_from_info(info)[0], data['records'][ply]):
            raise ValueError('Observer replay differs before search')
        policy, chosen, root = ENGINE.search(state, belief, history,
            temperature=1.0 if ply < 30 else .1, simulations=PLAN['simulations'],
            add_dirichlet=True, dirichlet_alpha=.3, dirichlet_epsilon=.25,
            return_root=True)
        counts, raw, pruned = audited_targets(root, ENGINE.config.c_puct)
        if (root.visits != PLAN['simulations'] or
                not np.array_equal(counts, data['visit_counts'][ply]) or
                not np.allclose(policy, raw, atol=1e-7)):
            raise ValueError(f'Exact source root-count reproduction failed: game{index}/ply{ply}')
        expected_move = tuple(map(int, data['moves'][ply]))
        if chosen != expected_move or chosen not in state.legal_moves():
            raise ValueError(f'Exact original acting/RNG reproduction failed: game{index}/ply{ply}')
        targets.append(pruned.astype(np.float32))
        entropies.append((entropy(raw), entropy(pruned)))
        variations.append(float(np.abs(raw-pruned).sum()/2))
        undo = AZ.im.apply_move_and_update_belief(state, belief, chosen, rng=environment_rng)
        if (undo.moved_reveal, undo.captured_reveal) != tuple(map(int, data['reveals'][ply])):
            raise ValueError('Exact environment reveal reproduction failed')
        history.append(chosen)
    moved_reveals = [Kind(LETTERS[int(r)]) if r else None for r in data['reveals'][:n, 0]]
    terminal = RULES.game_replay(public_start(), Color.RED,
        [AZ.jq.format_move_uci(m) for m in history], moved_reveals,
        no_capture_limit=PLAN['native_no_capture_limit'])
    if terminal.name != annotation['native_status'] or terminal.name == 'ONGOING':
        raise ValueError('Actual native prefix terminal changed')
    target = np.stack(targets)
    if (not np.isfinite(target).all() or (target < 0).any() or
            not np.allclose(target.sum(1), 1, atol=2e-7) or
            np.any(target[data['visit_counts'][:n] == 0] != 0)):
        raise ValueError('Invalid separate pruned targets')
    folder = ROOT / PLAN['out']
    file = folder / f'game-{index:06d}.npz'
    if file.exists() or file.with_suffix('.json').exists():
        raise ValueError('Refuse to overwrite derived targets')
    with file.with_suffix('.npz.tmp').open('xb') as handle:
        np.savez_compressed(handle, policy_targets=target)
    file.with_suffix('.npz.tmp').replace(file)
    with np.load(file, allow_pickle=False) as reload:
        if reload.files != ['policy_targets'] or not np.array_equal(reload['policy_targets'], target):
            raise ValueError('Safe target roundtrip differs')
    report = {'index': index, 'source_npz_sha256': sha(source), 'positions': n,
        'source_positions': metadata['positions'], 'native_status': terminal.name,
        'native_red_return': annotation['native_red_return'],
        'source_visits_reproduced_exactly': True, 'actual_moves_and_reveals_reproduced_exactly': True,
        'source_prefix_visits_sha256': hashlib.sha256(data['visit_counts'][:n].tobytes()).hexdigest(),
        'target_npz_sha256': sha(file),
        'policy_targets_sha256': hashlib.sha256(target.tobytes()).hexdigest(),
        'mean_raw_entropy': float(np.mean([e[0] for e in entropies])),
        'mean_pruned_entropy': float(np.mean([e[1] for e in entropies])),
        'mean_target_total_variation': float(np.mean(variations)),
        'changed_target_roots': int(sum(v > 1e-7 for v in variations)),
        'reanalysis_seconds': time.monotonic() - started}
    save_json(file.with_suffix('.json'), report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    digest = sha(path)
    out = ROOT / plan['out']
    if out.exists():
        raise ValueError('Refuse to overwrite or restart derived target generation')
    if len(set(plan['games'])) != len(plan['games']):
        raise ValueError('Duplicate planned source games')

    def verify():
        if sha(path) != digest:
            raise ValueError('Frozen target generation plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen target reanalysis input changed: ' + name)

    verify()
    out.mkdir(parents=True)
    status = {'schema': 1, 'status': 'running', 'pid': os.getpid(),
              'plan_sha256': digest, 'scope': plan['scope'], 'episodes': []}
    save_json(out / 'manifest.json', status)
    try:
        with multiprocessing.get_context('spawn').Pool(plan['workers'], initialize, (plan,)) as pool:
            for row in pool.imap_unordered(work, plan['games']):
                verify()
                status['episodes'].append(row)
                save_json(out / 'manifest.json', status)
                print(json.dumps({'game': row['index'], 'completed': len(status['episodes']),
                    'positions': row['positions'], 'changed_target_roots': row['changed_target_roots']}), flush=True)
        verify()
        if {r['index'] for r in status['episodes']} != set(plan['games']):
            raise ValueError('Missing reanalysed source games')
        status['episodes'].sort(key=lambda r: r['index'])
        status.update(status='completed', games=len(status['episodes']),
                      positions=sum(r['positions'] for r in status['episodes']))
        save_json(out / 'manifest.json', status)
    except BaseException as error:
        status.update(status='failed', error=str(error))
        save_json(out / 'manifest.json', status)
        raise


if __name__ == '__main__':
    main()

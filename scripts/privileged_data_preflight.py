"""Small audited fixed-deal data feasibility experiment; no teacher/actor fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_search import ObservationPUCT, load_model
from research.neural_replay import save_json, sha
from research.privileged_training import CONTRACT, fresh_fixed_game, training_pair, actor_projection
from research.policy_value import AZ


def replay(data, seed, outcome, reason):
    state, belief, rng = fresh_fixed_game(seed)
    public = captures = 0
    n = len(data['records'])
    if not n or data['dark_truth'].shape != (n, 90): raise ValueError('Invalid truth array')
    for i in range(n):
        if state.game_outcome()[0] is not None: raise ValueError('Record after terminal')
        raw, truth = training_pair(state, belief)
        if not np.array_equal(raw, data['records'][i]) or not np.array_equal(truth, data['dark_truth'][i]):
            raise ValueError('Seeded observation/true assignment mismatch')
        public += not truth.any()
        expected = AZ.outcome_value_for_player(outcome, state.side)
        if expected != float(data['values'][i]): raise ValueError('Incorrect actor-relative terminal target')
        move = tuple(map(int, data['moves'][i]))
        legal = set(state.legal_moves())
        if move not in legal: raise ValueError('Illegal recorded move')
        if int(data['root_visits'][i]) <= 0 or int(data['visit_counts'][i].sum()) != int(data['root_visits'][i]):
            raise ValueError('Incomplete simulation counts')
        for action in np.flatnonzero(data['visit_counts'][i]):
            if AZ.index_to_move(int(action)) not in legal:
                raise ValueError('Illegal target support')
        undo = AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
        if (undo.moved_reveal, undo.captured_reveal) != tuple(map(int, data['reveals'][i])):
            raise ValueError('Recorded reveal differs from fixed deal')
        captures += bool(undo.captured_reveal)
    if state.game_outcome() != (outcome, reason): raise ValueError('Incorrect/incomplete outcome')
    return {'positions_replayed': n, 'fully_public_positions': int(public),
            'private_dark_captures': int(captures), 'fixed_deal_labels_match': True,
            'observer_records_match': True, 'legal_policy_support': True,
            'terminal_actor_returns_match': True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); a = parser.parse_args()
    path = Path(a.plan); plan = json.loads(path.read_text()); digest = sha(path)
    def verify():
        if sha(path) != digest: raise ValueError('Plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT/name) != expected: raise ValueError('Frozen input changed: '+name)
    verify(); out = ROOT/plan['out']; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    net, model = load_model(ROOT/plan['model'], plan['device'])
    engine = ObservationPUCT(net, plan['device'], leaf_batch=plan['leaf_batch'])
    report = {'schema': 1, 'status': 'running', 'contract': CONTRACT, 'plan_sha256': digest,
              'network_sha256': model['npz_sha256'], 'scope': plan['scope'], 'games': []}
    save_json(out/'report.json', report)
    for index in range(plan['games']):
        seed = plan['seed']+index; state, belief, rng = fresh_fixed_game(seed)
        engine.rng.seed(seed ^ 0x5A17C9); engine.deadline = None
        history = []; rows = []; truths = []; counts = []; moves = []; reveals = []; sides = []; roots = []
        while state.game_outcome()[0] is None:
            if len(history) >= plan['max_plies']: raise ValueError('Truncated episode; refuse terminal label')
            raw, truth = training_pair(state, belief)
            # Crucially, this search receives no true assignment map.
            projected = actor_projection(state)
            pi, move, root = engine.search(projected, belief, history, simulations=plan['simulations'],
                temperature=1.0 if len(history)<30 else .1, add_dirichlet=True,
                dirichlet_alpha=.3, dirichlet_epsilon=.25, return_root=True)
            if root.visits != plan['simulations'] or move not in state.legal_moves():
                raise ValueError('Incomplete search or illegal actual move')
            visits = np.zeros(AZ.MOVE_SIZE, dtype=np.uint32)
            for choice, child in root.children.items(): visits[AZ.move_to_index(choice)] = child.visits
            if visits.sum() != root.visits or not np.allclose(pi, visits/root.visits, atol=1e-7):
                raise ValueError('Invalid visit target')
            rows.append(raw); truths.append(truth); counts.append(visits); roots.append(root.visits)
            sides.append(state.side); moves.append(move)
            undo = AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
            reveals.append((undo.moved_reveal, undo.captured_reveal)); history.append(move)
        outcome, reason = state.game_outcome()
        data = {'records': np.asarray(rows, dtype=np.uint8), 'dark_truth': np.asarray(truths, dtype=np.uint8),
                'moves': np.asarray(moves, dtype=np.uint8), 'reveals': np.asarray(reveals, dtype=np.uint8),
                'visit_counts': np.asarray(counts, dtype=np.uint32), 'root_visits': np.asarray(roots, dtype=np.uint32),
                'values': np.asarray([AZ.outcome_value_for_player(outcome,s) for s in sides], dtype=np.float32)}
        audit = replay(data, seed, outcome, reason)
        file = out/f'game-{index:06d}.npz'
        with file.open('xb') as handle: np.savez_compressed(handle, **data)
        # Reopen only safe arrays; the independent replay uses the written bytes.
        with np.load(file, allow_pickle=False) as arrays:
            second = replay({k: arrays[k] for k in arrays.files}, seed, outcome, reason)
        if second != audit: raise ValueError('Written replay differs')
        verify(); row = {'index': index, 'seed': seed, 'outcome': outcome, 'reason': reason,
                         'npz_sha256': sha(file), 'audit': audit}
        report['games'].append(row); save_json(out/'report.json', report)
        print(json.dumps(row), flush=True)
    verify(); report['status'] = 'completed'; save_json(out/'report.json', report)


if __name__ == '__main__': main()

"""Fixed TRAIN-root, fixed-work root-allocation diagnostic, CPU only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.policy_value import AZ
from research.neural_search import ObservationPUCT, load_model
from research.sequential_halving import HalvingPUCT, SEARCH_SHA256
from research.native_opponent_policy import FixedDepthOpponent
from research.teacher_leaf import observer_fen
from research.neural_replay import load_episode, fresh_game, save_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); args = parser.parse_args()
    path = Path(args.plan); digest = hashlib.sha256(path.read_bytes()).hexdigest(); plan = json.loads(path.read_text())
    def verify():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Plan changed')
        for name, h in plan['sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != h: raise ValueError('Pinned input changed: '+name)
    verify(); out = ROOT/plan['out']; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    net, _ = load_model(ROOT/plan['model'], 'cpu')
    report = {'schema': 1, 'status': 'running', 'scope': plan['scope'], 'plan_sha256': digest,
              'search_sha256': SEARCH_SHA256, 'cases': [], 'samples': []}
    save_json(out/'report.json', report)
    with FixedDepthOpponent(ROOT/plan['teacher_binary'], ROOT/plan['teacher_net'], depth=plan['teacher_depth']) as teacher:
        for index, case in enumerate(plan['cases']):
            data, meta = load_episode(ROOT/plan['episodes']/f"game-{case['game']:06d}.npz")
            state, belief, rng = fresh_game(meta['seed']); history = []
            for raw_move in data['moves'][:case['ply']]:
                move = tuple(map(int, raw_move)); AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng); history.append(move)
            info = belief.info_state(state, state.side)
            legal = [AZ.jq.format_move_uci(m) for m in state.legal_moves()]
            scores = teacher.scores(observer_fen(info), legal)
            report['cases'].append({**case, 'teacher_q_proxy_cp': scores, 'legal_actions': len(legal)})
            variants = [('puct', None, 0.), ('halving-zero-noise', 0, 0.)]
            variants += [('halving-gumbel', seed, 1.) for seed in plan['gumbel_seeds']]
            for variant, seed, noise in variants:
                engine_class = ObservationPUCT if variant == 'puct' else HalvingPUCT
                engine = engine_class(net, seed=plan['world_seed'], leaf_batch=plan['leaf_batch'])
                inputs = (copy.deepcopy(state), copy.deepcopy(belief), history.copy())
                kwargs = {} if variant == 'puct' else {'gumbel_seed': seed, 'noise_scale': noise, 'candidates': plan['candidates']}
                started = time.perf_counter()
                policy, selected, root = engine.search(*inputs, simulations=plan['simulations'], temperature=0., return_root=True, **kwargs)
                elapsed = time.perf_counter()-started
                if root.visits != plan['simulations'] or sum(c.visits for c in root.children.values()) != root.visits:
                    raise ValueError('Incomplete actual root backups')
                if abs(sum(policy)-1) > 1e-7 or selected not in state.legal_moves(): raise ValueError('Invalid policy/action')
                chosen = AZ.jq.format_move_uci(selected)
                report['samples'].append({'case': index, 'variant': variant, 'gumbel_seed': seed,
                    'elapsed_seconds_cpu': elapsed, 'selected_move': chosen, 'teacher_q_proxy_cp': scores[chosen],
                    'teacher_proxy_regret_cp': max(scores.values())-scores[chosen],
                    'root_visits': root.visits, 'visited_actions': sum(c.visits > 0 for c in root.children.values()),
                    'root_value': root.value_sum/root.visits, 'policy': policy,
                    'root_children': [{'move': AZ.jq.format_move_uci(m), 'visits': c.visits, 'next_player_value_sum': c.value_sum}
                                      for m, c in root.children.items()],
                    'halving_rounds': engine.halving.trace if variant != 'puct' else None})
                verify(); save_json(out/'report.json', report)
            print(json.dumps({'completed_case': index, 'samples': len(report['samples'])}), flush=True)
    verify(); report['status'] = 'completed'; save_json(out/'report.json', report)


if __name__ == '__main__': main()

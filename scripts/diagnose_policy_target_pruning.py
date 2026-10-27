"""Frozen phase-stratified TRAIN roots; target-only pruning and lawful expert proxy."""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import fresh_game, load_episode, save_json, sha
from research.neural_search import ObservationPUCT
from research.spatial_policy_value import load_model
from research.policy_value import AZ, records_from_info
from research.observer_expert import ObserverHistoryExpert, observer_token
from research.teacher_leaf import observer_fen
from research.policy_target_pruning import audited_targets, entropy, paired_game_summary


def forced_proxy(expert, tokens, fen, actions, nodes):
    results = {}
    for move in actions:
        snapshot = expert.position_snapshot(tokens, fen)
        if set(snapshot['legal_moves']) != set(actions):
            raise ValueError('Forced-search native action set changed')
        expert._send(f'go nodes {nodes} searchmoves {move}')
        lines = expert._until('bestmove ')
        if lines[-1].split()[1] != move:
            raise ValueError('Native forced action was not respected')
        scores = []
        for line in lines:
            match = re.search(r'^info depth (\d+).*score (cp|mate) (-?\d+)', line)
            if match and 'lowerbound' not in line and 'upperbound' not in line:
                value = int(match[3])
                proxy = float(np.tanh(value / 860)) if match[2] == 'cp' else float(np.sign(value))
                scores.append({'kind': match[2], 'value': value,
                               'completed_depth': int(match[1]), 'proxy': proxy})
        if not scores:
            raise ValueError('No completed native forced-action score')
        results[move] = scores[-1]
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    digest = sha(path)

    def verify():
        if sha(path) != digest:
            raise ValueError('Pruning diagnostic plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen pruning input changed: ' + name)

    verify()
    out = ROOT / plan['out']
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    net, metadata = load_model(ROOT / plan['model'], plan['device'])
    annotations = {r['index']: r for r in json.loads((ROOT / plan['annotations']).read_text())['games']}
    report = {'schema': 1, 'status': 'running', 'scope': plan['scope'],
              'plan_sha256': digest, 'model_sha256': metadata['npz_sha256'],
              'cases': [], 'quarantined': [], 'limitations': plan['limitations']}
    save_json(out / 'report.json', report)
    grouped = {}
    for case in plan['cases']:
        if case['game'] not in plan['allowed_training_games']:
            raise ValueError('Selection or strength layout entered diagnostic')
        if not 0 <= case['ply'] < annotations[case['game']]['native_terminal_ply']:
            raise ValueError('Post-native-terminal root entered diagnostic')
        grouped.setdefault(case['game'], []).append(case)
    with ObserverHistoryExpert(ROOT / plan['teacher_binary'], ROOT / plan['teacher_net'], depth=1) as teacher:
        for game, cases in grouped.items():
            data, meta = load_episode(ROOT / plan['episodes'] / f'game-{game:06d}.npz')
            state, belief, rng = fresh_game(meta['seed'])
            history, histories = [], [[], []]
            by_ply = {r['ply']: r for r in cases}
            if len(by_ply) != len(cases):
                raise ValueError('Duplicate selected root')
            for ply in range(max(by_ply) + 1):
                if ply in by_ply:
                    case = by_ply[ply]
                    actor = state.side
                    info = belief.info_state(state, actor)
                    if not np.array_equal(records_from_info(info)[0], data['records'][ply]):
                        raise ValueError('Full-history observation replay differs')
                    dark = sum(AZ.jq.is_dark(p) for p in state.board if p)
                    if dark != case['live_darks']:
                        raise ValueError('Selected physical phase differs')
                    fen = observer_fen(info)
                    snapshot = teacher.position_snapshot(histories[actor], fen)
                    actions = [AZ.jq.format_move_uci(m) for m in state.legal_moves()]
                    if set(snapshot['legal_moves']) != set(actions):
                        report['quarantined'].append({**case, 'reason': 'native/source complete legal sets differ',
                            'native_only': sorted(set(snapshot['legal_moves']) - set(actions)),
                            'source_only': sorted(set(actions) - set(snapshot['legal_moves']))})
                    else:
                        engine = ObservationPUCT(net, plan['device'], seed=case['search_seed'],
                                                 leaf_batch=plan['leaf_batch'])
                        original, chosen, root = engine.search(state, belief, history,
                            simulations=plan['simulations'], temperature=0., add_dirichlet=True,
                            dirichlet_alpha=.3, dirichlet_epsilon=.25, return_root=True)
                        counts, raw, pruned = audited_targets(root, engine.config.c_puct)
                        if root.visits != plan['simulations'] or not np.allclose(original, raw, atol=1e-7):
                            raise ValueError('Actual search visits changed')
                        if chosen not in state.legal_moves():
                            raise ValueError('Original search chose an illegal move')
                        actions = [AZ.jq.format_move_uci(m) for m in root.children]
                        scores = forced_proxy(teacher, histories[actor], fen, actions, plan['teacher_nodes'])
                        ids = [AZ.move_to_index(m) for m in root.children]
                        proxy = np.asarray([scores[a]['proxy'] for a in actions])
                        raw_score, pruned_score = [float(np.dot(p[ids], proxy)) for p in [raw, pruned]]
                        row = {**case, 'completed_simulations': root.visits,
                            'legal_actions': len(actions), 'actual_chosen_move': AZ.jq.format_move_uci(chosen),
                            'search_statistics_unchanged': True,
                            'entropy': {'raw': entropy(raw), 'pruned': entropy(pruned)},
                            'target_total_variation': float(np.abs(raw-pruned).sum()/2),
                            'teacher_forced_nodes_per_action': plan['teacher_nodes'],
                            'expected_teacher_proxy': {'raw': raw_score, 'pruned': pruned_score},
                            'pruned_minus_raw_teacher_proxy': pruned_score-raw_score,
                            'actions': [{'move': a, 'actual_visits': int(counts[i]),
                                'raw_target': float(raw[i]), 'pruned_target': float(pruned[i]),
                                'noised_prior': float(root.children[m].prior),
                                'root_actor_q': -root.children[m].q_value() if counts[i] else None,
                                'teacher_score': scores[a]}
                                for a, m, i in zip(actions, root.children, ids)]}
                        report['cases'].append(row)
                        print(json.dumps({k:v for k,v in row.items() if k != 'actions'}), flush=True)
                    verify()
                    save_json(out / 'report.json', report)
                if ply < max(by_ply):
                    move = tuple(map(int, data['moves'][ply]))
                    undo = AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
                    if (undo.moved_reveal, undo.captured_reveal) != tuple(map(int, data['reveals'][ply])):
                        raise ValueError('Independent environment reveal replay differs')
                    history.append(move)
                    for viewer in (AZ.jq.RED, AZ.jq.BLACK):
                        histories[viewer].append(observer_token(move, undo, viewer))
    verify()
    report['summary'] = paired_game_summary(report['cases'], plan['bootstrap_seed'])
    report['phase_descriptive'] = {phase: paired_game_summary(
        [r for r in report['cases'] if r['phase'] == phase], plan['bootstrap_seed'])
        for phase in sorted({r['phase'] for r in plan['cases']})}
    report['all_planned_cases_accounted'] = len(report['cases']) + len(report['quarantined']) == len(plan['cases'])
    report['eligible_for_controlled_target_study'] = bool(report['summary']['benefit_supported']
        and not report['quarantined'] and report['all_planned_cases_accounted'])
    report['status'] = 'completed'
    save_json(out / 'report.json', report)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fixed training-trajectory diagnostic; hidden labels never enter likelihoods."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import fresh_game, audit_episode, save_json
from research.neural_search import load_model
from research.policy_value import AZ
from research.private_inventory import PrivateInventory
from research.private_inventory_policy import action_likelihoods


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); plan = json.loads(Path(args.plan).read_text())
    def verify():
        for relative, digest in plan['sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Pinned artifact changed: '+relative)
    verify(); out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    net, _ = load_model(ROOT/plan['model'], 'cpu')
    report = {'schema': 1, 'status': 'running', 'plan_sha256': hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
              'scope': plan['scope'], 'games': [], 'samples': [], 'skipped_rule_ban_decisions': 0}
    save_json(out/'report.json', report)
    for episode in plan['episodes']:
        base = ROOT/episode
        with np.load(base.with_suffix('.npz'), allow_pickle=False) as arrays:
            data = {k: arrays[k] for k in arrays.files}
        metadata = json.loads(base.with_suffix('.json').read_text())
        audit = audit_episode(data, metadata)
        state, belief, rng = fresh_game(metadata['seed'])
        posterior = [PrivateInventory(tuple(AZ.jq.POOL_INIT[t] for t in AZ.jq.POOL_TYPES)) for _ in range(2)]
        reference = [PrivateInventory(tuple(AZ.jq.POOL_INIT[t] for t in AZ.jq.POOL_TYPES)) for _ in range(2)]
        # Only evaluation scoring reads this hidden-label accumulator.
        labels = [[0]*6 for _ in range(2)]
        for ply, move_array in enumerate(data['moves']):
            move = tuple(map(int, move_array)); actor = state.side; root = 1-actor
            info = belief.info_state(state, root)
            b, uniform = posterior[root], reference[root]
            if len(b.log_probabilities) > 1 and any(state._banned_for_side()):
                # Original long-check/chase legality probes simulate reveals.
                # Do not let their true-world pool enter a public policy mask.
                report['skipped_rule_ban_decisions'] += 1
            elif len(b.log_probabilities) > 1:
                likelihood = action_likelihoods(net, info, root, b, list(state._iter_legal_moves_fast(actor)), move,
                                               plan['temperature'], plan['epsilon'])
                b.condition_action(likelihood)
                label = tuple(labels[root])
                gain = b.log_probabilities[label]-uniform.log_probabilities[label]
                report['samples'].append({'episode': episode, 'ply': ply, 'root': root, 'support': len(likelihood),
                                          'log_score_gain_nats': gain, **b.entropies()})
            victim = state.board[move[1]]
            dark_capture = bool(victim and AZ.jq.is_dark(victim))
            undo = AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
            if undo.moved_reveal:
                t = list(AZ.jq.POOL_TYPES).index(undo.moved_reveal)
                posterior[actor].public_reveal(t); reference[actor].public_reveal(t)
            if dark_capture:
                posterior[root].hidden_capture(); reference[root].hidden_capture()
                labels[root][list(AZ.jq.POOL_TYPES).index(undo.captured_reveal)] += 1
        report['games'].append({'episode': episode, 'positions': len(data['moves']), 'audit': audit})
        save_json(out/'report.json', report)
        print(json.dumps({'completed_games': len(report['games']), 'scored_decisions': len(report['samples'])}), flush=True)
    verify()
    gains = [s['log_score_gain_nats'] for s in report['samples']]
    report.update(status='completed', summary={'scored_decisions': len(gains),
        'mean_log_score_gain_nats': math.fsum(gains)/len(gains) if gains else None,
        'warning': 'Correlated reused TRAIN trajectories; count-policy approximation to a search/noise actor. No independent calibration or strength claim.'})
    save_json(out/'report.json', report)


if __name__ == '__main__': main()

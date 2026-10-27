"""Exhaustive compatible-inventory root-mask audit of completed TRAIN episodes."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.policy_value import AZ
from research.private_inventory import PrivateInventory
from research.neural_replay import load_episode, fresh_game, save_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); a = p.parse_args()
    path = Path(a.plan); plan = json.loads(path.read_text()); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    def verify():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Plan changed')
        for name, h in plan['sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != h: raise ValueError('Frozen input changed: '+name)
    verify(); out = ROOT/plan['out']; out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'status': 'running', 'scope': plan['scope'], 'plan_sha256': digest,
              'datasets': [], 'counterexamples': []}
    save_json(out/'report.json', report)
    for dataset in plan['datasets']:
        summary = {'label': dataset['label'], 'games': 0, 'positions': 0, 'ban_positions': 0,
                   'ban_with_own_live_unknown_inventory': 0, 'compatible_worlds_checked': 0,
                   'inventory_sensitive_masks': 0, 'chosen_action_not_universally_legal': 0}
        folder = ROOT/dataset['episodes']
        for index in dataset['indices']:
            data, meta = load_episode(folder/f'game-{index:06d}.npz')
            state, belief, rng = fresh_game(meta['seed'])
            for ply, raw_move in enumerate(data['moves']):
                summary['positions'] += 1; actor = state.side; move = tuple(map(int, raw_move))
                ban = tuple(state._banned_for_side())
                if any(ban):
                    summary['ban_positions'] += 1
                    info = belief.info_state(state, actor)
                    if info.unknown_removals[actor] > 0 and info.pool_total_known[actor] > 0:
                        summary['ban_with_own_live_unknown_inventory'] += 1
                        if info.unknown_removals[1-actor] != 0: raise ValueError('Unexpected uncertain enemy inventory')
                        inventory = PrivateInventory(tuple(info.pool_known[actor][t] for t in AZ.jq.POOL_TYPES),
                                                     info.unknown_removals[actor])
                        masks = {}; intersection = None; union = set()
                        for h in inventory.log_probabilities:
                            pools = [row.copy() for row in info.pool_known]
                            for t, n in zip(AZ.jq.POOL_TYPES, h): pools[actor][t] -= n
                            missing = info.unknown_removals.copy(); missing[actor] = 0
                            world = replace(info, pool_known=pools, unknown_removals=missing).determinize(random.Random(0))
                            mask = frozenset(world.legal_moves()); union |= mask
                            intersection = set(mask) if intersection is None else intersection & mask
                            masks.setdefault(mask, []).append(h); summary['compatible_worlds_checked'] += 1
                        true_mask = frozenset(state.legal_moves())
                        if true_mask not in masks: raise ValueError('Actual world not represented by compatible inventories')
                        if len(masks) > 1:
                            summary['inventory_sensitive_masks'] += 1
                            not_universal = move not in intersection
                            summary['chosen_action_not_universally_legal'] += not_universal
                            report['counterexamples'].append({'dataset': dataset['label'], 'episode': index, 'ply': ply,
                                'ban_check_chase': ban, 'actor': actor, 'own_public_counts': inventory.public,
                                'unknown_captured_count': inventory.captured, 'live_dark_count': inventory.live,
                                'selected_move': AZ.jq.format_move_uci(move), 'chosen_action_not_universally_legal': not_universal,
                                'robust_actions': [AZ.jq.format_move_uci(m) for m in sorted(intersection)],
                                'possible_actions': [AZ.jq.format_move_uci(m) for m in sorted(union)],
                                'world_groups': [{'private_captured_counts': hs,
                                                 'legal_moves': [AZ.jq.format_move_uci(m) for m in sorted(mask)]}
                                                for mask, hs in masks.items()]})
                AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
            summary['games'] += 1
            if summary['games']%32 == 0: print(json.dumps(summary), flush=True)
        report['datasets'].append(summary); verify(); save_json(out/'report.json', report)
    verify(); report['status'] = 'completed'; save_json(out/'report.json', report)


if __name__ == '__main__': main()

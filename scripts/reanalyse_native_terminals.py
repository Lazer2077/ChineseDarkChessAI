"""Apply a frozen public native-terminal transform to all existing source games."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from research.neural_replay import load_episode, sha, save_json
from audit_spatial_source_rules import (audit_game, public_start, NativeRules,
    Color, Kind, LETTERS, AZ)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    out = ROOT / plan['out']
    if out.exists():
        raise ValueError('Refuse to overwrite terminal reanalysis')
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen reanalysis input changed: ' + name)
    rules = NativeRules(ROOT / plan['library'])
    rows = []
    for index in range(plan['games']):
        file = ROOT / plan['episodes'] / f'game-{index:06d}.npz'
        data, meta = load_episode(file)
        row = {'index': index, 'source_npz_sha256': sha(file),
               **audit_game(rules, data, meta, plan['native_no_capture_limit'])}
        row['accepted'] = (row['native_red_return'] is not None and row['illegal_chosen_move'] is None
            and row['native_illegal_visit_roots'] == 0 and row['counter_mismatch_roots'] == 0)
        if row['accepted']:
            n = row['native_terminal_ply']
            moves = [AZ.jq.format_move_uci(tuple(map(int, m))) for m in data['moves'][:n]]
            reveals = [Kind(LETTERS[int(r)]) if r else None for r in data['reveals'][:n, 0]]
            # Check the same full public prefix through the separate one-shot API.
            status = rules.game_replay(public_start(), Color.RED, moves, reveals,
                                       no_capture_limit=plan['native_no_capture_limit'])
            if status.name != row['native_status']:
                raise ValueError('Incremental and one-shot native terminals disagree')
            repetition = rules.repetition_replay(public_start(), Color.RED, moves, reveals)
            row['native_repetition_diagnostic'] = {'status': repetition.status.name,
                'begin': repetition.begin, 'end': repetition.end,
                'red_perpetual_check': repetition.red_perpetual_check,
                'black_perpetual_check': repetition.black_perpetual_check}
        rows.append(row)
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Reanalysis input changed while replaying: ' + name)
    save_json(out, {'schema': 1, 'status': 'completed', 'plan_sha256': sha(path),
        'scope': plan['scope'], 'games': rows,
        'accepted_games': sum(r['accepted'] for r in rows),
        'quarantined_games': sum(not r['accepted'] for r in rows),
        'accepted_prefix_positions': sum(r['native_terminal_ply'] for r in rows if r['accepted']),
        'earlier_terminal_games': sum(r['source_positions_after_native_terminal'] > 0 for r in rows),
        'native_terminal_value_disagreements': sum(r['terminal_value_matches_source'] is False for r in rows),
        'limitations': plan['limitations']})
    print(json.dumps({'status': 'completed', 'accepted': sum(r['accepted'] for r in rows),
                      'quarantined': sum(not r['accepted'] for r in rows)}), flush=True)


if __name__ == '__main__':
    main()

"""Check paired public prefixes and actual exposure of the fully-public switch."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from research.fixed_work_contract import INITIAL_BOARD


def events_and_boundary(game):
    board = list(INITIAL_BOARD); events = []; first_full = None
    for ply, row in enumerate(game['moves']):
        if first_full is None and not ('X' in board or 'x' in board): first_full = ply
        events.append((row['m'], row['r'], row['c']))
        a, b = row['m'][:2], row['m'][2:]
        frm = int(a[1])*9+ord(a[0])-97; to = int(b[1])*9+ord(b[0])-97
        pc = board[frm]
        if pc in 'Xx':
            if row['r'] is None: raise ValueError('Missing public reveal')
            pc = row['r'] if pc.isupper() else row['r'].lower()
        board[frm] = '.'; board[to] = pc
    return events, first_full


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); path = Path(args.report); report = json.loads(path.read_text())['reports'][0]
    raw = gzip.decompress((ROOT/report['raw_games']['path']).read_bytes())
    pairs = defaultdict(list)
    for line in raw.splitlines():
        game = json.loads(line); pairs[game['pair']].append(game)
    result = {'schema': 1, 'scope': 'Post-match phase-exposure and prefix analysis on already audited paired games; not a new hypothesis test or independent strength evidence.',
              'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
              'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'pairs': []}
    for index, games in sorted(pairs.items()):
        if len(games) != 2 or {g['red'] for g in games} != {'candidate', 'baseline'} or len({g['seed'] for g in games}) != 1:
            raise ValueError('Not a complete color-swapped physical layout pair')
        (ea, ba), (eb, bb) = [events_and_boundary(g) for g in games]
        prefix = min(ba if ba is not None else len(ea), bb if bb is not None else len(eb))
        if ea[:prefix] != eb[:prefix]: raise ValueError('Actors diverged before the fully-public boundary')
        if ba != bb or (ba is None and ea != eb): raise ValueError('Unmatched public transition/terminal before switch')
        score = sum(g['result'] if g['red'] == 'candidate' else 1-g['result'] for g in games)/2
        result['pairs'].append({'pair': index, 'identical_prefix_plies': prefix, 'first_full_ply': ba,
                                'post_boundary_behavior_diverged': ea != eb, 'candidate_pair_score': score})
    rows = result['pairs']
    result['summary'] = {'pairs': len(rows), 'all_pre_boundary_prefixes_identical': True,
        'pairs_entering_fully_public': sum(r['first_full_ply'] is not None for r in rows),
        'pairs_with_behavior_divergence': sum(r['post_boundary_behavior_diverged'] for r in rows),
        'pairs_scoring_above_half': sum(r['candidate_pair_score'] > .5 for r in rows),
        'pairs_scoring_below_half': sum(r['candidate_pair_score'] < .5 for r in rows)}
    target = Path(args.out)
    if target.exists(): raise ValueError('Refuse to overwrite phase analysis')
    target.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result['summary']))


if __name__ == '__main__': main()

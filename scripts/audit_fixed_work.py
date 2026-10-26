"""Join public native replay, actual go timings, and completed-simulation witnesses."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'external/devinokeefe-jieqi'))
from research.fixed_work_contract import INITIAL_BOARD, public_key, validate_record
from jieqi.native import NativeRules, NativeGame, GameStatus
from jieqi.setup import Color, Kind


def audit(run, witness_dir, simulations, candidate_strategy, baseline_strategy):
    manifest = json.loads((run/'manifest.json').read_text())
    summary = json.loads((run/'cdca-summary.json').read_text())
    games_raw = (run/'games.jsonl').read_bytes(); games = [json.loads(line) for line in games_raw.splitlines()]
    if manifest['status'] != 'completed' or len(games) != manifest['args']['games'] or not summary['valid_technical_run']:
        raise ValueError('Require a complete technically valid native match')
    timing_raw = (run/'timing.jsonl').read_bytes()
    timing = [json.loads(line) for line in timing_raw.splitlines()]
    grouped = defaultdict(list)
    for row in timing: grouped[row['pid']].append(row)
    if len(grouped) != 2*len(games): raise ValueError('Require two fresh actor processes per game')
    strategies = {'candidate': candidate_strategy, 'baseline': baseline_strategy}
    actual = Counter(); methods = Counter(); archived = []; warmups = 0
    for pid, calls in grouped.items():
        rows = [json.loads(line) for line in (witness_dir/f'{pid}.jsonl').read_text().splitlines()]
        if len(rows) != len(calls) or not calls[0]['warmup'] or any(c['warmup'] for c in calls[1:]):
            raise ValueError('Every actual go, including exactly one initial warmup, must have a witness')
        roles = {c['engine'] for c in calls}
        if len(roles) != 1: raise ValueError('Actor changed identity')
        role = roles.pop()
        if role not in strategies: raise ValueError('Unknown actor role')
        for ordinal, (row, call) in enumerate(zip(rows, calls), 1):
            if row['pid'] != pid or row['ordinal'] != ordinal or row['ply'] != call['ply'] or row['outer_go'] != call['go']:
                raise ValueError('Simulation witness does not match actual timed request')
            method = validate_record(row, simulations, strategies[role])
            archived.append({'engine': role, 'warmup': call['warmup'], **row})
            if call['warmup']: warmups += 1; continue
            actual[(role, row['ply'], row['public_key'], row['selected_move'], row['public_board'], row['actor'])] += 1
            methods[(role, method)] += 1
    rules = NativeRules(ROOT/'external/devinokeefe-jieqi/build/release/lib/libjieqi_rules.so')
    expected = Counter(); public_moves = 0; full_moves = 0
    for game in games:
        history = []
        with NativeGame(rules, INITIAL_BOARD, Color.RED) as native:
            for ply, move in enumerate(game['moves']):
                snap = native.snapshot()
                if snap.status != GameStatus.ONGOING: raise ValueError('Public continuation after terminal')
                actor = 'r' if snap.turn == Color.RED else 'b'
                role = game['red'] if actor == 'r' else game['black']
                key = public_key(snap.board, actor, history)
                expected[(role, ply, key, move['m'], snap.board, actor)] += 1
                public_moves += 1; full_moves += not ('X' in snap.board or 'x' in snap.board)
                native.advance(move['m'], Kind(move['r']) if move['r'] else None)
                history.append(move['m'])
            if native.snapshot().status == GameStatus.ONGOING: raise ValueError('Missing native terminal')
    if expected != actual: raise ValueError('Missing, duplicated, or wrong-observation/action simulation witness')
    raw = ''.join(json.dumps(row, separators=(',', ':'))+'\n' for row in archived).encode()
    digest = hashlib.sha256(raw).hexdigest(); archive = ROOT/'results/raw'/f'fixed-work-{digest[:16]}.jsonl.gz'
    if archive.exists():
        if gzip.decompress(archive.read_bytes()) != raw: raise ValueError('Archive digest collision')
    else: archive.parent.mkdir(parents=True, exist_ok=True); archive.write_bytes(gzip.compress(raw, mtime=0))
    return {'schema': 1, 'games': len(games), 'processes': len(grouped), 'warmup_calls': warmups,
            'public_moves_replayed': public_moves, 'fully_public_moves': full_moves, 'simulations_each': simulations,
            'complete_simulation_witnesses': True, 'public_phase_gate_verified': True,
            'method_counts': [{'engine': role, 'method': method, 'moves': n} for (role, method), n in sorted(methods.items())],
            'raw_games_sha256': hashlib.sha256(games_raw).hexdigest(), 'timing_sha256': hashlib.sha256(timing_raw).hexdigest(),
            'witness_archive': str(archive.relative_to(ROOT)), 'witness_uncompressed_sha256': digest,
            'scope': 'Completed root backups, normalized actual visits, selected actions and public-only phase gates match every native move and actual go request. Both players use equal simulation counts, not necessarily equal FLOPs/NN evaluations. Native seeded-identity/legal/terminal/actual-clock audit is separate. This is not playing-strength evidence.'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True); p.add_argument('--witness-dir', required=True)
    p.add_argument('--simulations', required=True, type=int); p.add_argument('--candidate-strategy', required=True)
    p.add_argument('--baseline-strategy', default='puct'); p.add_argument('--out', required=True); a = p.parse_args()
    out = Path(a.out)
    if out.exists(): raise ValueError('Refuse to overwrite a simulation audit')
    report = audit(Path(a.run), Path(a.witness_dir), a.simulations, a.candidate_strategy, a.baseline_strategy)
    out.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))


if __name__ == '__main__': main()

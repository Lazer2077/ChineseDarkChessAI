"""Describe live-dark exposure in an already completed, independently audited match."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    report = json.loads(Path(args.report).read_text())
    audit = json.loads(Path(args.audit).read_text())
    if len(report['reports']) != 1:
        raise ValueError('Require exactly one completed match')
    match = report['reports'][0]
    result = match['result']
    raw_spec = match['raw_games']
    raw = gzip.decompress(Path(raw_spec['path']).read_bytes())
    if sha(raw) != raw_spec['uncompressed_sha256'] or sha(raw) != audit['raw_games_sha256']:
        raise ValueError('Raw match/audit digests differ')
    if (not result['valid_technical_run'] or audit['technical_games'] or
            audit['clock_ambiguous_moves_from_rounding'] or
            not all(audit[k] for k in ['seeded_identities_match', 'all_recorded_moves_legal',
                                      'nontechnical_terminal_results_match'])):
        raise ValueError('Require independently clean completed match')
    initial = ({f + str(r) for r in [0, 9] for f in 'abcdfghi'} |
               {f + str(r) for r in [2, 7] for f in 'bh'} |
               {f + str(r) for r in [3, 6] for f in 'acegi'})
    assert len(initial) == 30
    rows = []
    totals = {actor: collections.Counter() for actor in ['candidate', 'baseline']}
    for line in raw.splitlines():
        game = json.loads(line)
        dark = set(initial)
        phases = collections.Counter()
        for ply, move in enumerate(game['moves']):
            count = len(dark)
            phase = 'D0' if count == 0 else 'D1-4' if count <= 4 else 'D5-14' if count <= 14 else 'D15-30'
            actor = game['red'] if ply % 2 == 0 else game['black']
            totals[actor][phase] += 1
            phases[phase] += 1
            src, dst = move['m'][:2], move['m'][2:4]
            if (src in dark) != (move['r'] is not None):
                raise ValueError('Reveal event disagrees with dark-square accounting')
            if src in dark:
                dark.remove(src)
            if dst in dark:
                if move['c'] is None:
                    raise ValueError('Dark capture lacks independently recorded identity')
                dark.remove(dst)
        rows.append({'pair': game['pair'], 'red': game['red'], 'plies': game['plies'],
                     'terminal_live_dark': len(dark), 'pre_action_phase_counts': dict(phases)})
    if (len(rows) != result['games'] or len(rows) != audit['games'] or
            sum(sum(c.values()) for c in totals.values()) != audit['moves_replayed']):
        raise ValueError('Incomplete audited move coverage')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Refuse to overwrite phase diagnostic')
    data = {'schema': 1, 'status': 'completed', 'source_report': args.report,
            'source_report_sha256': sha(Path(args.report).read_bytes()),
            'source_audit': args.audit, 'source_audit_sha256': sha(Path(args.audit).read_bytes()),
            'raw_games': raw_spec, 'script_sha256': sha(Path(__file__).read_bytes()),
            'scope': 'Post-hoc descriptive exposure on this already completed old-checkpoint match. No new games, checkpoint selection, training, causal search-versus-value attribution or independent confirmation.',
            'games': len(rows), 'wins_draws_losses': [result[k] for k in ['wins', 'draws', 'losses']],
            'games_with_a_fully_public_pre_action_root': sum(r['pre_action_phase_counts'].get('D0', 0) > 0 for r in rows),
            'games_ending_without_live_darks': sum(r['terminal_live_dark'] == 0 for r in rows),
            'median_terminal_live_dark': statistics.median(r['terminal_live_dark'] for r in rows),
            'pre_action_phase_counts_by_actor': {k: dict(v) for k, v in totals.items()},
            'per_game': rows,
            'limitations': 'D0 measures physical live-dark identity uncertainty only. It does not prove equality of private historical information or opponent beliefs. Existing records do not contain completed neural simulation counts, so wall times alone cannot identify a search-throughput cause.'}
    out.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({k: data[k] for k in ['games', 'games_with_a_fully_public_pre_action_root',
                                         'median_terminal_live_dark', 'pre_action_phase_counts_by_actor']}))


if __name__ == '__main__':
    main()

"""Pair-level diagnostic intervals for predeclared per-game endpoint log gain."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def analyze(report, seed=20261102, replicates=20000):
    if report['status'] != 'completed': raise ValueError('Require completed calibration')
    endpoints = []; pairs = {}
    for i, game in enumerate(report['games']):
        rows = [r for r in report['samples'] if r['game_index'] == i]
        if len(rows) != game['scored_decisions']: raise ValueError('Missing score records')
        endpoint = rows[-1]['log_score_gain_nats'] if rows else 0.
        endpoints.append({'game_index': i, 'pair': game['pair'], 'red': game['red'], 'gain_nats': endpoint})
        pairs.setdefault(game['pair'], []).append((game['red'], endpoint))
    if any(len(rows) != 2 or {c for c, _ in rows} != {'candidate', 'baseline'} for rows in pairs.values()):
        raise ValueError('Require whole color-swapped layout pairs')
    gains = [sum(v for _, v in rows)/2 for rows in pairs.values()]
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(gains) for _ in gains)/len(gains) for _ in range(replicates))
    def quantile(p):
        index = p*(len(samples)-1); lo = int(index); hi = min(lo+1, len(samples)-1)
        return samples[lo]+(index-lo)*(samples[hi]-samples[lo])
    return {'games': len(endpoints), 'pairs': len(gains), 'per_game_endpoints': endpoints,
            'mean_endpoint_log_gain_nats': sum(gains)/len(gains),
            'paired_bootstrap_95': [quantile(.025), quantile(.975)], 'bootstrap_seed': seed, 'replicates': replicates,
            'warning': 'Belief-label log score on reused sparse trajectories, not win rate, independent calibration assurance, or game-strength evidence.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); path = Path(args.report); report = json.loads(path.read_text())
    result = {'schema': 1, 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **analyze(report)}
    target = Path(args.out)
    if target.exists(): raise ValueError('Refuse to overwrite analysis')
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_game_endpoints'}))


if __name__ == '__main__': main()

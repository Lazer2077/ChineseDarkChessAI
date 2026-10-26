"""Archive a completed original-rule Mist screen; never infer promotion from a screen."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--out', required=True)
p.add_argument('--raw', required=True)
a = p.parse_args()
run = Path(a.run)
manifest = json.loads((run / 'manifest.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
raw = (run / 'games.jsonl').read_bytes()
games = [json.loads(line) for line in raw.splitlines()]
voidpath = run / 'void.jsonl'
voids = voidpath.read_text().splitlines() if voidpath.exists() else []
pairs = collections.defaultdict(list)
scores = []
for g in games:
    score = g['result'] if g['red'] == 'candidate' else 1 - g['result']
    scores.append(score)
    pairs[g['pair']].append((g['red'], score))
assert len(games) + len(voids) == manifest['games']
assert summary['wins'] == scores.count(1) and summary['draws'] == scores.count(.5) and summary['losses'] == scores.count(0)
complete = [statistics.mean(s for _, s in rows) for rows in pairs.values()
            if len(rows) == 2 and {red for red, _ in rows} == {'candidate', 'mist-native'}]
ci = None
bounded = None
if complete:
    radius = math.sqrt(math.log(40) / (2 * len(complete)))
    bounded = [max(0., statistics.mean(complete) - radius), min(1., statistics.mean(complete) + radius)]
if len(complete) >= 20:
    rng = random.Random(20261026)
    boot = sorted(statistics.mean(rng.choices(complete, k=len(complete))) for _ in range(10000))
    ci = [boot[249], boot[9749]]
summary.update(complete_pairs=len(complete), paired_bootstrap_95=ci, paired_hoeffding_95=bounded,
               single_match_superiority_gate=False, promotion_supported=False)
target = Path(a.raw)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(gzip.compress(raw, mtime=0))
report = {'manifest': manifest, 'summary': summary, 'raw_games': str(target),
          'raw_sha256': hashlib.sha256(raw).hexdigest(),
          'note': 'Development screen with budget+1000ms quarantine tolerance; not a strict-clock confirmation. Fewer than 20 pairs: bootstrap not reported.' if len(complete) < 20 else 'Development screen only; not a strict-clock confirmation.'}
Path(a.out).write_text(json.dumps(report, indent=2) + '\n')

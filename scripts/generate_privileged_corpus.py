"""Four isolated, frozen fixed-deal rollout workers; merge only complete shards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import save_json, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); args = p.parse_args()
    path = Path(args.plan); plan = json.loads(path.read_text()); digest = sha(path)
    out = ROOT/plan['out']
    def verify():
        if sha(path) != digest: raise ValueError('Corpus plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT/name) != expected: raise ValueError('Corpus input changed: '+name)
    verify(); out.mkdir(parents=True, exist_ok=False)
    state = {'schema': 1, 'status': 'running', 'pid': os.getpid(),
             'plan_sha256': digest, 'episodes': [], 'workers': []}
    children = []; logs = []
    try:
        for i in range(plan['workers']):
            childplan = dict(plan, out=plan['out']+'/shard-'+str(i),
                             seed=plan['seed']+i*plan['games_per_worker'], games=plan['games_per_worker'])
            childpath = out/('shard-'+str(i)+'-plan.json'); save_json(childpath, childplan)
            log = (out/('shard-'+str(i)+'.log')).open('x'); logs.append(log)
            child = subprocess.Popen([sys.executable, 'scripts/privileged_data_preflight.py',
                                      '--plan', str(childpath)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            children.append(child); state['workers'].append({'shard': i, 'pid': child.pid})
        save_json(out/'manifest.json', state)
        while any(c.poll() is None for c in children):
            verify()
            if any(c.poll() not in (None, 0) for c in children): raise RuntimeError('A corpus worker failed')
            time.sleep(1)
        if any(c.returncode != 0 for c in children): raise RuntimeError('A corpus worker failed')
        for i in range(plan['workers']):
            folder = out/('shard-'+str(i)); report = json.loads((folder/'report.json').read_text())
            if report['status'] != 'completed' or len(report['games']) != plan['games_per_worker']:
                raise ValueError('Incomplete corpus shard')
            for row in report['games']:
                index = i*plan['games_per_worker']+row['index']
                file = folder/f"game-{row['index']:06d}.npz"
                if sha(file) != row['npz_sha256'] or row['seed'] != plan['seed']+index:
                    raise ValueError('Corpus file/seed mismatch')
                state['episodes'].append(dict(row, index=index, file=str(file.relative_to(ROOT))))
        if len(state['episodes']) != plan['workers']*plan['games_per_worker']:
            raise ValueError('Wrong corpus size')
        verify(); state['status'] = 'completed'; save_json(out/'manifest.json', state)
    except BaseException as error:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children: child.wait()
        state.update(status='failed', error=str(error)); save_json(out/'manifest.json', state); raise
    finally:
        for log in logs: log.close()


if __name__ == '__main__': main()

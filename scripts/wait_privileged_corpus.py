"""Wait for the same live fixed-deal corpus controller; never restart it."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);args=p.parse_args()
    path=ROOT/args.plan;digest=sha(path);plan=json.loads(path.read_text())
    while True:
        if sha(path)!=digest:raise ValueError('Corpus plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Corpus source changed: '+name)
        state=json.loads((ROOT/plan['out']/'manifest.json').read_text())
        if state['plan_sha256']!=digest:raise ValueError('Wrong corpus recipe')
        if state['status']=='completed':return
        if state['status']!='running':raise ValueError('Corpus failed; never restart here')
        cmd=Path('/proc')/str(state['pid'])/'cmdline'
        if not cmd.exists() or b'scripts/generate_privileged_corpus.py' not in cmd.read_bytes():
            raise ValueError('Corpus controller missing; never restart here')
        time.sleep(15)


if __name__=='__main__':main()

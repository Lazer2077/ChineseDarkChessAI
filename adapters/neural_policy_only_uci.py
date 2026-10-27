"""Reuse strict observer/UCI parsing, with explicit no-search identification."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.policy_only import GreedyPolicy


class Output:
    def __init__(self, stream): self.stream = stream
    def write(self, text):
        if text.startswith('id name '): text = 'id name CDCA greedy masked warm policy (zero tree search)'
        elif text.startswith('info string batch deadline sha256 '): text = text.replace('batch deadline', 'policy-only source', 1)
        elif text.startswith('info string {'):
            record = json.loads(text[len('info string '):])
            if record.get('simulations') != 0: raise ValueError('Policy-only actor must not report tree simulations')
            record['selection'] = 'greedy legal-normalized policy, one inference'
            record['policy_mass'] = record.pop('visit_mass')
            text = 'info string '+json.dumps(record)
        return self.stream.write(text)
    def flush(self): return self.stream.flush()


if __name__ == '__main__':
    protocol.ObservationPUCT = GreedyPolicy
    protocol.TIMED_SEARCH_SHA256 = hashlib.sha256((ROOT/'research/policy_only.py').read_bytes()).hexdigest()
    sys.stdout = Output(sys.stdout)
    protocol.main()

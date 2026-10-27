"""Two complete frozen networks; original observer, search and clock protocol."""
import hashlib
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.factorial_network import FactorialPUCT


class ConfiguredFactorialPUCT(FactorialPUCT):
    def __init__(self, net, device='cpu', seed=20261029, leaf_batch=32):
        super().__init__(net, device, seed, leaf_batch,
                         value_directory=ROOT/os.environ['JIEQI_FACTORIAL_VALUE'])


class TaggedOutput:
    def __init__(self, stream): self.stream = stream
    def write(self, text):
        return self.stream.write(text.replace(
            'CDCA residual policy/value information-set PUCT (research)',
            'CDCA two-network policy/value factorial PUCT (research)'))
    def flush(self): self.stream.flush()


if __name__ == '__main__':
    protocol.ObservationPUCT = ConfiguredFactorialPUCT
    protocol.TIMED_SEARCH_SHA256 = hashlib.sha256((protocol.TIMED_SEARCH_SHA256+
        hashlib.sha256((ROOT/'research/factorial_network.py').read_bytes()).hexdigest()).encode()).hexdigest()
    sys.stdout = TaggedOutput(sys.stdout)
    protocol.main()

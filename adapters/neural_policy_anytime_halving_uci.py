"""Process-local timed hybrid actor; original observer/UCI parser unchanged."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.anytime_halving import AnytimeHalvingPUCT, SEARCH_SHA256


class Actor(AnytimeHalvingPUCT):
    def search(self, *args, **kwargs):
        result = super().search(*args, **kwargs)
        print('info string root-anytime '+json.dumps(self.last_diagnostics), flush=True)
        return result


class Output:
    def __init__(self, stream): self.stream = stream
    def write(self, text):
        if text.startswith('id name '): text = 'id name CDCA timed root halving then survivor PUCT (research)'
        return self.stream.write(text)
    def flush(self): return self.stream.flush()


if __name__ == '__main__':
    protocol.ObservationPUCT = Actor; protocol.TIMED_SEARCH_SHA256 = SEARCH_SHA256
    sys.stdout = Output(sys.stdout); protocol.main()

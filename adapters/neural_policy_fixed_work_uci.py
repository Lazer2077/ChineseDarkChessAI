"""Research-only equal-simulation wrapper; actual outer clocks remain charged."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.fixed_work_profile import Settings, FixedInput, constructor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--strategy', required=True, choices=['puct', 'halving', 'public-halving'])
    parser.add_argument('--simulations', type=int, default=256)
    args, rest = parser.parse_known_args()
    directory = os.environ.get('JIEQI_FIXED_WORK_WITNESS_DIR')
    if not directory: raise ValueError('Fixed-work experiments require a simulation witness directory')
    settings = Settings(args.strategy, args.simulations, directory)
    sys.argv = [sys.argv[0], *rest]
    protocol.ObservationPUCT = constructor(settings)
    protocol.TIMED_SEARCH_SHA256 = hashlib.sha256((protocol.TIMED_SEARCH_SHA256+
        hashlib.sha256((ROOT/'research/fixed_work_profile.py').read_bytes()).hexdigest()+args.strategy+str(args.simulations)).encode()).hexdigest()
    sys.stdin = FixedInput(sys.stdin, settings)
    try: protocol.main()
    finally: settings.close()


if __name__ == '__main__': main()

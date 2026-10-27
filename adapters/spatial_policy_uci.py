"""Original observer/search protocol with a safe tagged policy-decoder loader."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.spatial_policy_value import load_model


if __name__=='__main__':
    protocol.load_model=load_model
    protocol.TIMED_SEARCH_SHA256=hashlib.sha256((protocol.TIMED_SEARCH_SHA256+
        hashlib.sha256((ROOT/'research/spatial_policy_value.py').read_bytes()).hexdigest()).encode()).hexdigest()
    protocol.main()

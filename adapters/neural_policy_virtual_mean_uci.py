"""Process-local constructor substitution for the virtual-mean research actor."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.virtual_mean import VirtualMeanPUCT,SEARCH_SHA256

if __name__=='__main__':
    protocol.ObservationPUCT=VirtualMeanPUCT
    protocol.TIMED_SEARCH_SHA256=SEARCH_SHA256
    protocol.main()

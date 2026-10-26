"""Process-local teacher-value substitution; unchanged original UCI/history code."""
import hashlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'adapters'))
import neural_policy_uci as protocol
from research.teacher_leaf import TeacherLeafPUCT

if __name__ == '__main__':
    protocol.ObservationPUCT = TeacherLeafPUCT
    protocol.TIMED_SEARCH_SHA256 = hashlib.sha256((protocol.TIMED_SEARCH_SHA256+
        hashlib.sha256((ROOT/'research/teacher_leaf.py').read_bytes()).hexdigest()).encode()).hexdigest()
    protocol.main()

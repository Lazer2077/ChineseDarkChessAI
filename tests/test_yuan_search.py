"""The clock bridge must preserve the pinned search when no deadline applies."""
import random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from yuan_observer import reconstruct
from yuan_uci import timed_search_function

def tree_summary(tree):
    return tree.visits,tree.value,[(child.move,child.visits,child.value) for child in tree.children]

def test_fixed_iteration_search_matches_original_with_private_uncertainty():
    original,timed,namespace,_=timed_search_function()
    for tokens in [[],['b2b9R']]:
        outputs=[]
        for function in [original,timed]:
            state,belief,_=reconstruct(tokens)
            move,tree=function(state,belief,iterations=16,rng=random.Random(53))
            outputs.append((move,tree_summary(tree)))
        assert outputs[0]==outputs[1]

def test_expired_deadline_finishes_one_real_iteration_instead_of_random_fallback():
    original,timed,namespace,_=timed_search_function()
    namespace['_clock_deadline']=0
    state,belief,_=reconstruct([])
    move,tree=timed(state,belief,iterations=1000000,rng=random.Random(53))
    state,belief,_=reconstruct([])
    expected,reference=original(state,belief,iterations=1,rng=random.Random(53))
    assert move==expected and tree.visits==1
    assert tree_summary(tree)==tree_summary(reference)

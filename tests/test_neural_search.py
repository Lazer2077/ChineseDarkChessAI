import copy,random,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.policy_value import AZ,PhasePolicyValueNet
from research.neural_search import ObservationPUCT
from yuan_observer import reconstruct

def root_summary(root):
    return root.visits,root.value_sum,[(move,child.visits,child.value_sum) for move,child in root.children.items()]

def searcher(seed=53):
    torch.set_num_threads(1);torch.manual_seed(7)
    net=PhasePolicyValueNet(8,1);net.eval()
    return ObservationPUCT(net,seed=seed,leaf_batch=2)

def test_batch_deadline_preserves_fixed_simulation_search():
    engine=searcher();state,belief,_=reconstruct([])
    first=AZ.NeuralMCTS.search(engine,state,belief,[],temperature=1e-6,simulations=4,return_root=True)
    engine=searcher();state,belief,_=reconstruct([])
    second=engine.search(state,belief,[],temperature=1e-6,simulations=4,return_root=True)
    assert first[:2]==second[:2] and root_summary(first[2])==root_summary(second[2])

def test_actual_opponent_private_slots_never_affect_root_search():
    outputs=[]
    for poisoned in [False,True]:
        engine=searcher();state,belief,_=reconstruct(['b2b9R'])
        if poisoned:
            other=1-state.side
            belief.pool_counts[other]=[[0]*8,[0]*8]
            belief.pool_total[other]=[0,0]
            belief.unknown_removals[other]=[99,99]
        policy,move,root=engine.search(state,belief,[],temperature=1e-6,simulations=4,return_root=True)
        outputs.append((policy,move,root_summary(root)))
    assert outputs[0]==outputs[1]

def test_expired_budget_keeps_a_complete_batch_and_normalized_visit_target():
    engine=searcher();engine.deadline=0;state,belief,_=reconstruct([])
    policy,move,root=engine.search(state,belief,[],temperature=1e-6,simulations=1000000,return_root=True)
    assert root.visits==2 and move in state.legal_moves()
    assert abs(sum(policy)-1)<1e-7 and min(policy)>=0

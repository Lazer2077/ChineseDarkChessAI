import copy,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.policy_value import AZ,PhasePolicyValueNet
from research.neural_search import ObservationPUCT
from research.public_continuation import PublicContinuationPUCT,public_continuation_key
from yuan_observer import reconstruct

def full_info():
    jq=AZ.jq;state=jq.GameState.empty(with_pools=False)
    for square,kind,color in [(85,jq.TYPE_KING,jq.RED),(4,jq.TYPE_KING,jq.BLACK),(40,jq.TYPE_PAWN,jq.RED)]:
        state.board[square]=jq.make_piece(kind,color,0)
        if kind==jq.TYPE_KING:state.king_pos[color]=square
    state.position_key=state._position_key();state.position_counts={state.position_key:1}
    return AZ.im.init_belief_for_game(state).info_state(state,state.side)

def test_full_public_key_ignores_dead_inventory_but_original_key_does_not():
    info=full_info();other=copy.deepcopy(info);other.pool_known[info.side][AZ.jq.TYPE_ROOK]=1;other.pool_total_known[info.side]=0;other.unknown_removals[info.side]=1
    assert AZ.im.information_set_key(info,(),())!=AZ.im.information_set_key(other,(),((1,2),))
    assert public_continuation_key(info,())==public_continuation_key(other,())

@pytest.mark.parametrize('field',['no_capture_plies','consecutive_checks','check_anchor','chase_streak','chase_target','chase_target_code','chase_anchor','position_counts','rules'])
def test_public_rule_memory_and_configuration_never_merge(field):
    info=full_info();other=copy.deepcopy(info)
    if field=='position_counts':other.position_counts[tuple(other.position_counts)[0]]=2
    elif field=='rules':
        import dataclasses
        other.rules=dataclasses.replace(other.rules,no_capture_plies=79)
    elif field=='no_capture_plies':other.no_capture_plies+=1
    else:getattr(other,field)[0]+=1
    assert public_continuation_key(info,())!=public_continuation_key(other,())
    assert public_continuation_key(info,())!=public_continuation_key(info,(((1,2),0),))

def test_dark_or_missing_history_never_uses_public_abstraction():
    state,belief,_=reconstruct([]);assert public_continuation_key(belief.info_state(state,state.side),()) is None
    info=full_info();info.repetition_history_complete=False
    assert public_continuation_key(info,()) is None
    info.repetition_history_complete=True;info.position_counts=None
    assert public_continuation_key(info,()) is None

def test_profile_selection_distinguishes_own_private_bag_from_opponent_bag():
    import numpy as np
    from research.policy_value import REC
    from scripts.profile_public_continuation import classify_observation
    rec=np.zeros(1,dtype=REC);rec['board'][0,0]=8;rec['urn'][0,0,:2]=1
    assert classify_observation(rec.view(np.uint8))==(1,True)
    rec['stm'][0]=1
    assert classify_observation(rec.view(np.uint8))==(1,False)
    rec['stm'][0]=0;rec['urn'][0,0]=0;rec['urn'][0,1,:2]=1
    assert classify_observation(rec.view(np.uint8))==(1,False)

def test_instrumented_original_key_mode_matches_unmodified_search():
    torch.set_num_threads(1);torch.manual_seed(14);net=PhasePolicyValueNet(8,1);net.eval()
    state,belief,_=reconstruct([])
    first=ObservationPUCT(net,seed=61,leaf_batch=2).search(state,belief,[],temperature=1e-6,simulations=16,return_root=True)
    engine=PublicContinuationPUCT(net,seed=61,leaf_batch=2,share_public=False)
    second=engine.search(state,belief,[],temperature=1e-6,simulations=16,return_root=True)
    assert first[:2]==second[:2]
    assert [(m,c.visits,c.value_sum) for m,c in first[2].children.items()]==[(m,c.visits,c.value_sum) for m,c in second[2].children.items()]
    assert second[2].visits==16 and abs(sum(second[0])-1)<1e-7

def test_legal_private_capture_permutations_share_only_after_public_continuation():
    jq=AZ.jq;outputs=[]
    for captures in [(jq.TYPE_ROOK,jq.TYPE_PAWN),(jq.TYPE_PAWN,jq.TYPE_ROOK)]:
        state=jq.GameState.empty(with_pools=False)
        pieces=[(85,jq.TYPE_KING,jq.RED,0),(4,jq.TYPE_KING,jq.BLACK,0),
            (40,jq.TYPE_PAWN,jq.RED,0),(83,jq.TYPE_ELEPHANT,jq.RED,0),(87,jq.TYPE_ELEPHANT,jq.RED,0),
            (81,jq.TYPE_ROOK,jq.RED,1),(89,jq.TYPE_ROOK,jq.RED,1),
            (63,jq.TYPE_ROOK,jq.BLACK,0),(71,jq.TYPE_ROOK,jq.BLACK,0)]
        for square,kind,color,dark in pieces:
            state.board[square]=jq.make_piece(kind,color,dark)
            if kind==jq.TYPE_KING:state.king_pos[color]=square
        state.pool[jq.RED][jq.TYPE_ROOK]=state.pool[jq.RED][jq.TYPE_PAWN]=1
        state.pool_total[jq.RED]=2;state.position_key=state._position_key();state.position_counts={state.position_key:1}
        belief=AZ.im.init_belief_for_game(state);public=[];private=[[],[]]
        for move,captured in [((85,84),0),((63,81),captures[0]),((84,85),0),((71,89),captures[1]),((85,84),0)]:
            assert move in state.legal_moves()
            undo=AZ.im.apply_move_and_update_belief(state,belief,move,capture_reveal_choice=captured or None)
            AZ.im.record_observation(move,undo,public,private)
            if move==(63,81):assert public_continuation_key(state,public) is None
        assert state.game_outcome()[0] is None
        outputs.append((belief.information_set_key(state,state.side,public,private[state.side]),
            public_continuation_key(state,public),state.legal_moves(),state.game_outcome()))
    assert outputs[0][0]!=outputs[1][0]
    assert outputs[0][1] is not None and outputs[0][1:]==outputs[1][1:]

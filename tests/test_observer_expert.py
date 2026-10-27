from pathlib import Path
from types import SimpleNamespace
import pytest
from research.policy_value import AZ
from research.privileged_training import fresh_fixed_game
from research.observer_expert import observer_token,canonical_observer_fen,ObserverHistoryExpert
from research.teacher_leaf import observer_fen

ROOT=Path(__file__).resolve().parents[1]


def test_token_never_reads_opponent_private_capture_identity():
    class Event:
        prev_side=AZ.jq.RED
        moved_reveal=AZ.jq.TYPE_ROOK
        captured_before=AZ.jq.make_piece(AZ.jq.TYPE_KNIGHT,AZ.jq.BLACK,1)
        @property
        def captured_reveal(self):raise AssertionError('Opponent private identity accessed')
    move=(AZ.jq.name_to_square('b2'),AZ.jq.name_to_square('b9'))
    assert observer_token(move,Event(),AZ.jq.BLACK)=='b2b9R'
    own=SimpleNamespace(prev_side=AZ.jq.RED,moved_reveal=AZ.jq.TYPE_ROOK,
        captured_before=Event.captured_before,captured_reveal=AZ.jq.TYPE_KNIGHT)
    assert observer_token(move,own,AZ.jq.RED)=='b2b9Rn'


def test_fen_pool_order_is_not_information_difference():
    assert canonical_observer_fen('9/9/9/9/9/9/9/9/9/9 w R1P2n1 0 1')==canonical_observer_fen('9/9/9/9/9/9/9/9/9/9 w P2R1n1 0 1')


def test_original_native_echo_matches_both_observer_views_after_private_capture():
    binary=ROOT/'external/devinokeefe-jieqi/engine/build/jqe'
    if not binary.exists():pytest.skip('Pinned native build unavailable')
    state,belief,rng=fresh_fixed_game(1005);move=(AZ.jq.name_to_square('b2'),AZ.jq.name_to_square('b9'))
    undo=AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
    with ObserverHistoryExpert(binary,ROOT/'external/devinokeefe-jieqi/engine/nets/it014.jqn',depth=1) as expert:
        for viewer in (AZ.jq.BLACK,AZ.jq.RED):
            fen=observer_fen(belief.info_state(state,viewer))
            expert.position_snapshot([observer_token(move,undo,viewer)],fen)
        viewer=state.side;fen=observer_fen(belief.info_state(state,viewer))
        result=expert.analyse([observer_token(move,undo,viewer)],fen,128)
        assert result['bestmove'] in result['legal_moves']
        with pytest.raises(ValueError,match='rejected'):
            expert.position_snapshot(['a0a9'],fen)

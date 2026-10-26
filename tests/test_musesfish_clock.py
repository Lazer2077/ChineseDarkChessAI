import pytest
from adapters.musesfish_clock import START_FEN,move_budget,position_ply


def test_actor_clock_selection_and_budget_bounds():
    command='go wtime 10000 btime 4000 winc 100 binc 200'
    assert move_budget(command,0)==475
    assert move_budget(command,1)==310
    assert move_budget('go wtime 100 winc 2000',0)==25
    assert move_budget('go wtime 0 winc 0',0)==1
    assert move_budget('go movetime 120',1)==120


def test_entire_strict_history_is_counted_without_interpreting_private_suffixes():
    history='a0a1R a9a8r a1a8p'
    assert position_ply('position startpos moves '+history)==3
    assert position_ply('position fen '+START_FEN+' moves '+history)==3
    assert position_ply('position startpos')==0
    assert position_ply('position fen '+START_FEN)==0


@pytest.mark.parametrize('command',['go nodes 500','go infinite','go wtime nan',
    'go wtime -1','go movetime','go wtime inf'])
def test_unsupported_or_invalid_budget_does_not_silently_become_one_second(command):
    with pytest.raises(ValueError):move_budget(command,0)


def test_arbitrary_fen_is_not_silently_replaced_by_standard_start():
    with pytest.raises(ValueError):position_ply('position fen other')

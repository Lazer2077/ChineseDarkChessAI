import pytest
from scripts.audit_outer_clocks import certify


def fixture(red_ms=99.,black_ms=99.):
    games=[{'red':'candidate','black':'baseline','moves':[{},{}]}]
    rows=[]
    for pid,label,ply,elapsed in [(1,'candidate',0,red_ms),(2,'baseline',1,black_ms)]:
        rows.append(dict(pid=pid,engine=label,ply=0,go='movetime 1',elapsed_ms=9999.,warmup=True))
        rows.append(dict(pid=pid,engine=label,ply=ply,go='wtime 100 btime 100 winc 10 binc 10',elapsed_ms=elapsed,warmup=False))
    return games,rows


def test_safe_outer_intervals_certify_without_game_pid_assignment():
    games,rows=fixture();r=certify(games,rows,100,10)
    assert r['certified_outer_wall_clock_safe'] and r['calls_checked']==2
    assert r['minimum_outer_bank_lower_ms']==pytest.approx(.999)


def test_guard_and_negative_bound_cannot_be_ignored():
    games,rows=fixture(red_ms=100.)
    assert not certify(games,rows,100,10)['certified_outer_wall_clock_safe']
    with pytest.raises(ValueError):certify(games,rows,100,10,guard_ms=0)


def test_duplicate_missing_or_reused_lifetimes_are_rejected():
    games,rows=fixture()
    for bad in [rows[:-1],rows+[rows[-1]],rows+[rows[0]]]:
        with pytest.raises(ValueError):certify(games,bad,100,10)


def test_increments_are_not_granted_before_the_move():
    games,rows=fixture(red_ms=101.)
    assert not certify(games,rows,100,10)['certified_outer_wall_clock_safe']

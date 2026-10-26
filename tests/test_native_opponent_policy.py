import copy
import pytest
from pathlib import Path
from research.native_opponent_policy import hypothetical_fen, score_policy, FixedDepthOpponent

ROOT = Path(__file__).resolve().parents[1]
BOARD = 'XXXXKXXXX'+'.'*10+'X'+'.'*5+'X'+'.'+'X.X.X.X.X'+'.'*18+'x.x.x.x.x'+'.'+'x'+'.'*5+'x'+'.'*10+'xxxxkxxxx'


def test_masked_score_policy_is_normalized_robust_and_shift_invariant():
    scores = {'a0a1': -10000, 'a0a2': 2, 'a0a3': 10000}
    p = score_policy(scores)
    assert sum(p.values()) == pytest.approx(1.) and min(p.values()) >= .1/3
    assert p == pytest.approx(score_policy({a: s+140 for a, s in scores.items()}))
    with pytest.raises(ValueError): score_policy(scores, temperature=0)


def test_hypothetical_fen_uses_public_pools_and_count_hypothesis_only():
    pools = [[2, 2, 2, 2, 2, 5] for _ in range(2)]; before = copy.deepcopy(pools)
    fen = hypothetical_fen(BOARD, 0, pools, 1, (1, 0, 0, 0, 0, 0), 3)
    assert fen.startswith('xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w ')
    assert 'R2N2B2A2C2P5r1n2b2a2c2p5 3 1' in fen
    assert pools == before


def test_real_forced_search_completes_exact_depth_and_is_cold_hash_reproducible():
    binary = ROOT/'external/devinokeefe-jieqi/engine/build/jqe'
    network = ROOT/'external/devinokeefe-jieqi/engine/nets/it014.jqn'
    fen = hypothetical_fen(BOARD, 0, [[2, 2, 2, 2, 2, 5] for _ in range(2)], 1, (0,)*6, 0)
    with FixedDepthOpponent(binary, network, depth=2) as model:
        first = model.scores(fen, ['a3a4', 'c3c4'])
        second = model.scores(fen, ['c3c4', 'a3a4'])
    assert first == second and set(first) == {'a3a4', 'c3c4'}

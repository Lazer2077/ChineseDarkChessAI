import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_replay_weights import replay_weights

def test_equal_game_mass_despite_unequal_game_lengths():
    games=np.array([0,0,0,1]);phases=np.array([0,0,1,1])
    w=replay_weights(games,phases,'game')
    np.testing.assert_allclose(w[:3].sum(),w[3])
    np.testing.assert_allclose(w.mean(),1)

def test_equal_present_phase_mass_within_each_game():
    games=np.array([0,0,0,1]);phases=np.array([0,0,1,1])
    w=replay_weights(games,phases,'game-phase')
    np.testing.assert_allclose(w[:2].sum(),w[2])
    np.testing.assert_allclose(w[:3].sum(),w[3])
    # Missing phases neither invent data nor remove games with only one phase.
    np.testing.assert_allclose(w,[.5,.5,1,2])

def test_uniform_control_and_weighted_objective_identity():
    games=np.array([5,5,5,9]);phases=np.array([0,0,1,1]);errors=np.array([1.,3.,7.,11.])
    np.testing.assert_array_equal(replay_weights(games,phases,'position'),np.ones(4))
    w=replay_weights(games,phases,'game-phase')
    expected=(((1+3)/2+7)/2+11)/2
    np.testing.assert_allclose(np.mean(w*errors),expected)

def test_global_phase_balancing_differs_from_per_game_phase_balancing():
    games=np.array([0,0,0,1]);phases=np.array([0,0,1,1])
    w=replay_weights(games,phases,'phase')
    np.testing.assert_allclose(w[:2].sum(),w[2:].sum())
    assert not np.allclose(w,replay_weights(games,phases,'game-phase'))

import copy
import numpy as np
import torch
import pytest
from research.policy_value import AZ, PhasePolicyValueNet
from research.neural_search import ObservationPUCT
from research.privileged_training import fresh_fixed_game, dark_truth, actor_projection, training_pair


def test_ex_ante_deal_consistency_reveals_and_dark_capture():
    state, belief, rng = fresh_fixed_game(991)
    again, _, _ = fresh_fixed_game(991)
    assert state.dark_types == again.dark_types
    assert np.array_equal(dark_truth(state), dark_truth(again))
    move = tuple(AZ.jq.name_to_square(s) for s in ('b2', 'b9'))
    assert move in state.legal_moves()
    expected_move, expected_victim = (state.dark_types[sq] for sq in move)
    undo = AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
    assert undo.moved_reveal == expected_move and undo.captured_reveal == expected_victim
    labels = dark_truth(state)
    assert not labels[move[0]] and not labels[move[1]]
    assert np.count_nonzero(labels) == 28


def test_counterfactual_deals_change_only_training_labels_and_not_actor_search():
    torch.set_num_threads(1); torch.manual_seed(992)
    state, belief, _ = fresh_fixed_game(992)
    other = state.clone(); other.dark_types = dict(state.dark_types)
    squares = [sq for sq, pc in enumerate(state.board) if pc and AZ.jq.is_dark(pc)
               and AZ.jq.piece_color(pc) == state.side]
    a = squares[0]; b = next(sq for sq in squares if other.dark_types[sq] != other.dark_types[a])
    other.dark_types[a], other.dark_types[b] = other.dark_types[b], other.dark_types[a]
    raw, truth = training_pair(state, belief); raw2, truth2 = training_pair(other, belief)
    assert np.array_equal(raw, raw2) and not np.array_equal(truth, truth2)
    net = PhasePolicyValueNet(4, 1).eval(); outputs = []
    for world in (state, other):
        projected = actor_projection(world)
        assert projected.dark_types is None and world.dark_types is not None
        engine = ObservationPUCT(copy.deepcopy(net), seed=993, leaf_batch=2)
        pi, move, root = engine.search(projected, belief, [], simulations=16,
                                      temperature=1e-6, return_root=True)
        outputs.append((pi, move, root.value_sum,
                        [(m, c.visits, c.value_sum) for m, c in root.children.items()]))
    assert outputs[0] == outputs[1]


def test_refuse_lazy_truth_and_remove_privilege_after_public_collapse():
    state, _, _ = fresh_fixed_game(994)
    lazy = state.clone(); lazy.dark_types = None
    with pytest.raises(ValueError, match='ex ante'): dark_truth(lazy)
    for sq, pc in enumerate(state.board):
        if pc and AZ.jq.is_dark(pc):
            state.board[sq] = AZ.jq.make_piece(state.dark_types[sq], AZ.jq.piece_color(pc), 0)
    state.pool = [[0]*8, [0]*8]; state.pool_total = [0, 0]
    assert not dark_truth(state).any()

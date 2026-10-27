import math
import numpy as np
import pytest
from research.observation_uncertainty import live_assignment_entropy,record_uncertainty
from research.privileged_training import fresh_fixed_game,training_pair
from research.policy_value import AZ


def test_exact_sequence_entropy_after_unknown_loss():
    # With2A/1B and one unknown loss, live sequences AA/AB/BA each have1/3.
    assert live_assignment_entropy((2,1),1)==pytest.approx(math.log(3))
    assert live_assignment_entropy((2,1),3)==pytest.approx(0.)
    assert live_assignment_entropy((2,0),1)==pytest.approx(0.)


def test_initial_entropy_and_label_or_truth_isolation():
    state,belief,_=fresh_fixed_game(1001);raw,truth=training_pair(state,belief)
    first=record_uncertainty(raw[None])
    assert first['normalized_entropy'][0]==pytest.approx(1.)
    assert first['live_dark_count'][0]==30 and not first['fully_public'][0]
    poisoned=raw.copy();poisoned[103:111]=255
    second=record_uncertainty(poisoned[None])
    assert np.array_equal(first['entropy_nats'],second['entropy_nats'])
    other=state.clone();other.dark_types=dict(state.dark_types)
    squares=[sq for sq,pc in enumerate(state.board) if pc and AZ.jq.is_dark(pc)
             and AZ.jq.piece_color(pc)==state.side]
    a=squares[0];b=next(sq for sq in squares if other.dark_types[sq]!=other.dark_types[a])
    other.dark_types[a],other.dark_types[b]=other.dark_types[b],other.dark_types[a]
    other_raw,other_truth=training_pair(other,belief)
    assert not np.array_equal(truth,other_truth)
    assert np.array_equal(first['entropy_nats'],record_uncertainty(other_raw[None])['entropy_nats'])


def test_private_certainty_is_not_public_reveal_and_invalid_counts_reject():
    state,belief,_=fresh_fixed_game(1002);raw,_=training_pair(state,belief)
    certain=raw.copy();certain[:90]=0;certain[0]=8
    certain[91:103]=0;certain[91]=1 # One red dark whose only possible type is rook.
    value=record_uncertainty(certain[None])
    assert value['type_certain'][0] and not value['fully_public'][0]
    public=certain.copy();public[0]=2;public[91]=0
    assert record_uncertainty(public[None])['fully_public'][0]
    bad=certain.copy();bad[91]=0
    with pytest.raises(ValueError,match='cover'):record_uncertainty(bad[None])

import math
import random
import pytest
from research.private_inventory import PrivateInventory


def test_uniform_updates_equal_fresh_hypergeometric_prior():
    b = PrivateInventory((2, 1, 2))
    b.hidden_capture(); b.public_reveal(0); b.hidden_capture()
    reference = PrivateInventory((1, 1, 2), 2)
    assert b.probabilities == pytest.approx(reference.probabilities)
    assert b.uniform


def test_action_information_changes_future_reveal_and_is_not_truth_input():
    b = PrivateInventory((1, 1), 1)
    b.condition_action({(1, 0): .8, (0, 1): .2})
    assert b.predictive() == pytest.approx((.2, .8))
    assert not b.uniform
    samples = [b.sample(random.Random(i)) for i in range(4000)]
    assert abs(samples.count((1, 0))/4000-.8) < .025
    b.public_reveal(1)
    assert b.probabilities == {(1, 0): 1.}
    assert b.live == 0
    assert b.entropies() == {'private_ledger_nats': 0., 'live_assignment_nats': 0.}


def test_unknown_capture_cannot_change_uniform_next_reveal_law():
    b = PrivateInventory((2, 2, 1), 2)
    for draws in [(1, 0, 0), (2, 0, 0), (1, 1, 1)]:
        expected = math.prod(math.factorial(c)/math.factorial(c-n) for c, n in zip((2, 2, 1), draws))
        expected /= math.factorial(5)/math.factorial(5-sum(draws))
        assert math.exp(b.log_sequence_probability(draws)) == pytest.approx(expected)
    before = b.predictive(); b.hidden_capture()
    assert b.predictive() == pytest.approx(before)


def test_sequence_ratio_equals_conditioned_predictive():
    b = PrivateInventory((2, 2), 1)
    b.condition_action({h: .1+.3*h[0] for h in b.probabilities})
    first = b.log_sequence_probability((1, 0))
    ratio = math.exp(b.log_sequence_probability((1, 1))-first)
    b.public_reveal(0)
    assert b.predictive()[1] == pytest.approx(ratio)
    assert sum(b.predictive()) == pytest.approx(1.)


def test_impossible_observation_is_transactional_and_bad_inputs_rejected():
    b = PrivateInventory((1, 1), 1)
    before = b.log_probabilities
    with pytest.raises(ValueError): b.condition_action({h: 0. for h in before})
    assert b.log_probabilities == before and b.uniform
    with pytest.raises(ValueError): PrivateInventory((2., 1))
    with pytest.raises(ValueError): PrivateInventory((16,))
    with pytest.raises(ValueError): b.public_reveal(-1)


def test_rare_supported_state_survives_underflow_and_recovers_after_reveal():
    b = PrivateInventory((1, 1), 1)
    for _ in range(1000): b.condition_action({(1, 0): math.exp(-10), (0, 1): 1.})
    assert b.probabilities[(1, 0)] == 0. and (1, 0) in b.log_probabilities
    b.public_reveal(1)
    assert b.probabilities == {(1, 0): 1.}


def test_uncertainty_of_private_ledger_can_increase_then_vanish():
    b = PrivateInventory((1, 1))
    assert b.entropies()['private_ledger_nats'] == 0.
    b.hidden_capture()
    assert b.entropies()['private_ledger_nats'] == pytest.approx(math.log(2))
    b.hidden_capture()
    assert b.entropies()['live_assignment_nats'] == 0.

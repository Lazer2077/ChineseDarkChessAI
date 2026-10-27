import copy
import numpy as np
import pytest
from research.policy_value import AZ
from research.policy_target_pruning import audited_targets, root_statistics, paired_game_summary


def root_with(visits, priors, root_q):
    root = AZ.PUCTNode(1., AZ.jq.RED)
    for move, n, p, q in zip(AZ.LEGAL_MOVE_TABLE, visits, priors, root_q):
        child = AZ.PUCTNode(p, AZ.jq.BLACK)
        child.visits, child.value_sum = n, -q * n
        root.children[move] = child
    root.visits = sum(visits)
    return root


def test_prunes_low_value_tail_without_mutating_actual_work_or_inventing_support():
    root = root_with([100, 2, 0], [.8, .1, .1], [.8, -.8, 0.])
    before = copy.deepcopy(root_statistics(root))
    counts, raw, pruned = audited_targets(root, 1.5)
    ids = [AZ.move_to_index(m) for m in root.children]
    assert root_statistics(root) == before
    assert counts.sum() == 102 and raw[ids[1]] > 0
    assert pruned[ids[1]] == 0 and pruned[ids[0]] == 1 and pruned[ids[2]] == 0


def test_tied_most_visited_moves_protected_and_incomplete_work_rejected():
    root = root_with([5, 5], [.5, .5], [.9, -.9])
    _, raw, pruned = audited_targets(root, 1.5)
    assert np.array_equal(raw, pruned)
    root.visits += 1
    with pytest.raises(ValueError, match='completed child visits'):
        audited_targets(root, 1.5)


def test_proxy_summary_weights_games_and_rejects_tiny_sample_superiority():
    rows = [{'game': 0, 'pruned_minus_raw_teacher_proxy': 1.}] * 10
    rows += [{'game': 1, 'pruned_minus_raw_teacher_proxy': -1.}]
    result = paired_game_summary(rows, 19)
    assert result['game_mean_pruned_minus_raw_teacher_proxy'] == 0
    assert not result['benefit_supported']
    assert not paired_game_summary(rows[:1], 19)['benefit_supported']

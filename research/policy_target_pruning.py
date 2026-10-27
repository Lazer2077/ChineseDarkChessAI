"""Audit the unchanged upstream pruning heuristic without rewriting visits."""
import math
import numpy as np
from research.policy_value import AZ


def root_statistics(root):
    return (root.visits, root.value_sum, tuple(
        (move, child.visits, child.value_sum, child.prior)
        for move, child in root.children.items()))


def audited_targets(root, c_puct):
    if not root.children or root.visits <= 0 or not math.isfinite(c_puct) or c_puct <= 0:
        raise ValueError('Require a complete positive-work root')
    before = root_statistics(root)
    counts = np.zeros(AZ.MOVE_SIZE, dtype=np.int64)
    for move, child in root.children.items():
        if (child.visits < 0 or int(child.visits) != child.visits or
                not math.isfinite(child.value_sum) or
                not math.isfinite(child.prior) or child.prior < 0):
            raise ValueError('Invalid root child statistics')
        counts[AZ.move_to_index(move)] = child.visits
    if counts.sum() != root.visits:
        raise ValueError('Root visits do not match completed child visits')
    raw = counts.astype(np.float64) / root.visits
    pruned = AZ.pruned_policy_from_root(root, c_puct).numpy().astype(np.float64)
    if root_statistics(root) != before:
        raise ValueError('Target transform changed search statistics')
    if (not np.isfinite(pruned).all() or (pruned < 0).any() or
            not np.isclose(pruned.sum(), 1, atol=1e-6) or
            np.any(pruned[counts == 0] != 0)):
        raise ValueError('Pruned target is invalid or invents visit support')
    return counts, raw, pruned


def entropy(policy):
    p = np.asarray(policy)
    positive = p > 0
    return float(-(p[positive] * np.log(p[positive])).sum())


def paired_game_summary(rows, seed, resamples=10000):
    games = sorted({r['game'] for r in rows})
    if not games:
        return {'games': 0, 'roots': 0, 'benefit_supported': False}
    differences = np.asarray([np.mean([
        r['pruned_minus_raw_teacher_proxy'] for r in rows if r['game'] == game])
        for game in games])
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(len(games), size=(resamples, len(games)))].mean(1)
    interval = np.quantile(boot, [.025, .975]).tolist()
    return {'games': len(games), 'roots': len(rows),
            'game_mean_pruned_minus_raw_teacher_proxy': float(differences.mean()),
            'paired_game_bootstrap_95': interval, 'resamples': resamples,
            'benefit_supported': bool(len(games) >= 8 and interval[0] > 0),
            'scope': 'Descriptive TRAIN-root teacher-search proxy, not true values or strength.'}

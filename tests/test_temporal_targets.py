import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('temporal', Path(__file__).resolve().parents[1] / 'scripts/build_temporal_targets.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TemporalTargets(unittest.TestCase):
    def records(self):
        r = np.zeros(8, dtype=[('ply', 'u2'), ('stm', 'u1'), ('score', 'i2'), ('result', 'i1')])
        r['ply'] = [0, 1, 2, 3, 0, 1, 2, 3]
        r['stm'] = r['ply'] % 2
        r['result'] = [1, -1, 1, -1, -1, 1, -1, 1]
        r['score'] = [0, 10000, 430, -10000, 0, 0, 0, 0]
        return r

    def test_same_observer_bootstrap_and_terminal_boundary(self):
        r = self.records()
        target, has_next = m.temporal_targets(r, 2)
        self.assertAlmostEqual(target[0], .7 * .5 + .15 + .15 / (1 + np.exp(-1)))
        # Opponent private-state scores at odd plies cannot affect red's target.
        changed = r.copy()
        changed['score'][1::2] *= -1
        new, _ = m.temporal_targets(changed, 2)
        np.testing.assert_array_equal(new[::2], target[::2])
        control, _ = m.temporal_targets(r, 2, tau=0)
        np.testing.assert_array_equal(target[~has_next], control[~has_next])
        np.testing.assert_array_equal(has_next, [True, True, False, False, True, True, False, False])

    def test_board_transition_detects_boundary_even_when_ply_is_consecutive(self):
        r = np.zeros(2, dtype=[('ply', 'u2'), ('stm', 'u1'), ('score', 'i2'), ('result', 'i1'), ('board', 'u1', (90,)), ('move', 'u2')])
        r['ply'] = [0, 1]
        r['stm'] = [0, 1]
        r['result'] = [1, -1]
        r['move'][0] = 0 | (1 << 7)
        r['board'][0, 0] = 2
        r['board'][1, 1] = 2
        self.assertEqual(len(m.segment_games(r, 1)[0]), 1)
        r['board'][0, 0] = 8  # Hidden mover reveals as a rook.
        self.assertEqual(len(m.segment_games(r, 1)[0]), 1)
        r['board'][1, 80] = 18  # A new game cannot be this move's successor.
        self.assertEqual(len(m.segment_games(r, 2)[0]), 2)
        with self.assertRaises(ValueError):
            m.segment_games(r, 1)

    def test_inconsistent_or_uncertified_trajectories_refused(self):
        r = self.records()
        with self.assertRaises(ValueError):
            m.segment_games(r, 3)
        r['result'][2] = -1
        with self.assertRaises(ValueError):
            m.segment_games(r, 2)
        with self.assertRaises(ValueError):
            m.temporal_targets(self.records(), 2, tau=2)


if __name__ == '__main__':
    unittest.main()

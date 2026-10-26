import importlib.util
import unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('summarize',Path(__file__).parents[1]/'scripts/summarize.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class StatisticsTests(unittest.TestCase):
    def test_missing_partner_cannot_pass(self):
        a=m.summarize([{'layoutSeed':1,'candidate':'r','result':{'winner':'r'}}],samples=10)
        self.assertFalse(a['all_pairs_complete']);self.assertFalse(a['superiority_gate'])
    def test_truncation_has_bounds_and_no_pair(self):
        a=m.summarize([{'layoutSeed':1,'candidate':'r','result':{'winner':'r'}},{'layoutSeed':1,'candidate':'b','result':{'winner':None,'truncated':True}}],samples=10)
        self.assertEqual(a['score_bounds_including_unfinished'],[.5,1]);self.assertEqual(a['complete_pairs'],0)
    def test_paired_win_loss_scores_half(self):
        a=m.summarize([{'layoutSeed':1,'candidate':'r','result':{'winner':'r'}},{'layoutSeed':1,'candidate':'b','result':{'winner':'r'}}],samples=10)
        self.assertIsNone(a['paired_bootstrap_95']);self.assertEqual(a['completed_score'],.5)
    def test_perfect_pairs_have_non_degenerate_finite_sample_bound(self):
        games=[]
        for seed in range(100):
            for side in ['r','b']:games.append({'layoutSeed':seed,'candidate':side,'result':{'winner':side}})
        a=m.summarize(games,samples=20)
        self.assertEqual(a['paired_bootstrap_95'],[1.,1.])
        self.assertGreater(a['paired_hoeffding_95'][0],.5)
        self.assertLess(a['paired_hoeffding_95'][0],1.)
        self.assertTrue(a['superiority_gate'])
if __name__=='__main__':unittest.main()

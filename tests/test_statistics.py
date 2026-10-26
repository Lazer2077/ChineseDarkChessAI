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
if __name__=='__main__':unittest.main()

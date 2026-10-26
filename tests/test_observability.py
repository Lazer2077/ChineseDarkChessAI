import importlib.util
from pathlib import Path
import collections,itertools,math,unittest
import numpy as np
s=importlib.util.spec_from_file_location('observability',Path(__file__).resolve().parents[1]/'scripts/observability.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Observability(unittest.TestCase):
    def test_exact_joint_entropy_matches_physical_sampling(self):
        for counts in [(2,1),(1,1,1),(2,2)]:
            bag=[i for i,c in enumerate(counts) for _ in range(c)]
            for n in range(len(bag)+1):
                hist=collections.Counter(tuple(bag[j] for j in p) for p in itertools.permutations(range(len(bag)),n))
                total=sum(hist.values())
                expected=-sum((c/total)*math.log2(c/total) for c in hist.values())
                self.assertAlmostEqual(m.identity_entropy(counts,n),expected)
        self.assertEqual(m.identity_entropy((2,0),1),0)
        self.assertEqual(m.identity_entropy((2,2),0),0)
        self.assertEqual(m.identity_entropy((1,1),2),1)
        with self.assertRaises(ValueError):m.identity_entropy((1,1),3)
        with self.assertRaises(ValueError):m.identity_entropy((1.5,1),1)

    def test_public_inventory_does_not_use_captor_private_counts(self):
        r=np.zeros(2,dtype=[('stm','u1'),('urn','i1',(2,6)),('board','u1',(90,)),('move','u2')])
        r['stm']=[0,1]
        r['urn'][:]=[2,2,2,2,2,5]
        r['urn'][0,1,0]=1  # Root's private captured enemy rook, absent from public data.
        r['move'][0]=1<<7
        r['board'][0,0]=8
        r['board'][1,1]=2  # Public red rook reveal.
        r['urn'][1,0,0]=1
        pub,ok=m.public_counts(r,np.array([0]),np.array([2]))
        self.assertTrue(ok.all())
        self.assertEqual(pub[0,1,0],2)
        self.assertEqual(pub[1,0,0],1)
        changed=r.copy();changed['urn'][0,1,0]=0
        pub2,_=m.public_counts(changed,np.array([0]),np.array([2]))
        np.testing.assert_array_equal(pub,pub2)
        singleton,ok=m.public_counts(r[:1],np.array([0]),np.array([1]))
        self.assertFalse(ok[0])

if __name__=='__main__':unittest.main()

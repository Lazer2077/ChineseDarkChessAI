import importlib.util,json,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('continual',Path(__file__).parents[1]/'scripts/continual_learning.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class StateTests(unittest.TestCase):
    def test_partial_records_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'records.bin';p.write_bytes(b'\0'*111)
            with self.assertRaises(ValueError):m.retained_records(p)
            p.write_bytes(b'\0'*224);self.assertEqual(m.retained_records(p),2)
    def test_atomic_checkpoint_is_readable_and_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';m.atomic(p,{'completed':0});m.atomic(p,{'completed':1})
            self.assertEqual(json.loads(p.read_text()),{'completed':1});self.assertFalse(p.with_suffix('.json.tmp').exists())
if __name__=='__main__':unittest.main()

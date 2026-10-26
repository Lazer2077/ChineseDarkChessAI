import gzip,json,subprocess,sys,tempfile,unittest
from pathlib import Path
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/archive_native_results.py'
class ArchiveIntegrity(unittest.TestCase):
 def fixture(self,root,value,status='completed'):
  root.mkdir(parents=True)
  args={'candidate':'candidate-bin','baseline':'baseline-bin','candidate_options':'','stage':'development','games':2}
  (root/'manifest.json').write_text(json.dumps({'status':status,'args':args,'referee_commit':'test','rules':'test','hashes':{},'backbone_disclosure':'test fixture'}))
  (root/'cdca-summary.json').write_text(json.dumps({'games':2,'wins':2 if value else 0,'draws':0,'losses':0 if value else 2}))
  (root/'games.jsonl').write_text('\n'.join(json.dumps(g) for g in [{'red':'candidate','result':value},{'red':'baseline','result':1-value}])+'\n')
 def test_same_basename_preserves_both_different_raw_results(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);paths=[]
   for i,value in enumerate([0,1]):
    source=d/str(i)/'a';self.fixture(source,value)
    report=d/f'{i}.json'
    subprocess.run([sys.executable,str(SCRIPT),'--out',str(report),'--raw-dir',str(d/'raw'),str(source)],check=True,capture_output=True)
    paths.append(Path(json.loads(report.read_text())['reports'][0]['raw_games']['path']))
   self.assertNotEqual(*paths)
   self.assertNotEqual(gzip.decompress(paths[0].read_bytes()),gzip.decompress(paths[1].read_bytes()))
 def test_partial_experiment_is_not_published(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);self.fixture(d/'a',1,'running')
   result=subprocess.run([sys.executable,str(SCRIPT),'--out',str(d/'report.json'),'--raw-dir',str(d/'raw'),str(d/'a')],capture_output=True)
   self.assertNotEqual(result.returncode,0);self.assertFalse((d/'report.json').exists())
if __name__=='__main__':unittest.main()

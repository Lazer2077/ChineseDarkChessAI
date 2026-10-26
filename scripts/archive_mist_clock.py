"""Archive completed strict-clock Mist evidence without relabeling development screens."""
import argparse,gzip,hashlib,json,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
p.add_argument('--raw-dir',required=True);p.add_argument('--node',default='node');a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=Path(a.run);out=Path(a.out)
if out.exists():raise SystemExit('Refuse to overwrite a published report')
manifest=json.loads((run/'manifest.json').read_text())
if manifest.get('status')!='completed' or not manifest.get('tc'):
 raise SystemExit('Require a completed actual-clock match')
raw=(run/'games.jsonl').read_bytes();rows=[json.loads(line) for line in raw.splitlines()]
voidfile=run/'void.jsonl'
if voidfile.exists():rows.extend(json.loads(line) for line in voidfile.read_text().splitlines())
rows.sort(key=lambda row:row['game'])
if [row['game'] for row in rows]!=list(range(manifest['games'])):
 raise SystemExit('Missing or duplicate game attempts')
for row in rows:
 if int(row['seed'])!=manifest['seed']+row['pair'] or row['pair']!=row['game']//2:
  raise SystemExit('Unexpected layout seed or pair')
 if row['red']!=('candidate' if row['game']%2==0 else 'mist-native'):
  raise SystemExit('Unexpected color assignment')
code="""import fs from 'node:fs'; import {pairedEvidence} from './src/paired_evidence.js';
const x=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(pairedEvidence(x.rows,x.games,{confirmation:x.confirmation})));"""
summary=json.loads(subprocess.check_output([a.node,'--input-type=module','-e',code],
 input=json.dumps({'rows':rows,'games':manifest['games'],'confirmation':manifest['stage']=='confirmation'}),text=True,cwd=root))
if summary!=json.loads((run/'summary.json').read_text()):
 raise SystemExit('Raw evidence disagrees with original summary')
digest=hashlib.sha256(raw).hexdigest();target=Path(a.raw_dir)/(run.name+'-'+digest[:12]+'.jsonl.gz')
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists() and gzip.decompress(target.read_bytes())!=raw:raise SystemExit('Artifact collision')
target.write_bytes(gzip.compress(raw,mtime=0))
voids=[row for row in rows if row['result'] is None]
report={'schema':1,'manifest':manifest,'summary':summary,'raw_games':str(target),'raw_sha256':digest,
 'void_attempts':voids,'scope':'Completed actual-clock match, full position-to-bestmove wall time charged with zero grace. Technical outcomes remain in WDL and invalidate acceptance. No global promotion or general learned-improvement claim.'}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')

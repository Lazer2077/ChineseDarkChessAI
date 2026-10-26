"""Publish compact evidence for completed native screens; raw games stay local."""
import argparse, gzip, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--raw-dir');p.add_argument('runs',nargs='+');a=p.parse_args()
def public_path(value):
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return path.name

reports=[]
for name in a.runs:
    root=Path(name);manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'cdca-summary.json').read_text())
    if manifest.get('status') != 'completed':
        raise SystemExit('Only completed experiments can be archived')
    raw_info={}
    if a.raw_dir:
        raw_dir=Path(a.raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
        raw=(root/'games.jsonl').read_bytes()
        games=[json.loads(line) for line in raw.splitlines()]
        scores=[g['result'] if g['red']=='candidate' else 1-g['result'] for g in games]
        if len(games)!=summary['games'] or [scores.count(x) for x in [1,.5,0]] != [summary[k] for k in ['wins','draws','losses']]:
            raise SystemExit('Raw games and report WDL disagree')
        digest=hashlib.sha256(raw).hexdigest()
        target=raw_dir/(root.name+'-'+digest[:12]+'.jsonl.gz')
        if target.exists() and gzip.decompress(target.read_bytes())!=raw:
            raise SystemExit('Raw artifact name collision; refusing overwrite')
        target.write_bytes(gzip.compress(raw,mtime=0))
        raw_info={'path':str(target),'uncompressed_sha256':digest}
    reports.append({'raw_games':raw_info,'run':root.name,'stage':manifest['args']['stage'],'source_referee_commit':manifest['referee_commit'],'rules':manifest['rules'],'budget':{k:manifest['args'][k] for k in ['games','movetime','seed','concurrency','tc'] if k in manifest['args']},'candidate':Path(manifest['args']['candidate']).name,'baseline':public_path(manifest['args']['baseline']),'candidate_options':manifest['args']['candidate_options'],'hashes':manifest['hashes'],'result':summary,'attribution':manifest['backbone_disclosure']})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'schema':1,'scope':'Completed experimental matches with explicit per-report stages; no promoted or defeated opponent. Some candidates use the unchanged published teacher net.','reports':reports},indent=2)+'\n')

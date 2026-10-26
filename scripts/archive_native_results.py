"""Publish compact evidence for completed native screens; raw games stay local."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('runs',nargs='+');a=p.parse_args()
reports=[]
for name in a.runs:
    root=Path(name);manifest=json.loads((root/'manifest.json').read_text());summary=json.loads((root/'cdca-summary.json').read_text())
    reports.append({'run':root.name,'stage':manifest['args']['stage'],'source_referee_commit':manifest['referee_commit'],'rules':manifest['rules'],'budget':{k:manifest['args'][k] for k in ['games','movetime','seed','concurrency']},'candidate':Path(manifest['args']['candidate']).name,'baseline':str(Path(manifest['args']['baseline'])),'candidate_options':manifest['args']['candidate_options'],'hashes':manifest['hashes'],'result':summary,'attribution':manifest['backbone_disclosure']})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'schema':1,'scope':'Development/integration screens; no promoted or defeated opponent. Some candidates use the unchanged published teacher net.','reports':reports},indent=2)+'\n')

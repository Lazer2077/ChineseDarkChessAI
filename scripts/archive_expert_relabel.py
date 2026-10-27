"""Archive only complete, frozen expert annotations and quarantine accounting."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    path=ROOT/a.plan;plan=json.loads(path.read_text());folder=ROOT/plan['out']
    report=json.loads((folder/'report.json').read_text());output=ROOT/a.out
    if output.exists():raise ValueError('Refuse to overwrite annotation report')
    if report['status']!='completed' or report['plan_sha256']!=sha(path):raise ValueError('Annotation not complete/frozen')
    for name,h in plan['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen annotation input changed: '+name)
    entries=report['episodes']
    if [r['game'] for r in entries]!=plan['games']:raise ValueError('Missing/reordered annotation games')
    compatible=[r for r in entries if r['status']=='compatible-roots']
    quarantined=[r for r in entries if r['status']=='quarantined']
    if len(compatible)+len(quarantined)!=len(entries):raise ValueError('Unfinished annotation episode')
    if report['compatible_games']!=len(compatible) or report['quarantined_games']!=len(quarantined):
        raise ValueError('Quarantine accounting differs')
    if report['checked_roots']!=sum(r['roots_checked'] for r in entries):raise ValueError('Root totals differ')
    if any(r['roots_checked']!=r['source_positions'] for r in compatible):raise ValueError('Incomplete compatible episode')
    if 'training_games' in plan:
        splits=plan['training_games']+plan['selection_validation_games']
        if len(set(splits))!=len(splits) or set(splits)!=set(plan['games']):raise ValueError('Invalid annotation splits')
    for row in quarantined:
        if (folder/f"game-{row['game']:06d}-labels.json").exists():raise ValueError('Quarantined prefix retained')
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for row in compatible:
            file=folder/f"game-{row['game']:06d}-labels.json"
            if sha(file)!=row['labels_sha256']:raise ValueError('Expert label digest differs')
            labels=json.loads(file.read_text())
            if labels['source_npz_sha256']!=row['source_npz_sha256'] or len(labels['labels'])!=row['source_positions']:
                raise ValueError('Incomplete or foreign labels')
            b=file.read_bytes();info=tarfile.TarInfo(file.name);info.size=len(b);info.mode=0o644;info.mtime=0
            tar.addfile(info,io.BytesIO(b))
    content=raw.getvalue();digest=hashlib.sha256(content).hexdigest()
    archive=ROOT/'results/raw'/f'observer-expert-labels-{digest[:12]}.tar.gz'
    with archive.open('xb') as handle:
        with gzip.GzipFile(fileobj=handle,mode='wb',filename='',mtime=0) as stream:stream.write(content)
    report['label_archive']={'path':str(archive.relative_to(ROOT)),'uncompressed_sha256':digest,'sha256':sha(archive)}
    if 'training_games' in plan:
        report['split_accounting']={role:{'compatible_games':sum(r['game'] in plan[role] for r in compatible),
            'quarantined_games':sum(r['game'] in plan[role] for r in quarantined),
            'labelled_positions':sum(r['source_positions'] for r in compatible if r['game'] in plan[role])}
            for role in ['training_games','selection_validation_games']}
    save_json(output,report)
    print(json.dumps({'status':'archived','games':len(entries),'compatible_games':len(compatible),
                      'quarantined_games':len(quarantined),'report':a.out}),flush=True)


if __name__=='__main__':main()

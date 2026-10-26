"""Bounded native-teacher experiment; reuses jqe unchanged, never claims it as ours."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--games',type=int,default=512);p.add_argument('--validation-games',type=int,default=128);p.add_argument('--nodes',type=int,default=2000);p.add_argument('--threads',type=int,default=2);p.add_argument('--epochs',type=int,default=10);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi';binary=source/'engine/build/jqe';net=source/'engine/nets/it014.jqn';out=Path(a.out).resolve()
if out.exists():raise SystemExit('Use a new output directory')
sha=subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()
if sha!='331a6e612a038c4a04cf980f7b9482910fd50078':raise SystemExit('Unexpected native teacher commit')
if subprocess.check_output(['git','-C',source,'diff','HEAD'],text=True).strip():raise SystemExit('Teacher source changed')
for x in [binary,net]:
    if not x.is_file():raise SystemExit(f'Missing {x}')
out.mkdir(parents=True)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'source':'https://github.com/devinokeefe/jieqi','commit':sha,'license':'MIT','teacher_sha256':digest(net),'binary_sha256':digest(binary),'args':vars(a),'stage':'starting','note':'Uses external jqe engine, rules and trainer unchanged. New weights trained from scratch on teacher selfplay; no original-engine or strength claim.'}
def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def run(stage,cmd):
    manifest['stage']=stage;manifest.setdefault('commands',[]).append(list(map(str,cmd)));save();print(stage,flush=True)
    with (out/f'{stage}.log').open('w') as log:subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True,cwd=root)
for name,games,seed in [('train-data',a.games,5100001),('validation-data',a.validation_games,5200001)]:
    run(name,[binary,'selfplay','games',games,'nodes',a.nodes,'threads',a.threads,'seed',seed,'random',8,'evalfile',net,'out',out/f'{name}.bin'])
run('train',[sys.executable,source/'engine/train/train.py',out/'train-data.bin','--val-file',out/'validation-data.bin','--out',out/'model','--epochs',a.epochs,'--batch',1024,'--seed',20261026,'--lam',.7])
manifest['stage']='trained';manifest['candidate_sha256']=digest(out/'model/best.jqn');save();print('Native teacher training pilot completed; candidate not promoted.',flush=True)
